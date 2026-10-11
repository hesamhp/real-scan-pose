import { useCallback, useEffect, useRef, useState } from 'react';
import {
  FilesetResolver,
  PoseLandmarker,
  type NormalizedLandmark,
} from '@mediapipe/tasks-vision';
import {
  ANGLE_DEFS,
  IDX,
  angle3D,
  isVis,
  mid,
  tiltFromVertical,
  tiltFromHorizontal,
  torsoTwist,
  type AngleKey,
  type ExerciseKey,
  type Lm,
} from '../lib/pose';

const WASM = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';
const MODEL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';
const GRANTED_KEY = 'posecam.granted';

export type CamState = 'idle' | 'starting' | 'running' | 'error';
export type ModelState = 'loading' | 'ready' | 'error';
export type Facing = 'user' | 'environment';

function camErrMsg(e: unknown): string {
  if (e instanceof Error && e.message === 'INSECURE') return 'دسترسی به دوربین فقط در HTTPS ممکن است.';
  if (e instanceof DOMException) {
    if (e.name === 'NotAllowedError') return 'دسترسی به دوربین رد شد. در تنظیمات مرورگر فعالش کنید.';
    if (e.name === 'NotFoundError') return 'دوربینی پیدا نشد.';
    if (e.name === 'NotReadableError') return 'دوربین توسط برنامه‌ی دیگری در حال استفاده است.';
  }
  return 'دوربین در دسترس نیست.';
}

function frameIsBlack(ctx: CanvasRenderingContext2D, w: number, h: number) {
  try {
    const d = ctx.getImageData(0, 0, w, h).data;
    let s = 0, n = 0;
    for (let i = 0; i + 2 < d.length; i += 97 * 4) { s += d[i] + d[i + 1] + d[i + 2]; n++; }
    return n === 0 || s / (n * 3) < 3.5;
  } catch { return false; }
}

export interface LiveStats {
  fps: number;
  present: boolean;
  tracked: number;
  angles: {
    elbowL: number | null; elbowR: number | null;
    kneeL: number | null; kneeR: number | null;
    hip: number | null; shoulder: number | null;
    neck: number | null; neckR: number | null;
  };
  posture: {
    torso: number | null; shoulderTilt: number | null;
    hipTilt: number | null; twist: number | null;
  };
  seconds: number;
}

const NULL_ANG = { elbowL: null, elbowR: null, kneeL: null, kneeR: null, hip: null, shoulder: null, neck: null, neckR: null };
const NULL_POST = { torso: null, shoulderTilt: null, hipTilt: null, twist: null };
const NULL_STATS: LiveStats = { fps: 0, present: false, tracked: 0, angles: NULL_ANG, posture: NULL_POST, seconds: 0 };

type StableLm = NormalizedLandmark & { hold: number };

function stabilizeLandmarks(
  raw: NormalizedLandmark[] | null,
  prev: StableLm[] | null,
): StableLm[] | null {
  if (!raw) return prev;
  const out: StableLm[] = [];
  for (let i = 0; i < raw.length; i++) {
    const cur = raw[i];
    const p = prev?.[i];
    const vis = cur.visibility ?? 1;
    const good = vis >= 0.38;

    if (!p) {
      out.push({ ...cur, hold: 0 });
      continue;
    }

    if (good) {
      const dx = cur.x - p.x;
      const dy = cur.y - p.y;
      const dz = (cur.z ?? 0) - (p.z ?? 0);
      const dist = Math.hypot(dx, dy, dz);
      const alpha = dist > 0.05 ? 0.58 : dist > 0.025 ? 0.42 : 0.24;
      out.push({
        x: p.x + dx * alpha,
        y: p.y + dy * alpha,
        z: (p.z ?? 0) + dz * alpha,
        visibility: (p.visibility ?? 1) + (vis - (p.visibility ?? 1)) * 0.25,
        hold: 0,
      });
      continue;
    }

    const prevVis = p.visibility ?? 1;
    if (prevVis >= 0.38 && p.hold < 5) {
      out.push({ ...p, visibility: prevVis * 0.92, hold: p.hold + 1 });
    } else {
      out.push({ ...cur, hold: 0 });
    }
  }
  return out;
}

/* ───────────────────────────────────────────────────────────── */
export function usePoseEngine(
  videoRef: React.RefObject<HTMLVideoElement | null>,
  canvasRef: React.RefObject<HTMLCanvasElement | null>,
  _exercise: ExerciseKey,
  soundOn: boolean,
) {
  const [modelState, setModelState] = useState<ModelState>('loading');
  const [camState, setCamState] = useState<CamState>('idle');
  const [camError, setCamError] = useState('');
  const [facing, setFacing] = useState<Facing>('environment');
  const [repFlash] = useState(0);
  const [stats, setStats] = useState<LiveStats>(NULL_STATS);
  const [flipBusy, setFlipBusy] = useState(false);

  const lmRef = useRef<PoseLandmarker | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const rafRef = useRef(0);
  const runRef = useRef(false);
  const lastTRef = useRef(-1);
  const facingRef = useRef<Facing>('environment');
  const sessRef = useRef({ start: 0, carry: 0 });
  const audioRef = useRef<AudioContext | null>(null);
  const soundRef = useRef(soundOn);
  const fpsRef = useRef<number[]>([]);
  const autoRef = useRef(false);
  const liveRef = useRef<LiveStats>({ ...NULL_STATS });
  const frameRef = useRef<{ lms: NormalizedLandmark[] | null; vw: number; vh: number } | null>(null);
  const snapRef = useRef<HTMLCanvasElement | null>(null);
  const snapDirty = useRef(0);
  const smoothA = useRef<Record<AngleKey, number | null>>({ ...NULL_ANG });
  const smoothP = useRef<LiveStats['posture']>({ ...NULL_POST });
  const stableLmRef = useRef<StableLm[] | null>(null);

  soundRef.current = soundOn;

  /* ── model ── */
  useEffect(() => {
    let dead = false;
    (async () => {
      try {
        const v = await FilesetResolver.forVisionTasks(WASM);
        const o = (d: 'GPU' | 'CPU') => ({ baseOptions: { modelAssetPath: MODEL, delegate: d }, runningMode: 'VIDEO' as const, numPoses: 1, minPoseDetectionConfidence: 0.5, minPosePresenceConfidence: 0.5, minTrackingConfidence: 0.5 });
        let lm: PoseLandmarker;
        try { lm = await PoseLandmarker.createFromOptions(v, o('GPU')); } catch { lm = await PoseLandmarker.createFromOptions(v, o('CPU')); }
        if (dead) { lm.close(); return; }
        lmRef.current = lm;
        setModelState('ready');
      } catch { if (!dead) setModelState('error'); }
    })();
    return () => { dead = true; };
  }, []);

  /* ── beep ── */
  const beep = useCallback((freq = 760, dur = 0.09, gain = 0.05) => {
    if (!soundRef.current) return;
    try {
      const c = (audioRef.current ??= new AudioContext());
      if (c.state === 'suspended') void c.resume();
      const o = c.createOscillator(), g = c.createGain();
      o.type = 'sine'; o.frequency.value = freq;
      g.gain.setValueAtTime(gain, c.currentTime);
      g.gain.exponentialRampToValueAtTime(0.0001, c.currentTime + dur);
      o.connect(g).connect(c.destination); o.start(); o.stop(c.currentTime + dur + 0.02);
    } catch { /* */ }
  }, []);

  /* ── process ── */
  const processFrame = useCallback((lms: NormalizedLandmark[] | null, vw: number, vh: number) => {
    const L = liveRef.current;
    const vis = lms ? lms.reduce((n, l) => n + (isVis(l) ? 1 : 0), 0) : 0;
    L.tracked = vis; L.present = vis >= 16;
    if (!lms || !L.present) { L.angles = { ...NULL_ANG }; L.posture = { ...NULL_POST }; return; }

    const sm = smoothA.current;
    const out = { ...NULL_ANG } as LiveStats['angles'];
    (Object.keys(ANGLE_DEFS) as AngleKey[]).forEach(k => {
      const [a, b, c] = ANGLE_DEFS[k];
      const A = lms[a], B = lms[b], C = lms[c];
      let raw: number | null = null;
      if (isVis(A, 0.5) && isVis(B, 0.5) && isVis(C, 0.5)) { const v = angle3D(A, B, C, vw, vh); if (Number.isFinite(v)) raw = v; }
      if (raw === null) { sm[k] = null; } else { sm[k] = sm[k] === null ? raw : sm[k]! + 0.35 * (raw - sm[k]!); (out as any)[k] = Math.round(sm[k]!); }
    });
    L.angles = out;

    const sp = smoothP.current;
    const ease = (k: keyof LiveStats['posture'], raw: number | null) => {
      if (raw === null || !Number.isFinite(raw)) { sp[k] = null; return null; }
      sp[k] = sp[k] === null ? raw : sp[k]! + 0.3 * (raw - sp[k]!);
      return Math.round(sp[k]!);
    };
    const SL = lms[IDX.shoulderL], SR = lms[IDX.shoulderR], HL = lms[IDX.hipL], HR = lms[IDX.hipR];
    const sOk = isVis(SL, 0.5) && isVis(SR, 0.5), hOk = isVis(HL, 0.5) && isVis(HR, 0.5);
    L.posture = {
      torso: sOk && hOk ? ease('torso', tiltFromVertical(mid(HL, HR), mid(SL, SR), vw, vh)) : ease('torso', null),
      shoulderTilt: sOk ? ease('shoulderTilt', tiltFromHorizontal(SL, SR, vw, vh)) : ease('shoulderTilt', null),
      hipTilt: hOk ? ease('hipTilt', tiltFromHorizontal(HL, HR, vw, vh)) : ease('hipTilt', null),
      twist: sOk && hOk ? ease('twist', torsoTwist(SL, SR, HL, HR)) : ease('twist', null),
    };
  }, []);

  /* ── draw ── */
  const drawScene = useCallback((ctx: CanvasRenderingContext2D, W: number, H: number, lms: NormalizedLandmark[] | null, vw: number, vh: number, crop?: { sx: number; sy: number; sw: number; sh: number }) => {
    if (!lms || !vw || !vh) return;
    const mirror = facingRef.current === 'user';
    let sx = 0, sy = 0, sw = vw, sh = vh;
    if (crop) { sx = crop.sx; sy = crop.sy; sw = crop.sw; sh = crop.sh; }
    else { const s = Math.max(W / vw, H / vh); sw = W / s; sh = H / s; sx = (vw - sw) / 2; sy = (vh - sh) / 2; }
    const kx = W / sw, ky = H / sh;
    const u = Math.max(0.6, W / 640);
    const px = (lm: Lm) => { const x = (lm.x * vw - sx) * kx; return mirror ? W - x : x; };
    const py = (lm: Lm) => (lm.y * vh - sy) * ky;

    /* pill helper */
    const pill = (x: number, y: number, txt: string) => {
      ctx.font = `bold ${14 * u}px "Space Grotesk", sans-serif`;
      const tw = ctx.measureText(txt).width, pw = tw + 18 * u, ph = 26 * u;
      ctx.shadowColor = 'rgba(0,0,0,0.5)'; ctx.shadowBlur = 6 * u;
      ctx.fillStyle = '#111'; ctx.beginPath(); ctx.roundRect(x - pw / 2, y - ph / 2, pw, ph, 7 * u); ctx.fill();
      ctx.shadowBlur = 0;
      ctx.fillStyle = '#fff'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(txt, x, y + 1 * u);
      ctx.textAlign = 'start'; ctx.textBaseline = 'alphabetic';
    };

    /* dot */
    const dot = (x: number, y: number, r: number, c = '#fff') => { ctx.fillStyle = c; ctx.beginPath(); ctx.arc(x, y, r * u, 0, Math.PI * 2); ctx.fill(); };
    /* ring */
    const ring = (x: number, y: number, r: number, c = '#fff') => { ctx.strokeStyle = c; ctx.lineWidth = 1.5 * u; ctx.beginPath(); ctx.arc(x, y, r * u, 0, Math.PI * 2); ctx.stroke(); };
    /* line */
    const line = (a: Lm, b: Lm, c: string, w2: number) => { ctx.strokeStyle = c; ctx.lineWidth = w2 * u; ctx.beginPath(); ctx.moveTo(px(a), py(a)); ctx.lineTo(px(b), py(b)); ctx.stroke(); };

    const SL = lms[IDX.shoulderL], SR = lms[IDX.shoulderR];
    const HL = lms[IDX.hipL], HR = lms[IDX.hipR];
    const KL = lms[IDX.kneeL], KR = lms[IDX.kneeR];
    const AL = lms[IDX.ankleL], AR = lms[IDX.ankleR];
    const EL = lms[IDX.elbowL], ER = lms[IDX.elbowR];
    const WL = lms[IDX.wristL], WR = lms[IDX.wristR];
    const sOk = isVis(SL, 0.4) && isVis(SR, 0.4);
    const hOk = isVis(HL, 0.4) && isVis(HR, 0.4);
    ctx.lineCap = 'round';

    /* torso cage — RED */
    if (sOk && hOk) {
      line(SL, SR, '#ff3333', 2.5);
      line(HL, HR, '#ff3333', 2.5);
      line(SL, HL, '#ff3333', 2.5);
      line(SR, HR, '#ff3333', 2.5);
      for (const p of [SL, SR, HL, HR]) dot(px(p), py(p), 4, '#ff3333');

      const post = liveRef.current.posture;
      if (post.torso !== null) {
        const cx = (px(SL) + px(SR)) / 2;
        const cy = (py(SL) + py(HL)) / 2;
        pill(cx, cy, `${90 + post.torso}°`);
      }
    }

    /* shoulder midpoint arc + pill */
    if (sOk) {
      const mS = mid(SL, SR);
      const mx = px(mS), my = py(mS);
      ring(mx, my, 14, '#fff');
      const a = liveRef.current.posture;
      if (a.shoulderTilt !== null) pill(mx, my - 28 * u, `${Math.max(0, 180 - Math.abs(a.shoulderTilt))}°`);
    }

    /* hip midpoint arc + pill */
    if (hOk) {
      const mH = mid(HL, HR);
      const mx = px(mH), my = py(mH);
      ring(mx, my, 14, '#fff');
      const a = liveRef.current.posture;
      if (a.hipTilt !== null) pill(mx, my + 28 * u, `${Math.max(0, 180 - Math.abs(a.hipTilt))}°`);
    }

    /* limbs — WHITE */
    const limbs: [Lm, Lm][] = [[SL, EL], [EL, WL], [SR, ER], [ER, WR], [HL, KL], [KL, AL], [HR, KR], [KR, AR]];
    for (const [a, b] of limbs) { if (isVis(a, 0.4) && isVis(b, 0.4)) { line(a, b, '#ffffff', 2); dot(px(b), py(b), 3); } }

    /* elbow / knee angle pills */
    const sm = smoothA.current;
    const LABELS: { j: number; k: AngleKey; off: [number, number] }[] = [
      { j: IDX.elbowL, k: 'elbowL', off: [-35, -10] },
      { j: IDX.elbowR, k: 'elbowR', off: [35, -10] },
      { j: IDX.kneeL, k: 'kneeL', off: [-35, 10] },
      { j: IDX.kneeR, k: 'kneeR', off: [35, 10] },
    ];
    for (const f of LABELS) {
      const B = lms[f.j]; const v = sm[f.k];
      if (isVis(B, 0.5) && v !== null) pill(px(B) + f.off[0] * u, py(B) + f.off[1] * u, `${Math.round(v)}°`);
    }

    /* head marker */
    const earL = lms[IDX.earL], earR = lms[IDX.earR], nose = lms[IDX.nose];
    let hx: number | null = null, hy: number | null = null;
    if (isVis(earL, 0.4) && isVis(earR, 0.4)) { const e = mid(earL, earR); hx = px(e); hy = py(e) - Math.max(10 * u, Math.abs(px(earL) - px(earR))) * 0.3; }
    else if (isVis(nose, 0.4)) { hx = px(nose); hy = py(nose); }
    if (hx !== null && hy !== null) { dot(hx, hy, 5, '#fff'); ring(hx, hy, 10, '#fff'); }

    /* neck dot */
    if (sOk) {
      const n = mid(SL, SR);
      dot(px(n), py(n), 3.5, '#ddd');
      if (hx !== null && hy !== null) { ctx.strokeStyle = 'rgba(255,255,255,0.5)'; ctx.lineWidth = 1.5 * u; ctx.beginPath(); ctx.moveTo(hx, hy); ctx.lineTo(px(n), py(n)); ctx.stroke(); }
    }
  }, []);

  const drawLive = useCallback((lms: NormalizedLandmark[] | null, vw: number, vh: number) => {
    const c = canvasRef.current; if (!c) return;
    const ctx = c.getContext('2d'); if (!ctx) return;
    ctx.clearRect(0, 0, c.clientWidth, c.clientHeight);
    drawScene(ctx, c.clientWidth, c.clientHeight, lms, vw, vh);
  }, [canvasRef, drawScene]);

  /* ── loop ── */
  const loop = useCallback(() => {
    if (!runRef.current) return;
    const v = videoRef.current, lm = lmRef.current;
    if (v && lm && v.readyState >= 2 && v.currentTime !== lastTRef.current) {
      lastTRef.current = v.currentTime;
      const now = performance.now();
      try {
        const res = lm.detectForVideo(v, now);
        const rawPts = res.landmarks?.[0] ?? null;
        const pts = stabilizeLandmarks(rawPts, stableLmRef.current);
        stableLmRef.current = pts;
        frameRef.current = { lms: pts, vw: v.videoWidth, vh: v.videoHeight };
        processFrame(pts, v.videoWidth, v.videoHeight);
        drawLive(pts, v.videoWidth, v.videoHeight);
      } catch { /* skip */ }
      /* snapshot for capture */
      try {
        if (v.videoWidth) {
          let fc = snapRef.current;
          if (!fc) { fc = document.createElement('canvas'); snapRef.current = fc; }
          if (fc.width !== v.videoWidth || fc.height !== v.videoHeight) { fc.width = v.videoWidth; fc.height = v.videoHeight; }
          fc.getContext('2d')?.drawImage(v, 0, 0); snapDirty.current = now;
        }
      } catch { /* */ }
      const m = fpsRef.current; m.push(now); while (m.length && m[0] < now - 1000) m.shift(); liveRef.current.fps = m.length;
    }
    rafRef.current = requestAnimationFrame(loop);
  }, [drawLive, processFrame, videoRef]);

  /* ── open stream ── */
  const openStream = useCallback(async (f: Facing) => {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error('INSECURE');
    const tries = [
      { video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: { ideal: f } }, audio: false },
      { video: { facingMode: { ideal: f } }, audio: false },
      { video: true, audio: false },
    ];
    let err: unknown; for (const c of tries) { try { return await navigator.mediaDevices.getUserMedia(c); } catch (e) { err = e; if (e instanceof DOMException && e.name === 'NotAllowedError') break; } }
    throw err ?? new Error('unavailable');
  }, []);

  const attachStream = useCallback(async (s: MediaStream) => {
    const v = videoRef.current; if (!v) return;
    streamRef.current = s; v.muted = true; v.playsInline = true; v.setAttribute('playsinline', ''); v.setAttribute('webkit-playsinline', '');
    try { v.srcObject = s; } catch { v.src = URL.createObjectURL(s as any); }
    try { await v.play(); } catch { await new Promise<void>(r => { v.addEventListener('loadedmetadata', () => { void v.play().finally(r); }, { once: true }); setTimeout(r, 2000); }); }
  }, [videoRef]);

  /* ── start ── */
  const start = useCallback(async (auto = false) => {
    if (modelState !== 'ready' || camState === 'running' || camState === 'starting') return;
    setCamState('starting'); setCamError('');
    try {
      const s = await openStream(facingRef.current); await attachStream(s);
      try { localStorage.setItem(GRANTED_KEY, '1'); } catch { /* */ }
      sessRef.current = { start: performance.now(), carry: sessRef.current.carry };
      runRef.current = true; lastTRef.current = -1; setCamState('running');
      rafRef.current = requestAnimationFrame(loop);
      try { audioRef.current ??= new AudioContext(); if (audioRef.current.state === 'suspended') void audioRef.current.resume(); } catch { /* */ }
      if (soundRef.current) { beep(520, 0.08, 0.04); setTimeout(() => beep(760, 0.08, 0.04), 110); }
    } catch (e) {
      if (auto && e instanceof DOMException && e.name === 'NotAllowedError') { setCamState('idle'); return; }
      setCamState('error'); setCamError(camErrMsg(e));
    }
  }, [modelState, camState, openStream, attachStream, loop, beep]);

  /* auto-start: in PWA standalone mode, aggressively auto-start so iOS
     doesn't ask for permission again (permissions persist per-PWA on iOS 16.4+) */
  useEffect(() => {
    if (autoRef.current || modelState !== 'ready') return;
    autoRef.current = true;
    const isPWA =
      window.matchMedia?.('(display-mode: standalone)').matches ||
      (navigator as any).standalone === true;
    try {
      if (localStorage.getItem(GRANTED_KEY) || isPWA) void start(true);
    } catch { /* */ }
  }, [modelState, start]);

  /* ── stop ── */
  const stop = useCallback(() => {
    runRef.current = false;
    cancelAnimationFrame(rafRef.current);
    streamRef.current?.getTracks().forEach(t => t.stop());
    streamRef.current = null;
    const v = videoRef.current;
    if (v) v.srcObject = null;
    if (sessRef.current.start) {
      sessRef.current.carry += performance.now() - sessRef.current.start;
      sessRef.current.start = 0;
    }
    const c = canvasRef.current;
    c?.getContext('2d')?.clearRect(0, 0, c.clientWidth, c.clientHeight);
    stableLmRef.current = null;
    smoothA.current = { ...NULL_ANG };
    smoothP.current = { ...NULL_POST };
    liveRef.current = { ...NULL_STATS };
    setCamState('idle');
  }, [videoRef, canvasRef]);

  const reset = useCallback(() => { sessRef.current = { start: runRef.current ? performance.now() : 0, carry: 0 }; liveRef.current.seconds = 0; }, []);

  /* ── flip ── */
  const flipCamera = useCallback(async () => {
    if (flipBusy) return;
    const next: Facing = facingRef.current === 'user' ? 'environment' : 'user';
    facingRef.current = next; setFacing(next);
    if (!runRef.current) return; setFlipBusy(true);
    try { const s = await openStream(next); streamRef.current?.getTracks().forEach(t => t.stop()); await attachStream(s); lastTRef.current = -1; }
    catch { /* */ }
    finally { setFlipBusy(false); }
  }, [flipBusy, openStream, attachStream]);

  /* ── capture ── */
  const capture = useCallback(async (): Promise<boolean> => {
    const v = videoRef.current; if (!v || !v.videoWidth) return false;
    const vw = v.videoWidth, vh = v.videoHeight;
    const overlay = canvasRef.current;
    const cw = Math.max(1, overlay?.clientWidth ?? vw), ch = Math.max(1, overlay?.clientHeight ?? vh);
    const sc = Math.max(cw / vw, ch / vh);
    const sw2 = Math.round(cw / sc), sh2 = Math.round(ch / sc), sx2 = Math.round((vw - sw2) / 2), sy2 = Math.round((vh - sh2) / 2);
    const c = document.createElement('canvas'); c.width = sw2; c.height = sh2;
    const ctx = c.getContext('2d'); if (!ctx) return false;
    ctx.fillStyle = '#000'; ctx.fillRect(0, 0, sw2, sh2);
    const mir = facingRef.current === 'user';
    let ok = false;
    /* try snapshot canvas first */
    const fc = snapRef.current;
    if (fc && performance.now() - snapDirty.current < 400) {
      try { ctx.save(); if (mir) { ctx.translate(sw2, 0); ctx.scale(-1, 1); } ctx.drawImage(fc, sx2, sy2, sw2, sh2, 0, 0, sw2, sh2); ctx.restore(); ok = !frameIsBlack(ctx, sw2, sh2); } catch { /* */ }
    }
    if (!ok) { try { ctx.save(); if (mir) { ctx.translate(sw2, 0); ctx.scale(-1, 1); } ctx.drawImage(v, sx2, sy2, sw2, sh2, 0, 0, sw2, sh2); ctx.restore(); } catch { /* */ } }
    drawScene(ctx, sw2, sh2, frameRef.current?.lms ?? null, vw, vh, { sx: sx2, sy: sy2, sw: sw2, sh: sh2 });
    ctx.font = '600 12px Tahoma, sans-serif'; ctx.fillStyle = 'rgba(250,204,21,0.75)';
    const wm = 'آنالیز سعیدکلاب · SAEED CLUB'; ctx.fillText(wm, sw2 - ctx.measureText(wm).width - 14, sh2 - 14);
    /* Save: prefer Web Share API (→ iOS Photos directly), fallback to download */
    try {
      const blob = await new Promise<Blob | null>((res) => c.toBlob(res, 'image/jpeg', 0.92));
      if (!blob) return false;
      const fname = `saeed-club-${Date.now()}.jpg`;
      const file = new File([blob], fname, { type: 'image/jpeg' });

      if (navigator.share && navigator.canShare?.({ files: [file] })) {
        await navigator.share({ files: [file], title: 'SaeidClub Analyse' });
      } else {
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = fname;
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
      }
    } catch (e) {
      if (e instanceof DOMException && e.name === 'AbortError') return true; /* user cancelled share sheet */
      return false;
    }
    return true;
  }, [videoRef, canvasRef, drawScene]);

  /* ── stats pump ── */
  useEffect(() => { const id = setInterval(() => { const s = sessRef.current; liveRef.current.seconds = Math.floor((s.start ? s.carry + performance.now() - s.start : s.carry) / 1000); setStats({ ...liveRef.current }); }, 125); return () => clearInterval(id); }, []);

  /* ── canvas resize ── */
  useEffect(() => { const c = canvasRef.current; if (!c) return; const p = c.parentElement; if (!p) return; const ro = new ResizeObserver(() => { const d = Math.min(2, devicePixelRatio || 1); c.width = Math.round(p.clientWidth * d); c.height = Math.round(p.clientHeight * d); c.getContext('2d')?.setTransform(d, 0, 0, d, 0, 0); }); ro.observe(p); return () => ro.disconnect(); }, [canvasRef]);

  /* ── tab resume ── */
  useEffect(() => { const h = () => { if (document.visibilityState === 'visible' && runRef.current && videoRef.current?.paused) { lastTRef.current = -1; void videoRef.current.play().catch(() => {}); } }; document.addEventListener('visibilitychange', h); return () => document.removeEventListener('visibilitychange', h); }, [videoRef]);

  /* ── video recording (composite: camera + skeleton overlay) ── */
  const [recording, setRecording] = useState(false);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const recChunks = useRef<Blob[]>([]);
  const recCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const recRafRef = useRef(0);

  const startRecording = useCallback(() => {
    const vid = videoRef.current;
    if (!vid || !vid.videoWidth || !runRef.current) return;

    // create an offscreen canvas for compositing
    const rc = document.createElement('canvas');
    rc.width = vid.videoWidth;
    rc.height = vid.videoHeight;
    recCanvasRef.current = rc;

    const stream = rc.captureStream(30);
    const mimeTypes = [
      'video/mp4;codecs=avc1',
      'video/webm;codecs=vp9',
      'video/webm;codecs=vp8',
      'video/webm',
    ];
    let mime = '';
    for (const m of mimeTypes) {
      if (MediaRecorder.isTypeSupported(m)) { mime = m; break; }
    }
    const mr = new MediaRecorder(stream, mime ? { mimeType: mime } : {});
    recChunks.current = [];
    mr.ondataavailable = (e) => { if (e.data.size > 0) recChunks.current.push(e.data); };
    mr.onstop = async () => {
      const ext = mime.includes('mp4') ? 'mp4' : 'webm';
      const blob = new Blob(recChunks.current, { type: mime || 'video/webm' });
      const fname = `saeed-club-${Date.now()}.${ext}`;
      const file = new File([blob], fname, { type: blob.type });
      try {
        if (navigator.share && navigator.canShare?.({ files: [file] })) {
          await navigator.share({ files: [file], title: 'SaeidClub Analyse' });
        } else {
          const url = URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url; a.download = fname;
          document.body.appendChild(a); a.click(); a.remove();
          URL.revokeObjectURL(url);
        }
      } catch { /* user cancelled or error */ }
    };
    mr.start(200);
    recorderRef.current = mr;
    setRecording(true);

    // composite loop: draw video + skeleton onto recording canvas
    const mirror = facingRef.current === 'user';
    const paint = () => {
      if (!recorderRef.current || recorderRef.current.state !== 'recording') return;
      const ctx = rc.getContext('2d');
      if (ctx && vid.readyState >= 2) {
        ctx.clearRect(0, 0, rc.width, rc.height);
        ctx.save();
        if (mirror) { ctx.translate(rc.width, 0); ctx.scale(-1, 1); }
        ctx.drawImage(vid, 0, 0, rc.width, rc.height);
        ctx.restore();
        const lastLms = frameRef.current?.lms ?? null;
        drawScene(ctx, rc.width, rc.height, lastLms, vid.videoWidth, vid.videoHeight,
          { sx: 0, sy: 0, sw: vid.videoWidth, sh: vid.videoHeight });
        // watermark
        ctx.font = '600 14px Tahoma, sans-serif';
        ctx.fillStyle = 'rgba(250,204,21,0.6)';
        const wm = 'SaeidClub Analyse';
        ctx.fillText(wm, rc.width - ctx.measureText(wm).width - 16, rc.height - 16);
      }
      recRafRef.current = requestAnimationFrame(paint);
    };
    recRafRef.current = requestAnimationFrame(paint);
  }, [videoRef, drawScene]);

  const stopRecording = useCallback(() => {
    cancelAnimationFrame(recRafRef.current);
    if (recorderRef.current && recorderRef.current.state === 'recording') {
      recorderRef.current.stop();
    }
    recorderRef.current = null;
    recCanvasRef.current = null;
    setRecording(false);
  }, []);

  /* ── cleanup ── */
  useEffect(() => () => {
    runRef.current = false;
    cancelAnimationFrame(rafRef.current);
    cancelAnimationFrame(recRafRef.current);
    if (recorderRef.current?.state === 'recording') recorderRef.current.stop();
    streamRef.current?.getTracks().forEach(t => t.stop());
    void lmRef.current?.close();
    void audioRef.current?.close().catch(() => {});
  }, []);

  return {
    modelState, camState, camError, stats, repFlash, facing, flipBusy,
    start, stop, reset, flipCamera, capture, beep,
    recording, startRecording, stopRecording,
  };
}
