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
  effortProgress,
  exerciseByKey,
  feedRep,
  freshRepMachine,
  isVis,
  mid,
  tiltFromHorizontal,
  tiltFromVertical,
  torsoTwist,
  type AngleKey,
  type ExerciseKey,
  type Lm,
  type RepMachine,
} from '../lib/pose';

const WASM_URL =
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';
const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';

const GRANTED_KEY = 'posecam.granted';

export type CamState = 'idle' | 'starting' | 'running' | 'error';
export type ModelState = 'loading' | 'ready' | 'error';
export type Facing = 'user' | 'environment';

const storage = {
  get(key: string) {
    try {
      return localStorage.getItem(key);
    } catch {
      return null;
    }
  },
  set(key: string, val: string) {
    try {
      localStorage.setItem(key, val);
    } catch {
      /* private mode */
    }
  },
};

/** Persian, user-friendly camera error messages (incl. iOS/Safari cases). */
function cameraErrorMessage(e: unknown): string {
  if (e instanceof Error && e.message === 'INSECURE_CONTEXT') {
    return 'دسترسی به دوربین فقط در اتصال امن (HTTPS) یا localhost ممکن است. لطفاً صفحه را با آدرس https باز کنید.';
  }
  if (e instanceof DOMException) {
    switch (e.name) {
      case 'NotAllowedError':
      case 'PermissionDeniedError':
        return 'دسترسی به دوربین رد شد. روی آیکون دوربین در نوار آدرس مرورگر بزنید و اجازه دسترسی را فعال کنید، سپس دوباره تلاش کنید.';
      case 'NotFoundError':
      case 'DevicesNotFoundError':
        return 'هیچ دوربینی روی این دستگاه پیدا نشد.';
      case 'NotReadableError':
      case 'TrackStartError':
        return 'دوربین توسط برنامه دیگری در حال استفاده است. آن برنامه را ببندید و دوباره تلاش کنید.';
      case 'OverconstrainedError':
        return 'دوربین دستگاه با تنظیمات درخواستی سازگار نیست.';
      case 'SecurityError':
        return 'دسترسی دوربین به دلیل محدودیت امنیتی مرورگر مسدود است؛ اتصال HTTPS لازم است.';
      case 'AbortError':
        return 'راه‌اندازی دوربین متوقف شد. لطفاً دوباره تلاش کنید.';
      default:
        break;
    }
  }
  return 'دوربین در دسترس نیست. اتصال وب‌کم را بررسی کنید و دوباره تلاش کنید.';
}

/** Detect a (near-)all-black frame — used to validate capture methods. */
function frameAlmostBlack(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
): boolean {
  try {
    const { data } = ctx.getImageData(0, 0, w, h);
    let sum = 0;
    let n = 0;
    const stride = 97 * 4;
    for (let i = 0; i + 2 < data.length; i += stride) {
      sum += data[i] + data[i + 1] + data[i + 2];
      n++;
    }
    return n === 0 ? false : sum / (n * 3) < 3.5;
  } catch {
    return false;
  }
}

export interface LiveStats {
  fps: number;
  present: boolean;
  tracked: number;
  reps: number;
  phase: 'up' | 'down' | null;
  lastQuality: number | null;
  primary: number | null;
  effort: number;
  angles: {
    elbowL: number | null;
    elbowR: number | null;
    kneeL: number | null;
    kneeR: number | null;
    hip: number | null;
    shoulder: number | null;
    /** neck flexion (ear-shoulder-hip) — lower = head tilting forward */
    neck: number | null;
    neckR: number | null;
  };
  /** relations between the major joint pairs (segment-to-segment) */
  posture: {
    /** spine lean away from vertical: 0° = upright */
    torso: number | null;
    /** shoulder line vs horizontal: 0° = level */
    shoulderTilt: number | null;
    /** hip line vs horizontal: 0° = level */
    hipTilt: number | null;
    /** shoulders rotated against hips: 0° = square */
    twist: number | null;
  };
  spark: number[];
  seconds: number;
}

const EMPTY_ANGLES = {
  elbowL: null,
  elbowR: null,
  kneeL: null,
  kneeR: null,
  hip: null,
  shoulder: null,
  neck: null,
  neckR: null,
};

const EMPTY_POSTURE = {
  torso: null,
  shoulderTilt: null,
  hipTilt: null,
  twist: null,
};

const EMPTY_STATS: LiveStats = {
  fps: 0,
  present: false,
  tracked: 0,
  reps: 0,
  phase: null,
  lastQuality: null,
  primary: null,
  effort: 0,
  angles: EMPTY_ANGLES,
  posture: EMPTY_POSTURE,
  spark: [],
  seconds: 0,
};

type FacingState = 'user' | 'environment';

export function usePoseEngine(
  videoRef: React.RefObject<HTMLVideoElement | null>,
  canvasRef: React.RefObject<HTMLCanvasElement | null>,
  exercise: ExerciseKey,
  soundOn: boolean,
) {
  const [modelState, setModelState] = useState<ModelState>('loading');
  const [camState, setCamState] = useState<CamState>('idle');
  const [camError, setCamError] = useState<string>('');
  const [facing, setFacing] = useState<Facing>('environment');
  const [repFlash, setRepFlash] = useState(0);
  const [stats, setStats] = useState<LiveStats>(EMPTY_STATS);

  const landmarkerRef = useRef<PoseLandmarker | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const rafRef = useRef(0);
  const runningRef = useRef(false);
  const lastVideoTimeRef = useRef(-1);
  const exerciseRef = useRef(exercise);
  const facingRef = useRef<FacingState>('environment');
  const repRef = useRef<RepMachine | null>(null);
  const sessionRef = useRef({ startedAt: 0, carried: 0 });
  const audioRef = useRef<AudioContext | null>(null);
  const soundRef = useRef(soundOn);
  const fpsRef = useRef({ marks: [] as number[] });
  const sparkRef = useRef<number[]>([]);
  const lastSparkAtRef = useRef(0);
  const autoTriedRef = useRef(false);
  const lastFrameRef = useRef<{
    lms: NormalizedLandmark[] | null;
    vw: number;
    vh: number;
  } | null>(null);
  const frameCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const frameCtxRef = useRef<CanvasRenderingContext2D | null>(null);
  const frameDirtyRef = useRef(0); // last performance.now() we painted
  /** EMA-smoothed joint angles (null = joint not confidently visible) */
  const smoothRef = useRef<Record<AngleKey, number | null>>({
    elbowL: null,
    elbowR: null,
    kneeL: null,
    kneeR: null,
    hip: null,
    shoulder: null,
    neck: null,
    neckR: null,
  });
  const smoothPrimaryRef = useRef<number | null>(null);
  const smoothMirrorRef = useRef<number | null>(null);
  const smoothPostureRef = useRef<LiveStats['posture']>({
    torso: null,
    shoulderTilt: null,
    hipTilt: null,
    twist: null,
  });
  const liveRef = useRef<LiveStats>(EMPTY_STATS);

  soundRef.current = soundOn;

  // ── load model once ─────────────────────────────────────────
  useEffect(() => {
    let dead = false;
    (async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks(WASM_URL);
        let lm: PoseLandmarker;
        const opts = (delegate: 'GPU' | 'CPU') => ({
          baseOptions: { modelAssetPath: MODEL_URL, delegate },
          runningMode: 'VIDEO' as const,
          numPoses: 1,
          minPoseDetectionConfidence: 0.5,
          minPosePresenceConfidence: 0.5,
          minTrackingConfidence: 0.5,
        });
        try {
          lm = await PoseLandmarker.createFromOptions(vision, opts('GPU'));
        } catch {
          lm = await PoseLandmarker.createFromOptions(vision, opts('CPU'));
        }
        if (dead) {
          lm.close();
          return;
        }
        landmarkerRef.current = lm;
        setModelState('ready');
      } catch (e) {
        console.error('Pose model failed to load', e);
        if (!dead) setModelState('error');
      }
    })();
    return () => {
      dead = true;
    };
  }, []);

  // ── reset counters when exercise changes ────────────────────
  useEffect(() => {
    exerciseRef.current = exercise;
    repRef.current = null;
    sparkRef.current = [];
    liveRef.current = {
      ...liveRef.current,
      reps: 0,
      phase: null,
      lastQuality: null,
      primary: null,
      effort: 0,
      spark: [],
    };
  }, [exercise]);

  // ── beep ────────────────────────────────────────────────────
  const beep = useCallback((freq = 760, dur = 0.09, gain = 0.05) => {
    if (!soundRef.current) return;
    try {
      const ctx = (audioRef.current ??= new AudioContext());
      if (ctx.state === 'suspended') void ctx.resume();
      const o = ctx.createOscillator();
      const g = ctx.createGain();
      o.type = 'sine';
      o.frequency.value = freq;
      g.gain.setValueAtTime(gain, ctx.currentTime);
      g.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + dur);
      o.connect(g).connect(ctx.destination);
      o.start();
      o.stop(ctx.currentTime + dur + 0.02);
    } catch {
      /* audio unavailable */
    }
  }, []);

  // ── scene renderer (shared by live overlay + photo capture) ─
  const drawScene = useCallback(
    (
      ctx: CanvasRenderingContext2D,
      W: number,
      H: number,
      lms: NormalizedLandmark[] | null,
      vw: number,
      vh: number,
      crop?: { sx: number; sy: number; sw: number; sh: number },
    ) => {
      if (!vw || !vh || !lms) return;

      const mirror = facingRef.current === 'user';
      // source-region → output mapping (defaults to object-cover fit)
      let sx = 0;
      let sy = 0;
      let sw = vw;
      let sh = vh;
      if (crop) {
        sx = crop.sx;
        sy = crop.sy;
        sw = crop.sw;
        sh = crop.sh;
      } else {
        const sCover = Math.max(W / vw, H / vh);
        sw = W / sCover;
        sh = H / sCover;
        sx = (vw - sw) / 2;
        sy = (vh - sh) / 2;
      }
      const kx = W / sw;
      const ky = H / sh;
      const px = (lm: Lm) => {
        const x = (lm.x * vw - sx) * kx;
        return mirror ? W - x : x;
      };
      const py = (lm: Lm) => (lm.y * vh - sy) * ky;
      const u = Math.max(0.65, W / 640); // scale factor

      const def = exerciseByKey(exerciseRef.current);
      const active = new Set<number>();
      if (def.joints) {
        def.joints.forEach((j) => active.add(j));
        def.mirror?.forEach((j) => active.add(j));
      }

      // ── connections ──
      const conns = PoseLandmarker.POSE_CONNECTIONS;
      ctx.lineCap = 'round';
      for (const { start, end } of conns) {
        if (start < 11 || end < 11) continue;
        const a = lms[start];
        const b = lms[end];
        if (!isVis(a) || !isVis(b)) continue;
        const isActive = active.has(start) && active.has(end);
        ctx.strokeStyle = isActive
          ? 'rgba(255,255,46,0.95)'
          : 'rgba(250,204,21,0.85)';
        ctx.lineWidth = (isActive ? 4 : 2.2) * u;
        ctx.shadowColor = isActive
          ? 'rgba(255,255,46,0.8)'
          : 'rgba(250,204,21,0.65)';
        ctx.shadowBlur = (isActive ? 14 : 8) * u;
        ctx.beginPath();
        ctx.moveTo(px(a), py(a));
        ctx.lineTo(px(b), py(b));
        ctx.stroke();
      }
      ctx.shadowBlur = 0;

      // ── torso frame: shoulder line, hip line, spine axis ──
      {
        const SL = lms[IDX.shoulderL];
        const SR = lms[IDX.shoulderR];
        const HL = lms[IDX.hipL];
        const HR = lms[IDX.hipR];
        const sOk = isVis(SL, 0.5) && isVis(SR, 0.5);
        const hOk = isVis(HL, 0.5) && isVis(HR, 0.5);
        const seg = (a: Lm, b: Lm, tint: string, dash: boolean) => {
          ctx.strokeStyle = tint;
          ctx.lineWidth = 2.2 * u;
          ctx.shadowColor = tint;
          ctx.shadowBlur = 8 * u;
          if (dash) ctx.setLineDash([7 * u, 5 * u]);
          ctx.beginPath();
          ctx.moveTo(px(a), py(a));
          ctx.lineTo(px(b), py(b));
          ctx.stroke();
          ctx.setLineDash([]);
          ctx.shadowBlur = 0;
        };
        if (sOk) seg(SL, SR, 'rgba(250,204,21,0.8)', true);
        if (hOk) seg(HL, HR, 'rgba(245,158,11,0.8)', true);
        if (sOk && hOk) {
          const sm2 = mid(SL, SR);
          const hm2 = mid(HL, HR);
          seg(hm2, sm2, 'rgba(255,255,46,0.9)', false);
          // midpoint markers
          for (const m of [sm2, hm2]) {
            ctx.fillStyle = 'rgba(244,114,182,0.95)';
            ctx.beginPath();
            ctx.arc(px(m), py(m), 3.4 * u, 0, Math.PI * 2);
            ctx.fill();
          }
        }
      }

      // ── head & neck markers vs. the symmetric plumb line ──
      {
        const earL = lms[IDX.earL];
        const earR = lms[IDX.earR];
        const nose = lms[IDX.nose];
        const SL3 = lms[IDX.shoulderL];
        const SR3 = lms[IDX.shoulderR];
        const HL3 = lms[IDX.hipL];
        const HR3 = lms[IDX.hipR];

        // ─ head center, robust from every angle (ears mid → nose → ear) ─
        let headX: number | null = null;
        let headY: number | null = null;
        if (isVis(earL, 0.45) && isVis(earR, 0.45)) {
          const e = mid(earL, earR);
          const earSpan = Math.max(10 * u, Math.abs(px(earL) - px(earR)));
          headX = px(e);
          headY = py(e) - earSpan * 0.3; // lift toward skull center
        } else if (isVis(nose, 0.45)) {
          headX = px(nose);
          headY = py(nose);
        } else if (isVis(earL, 0.45)) {
          headX = px(earL);
          headY = py(earL);
        } else if (isVis(earR, 0.45)) {
          headX = px(earR);
          headY = py(earR);
        }

        const neckOk = isVis(SL3, 0.45) && isVis(SR3, 0.45);
        const hipsOk = isVis(HL3, 0.45) && isVis(HR3, 0.45);
        const neckPt = neckOk ? mid(SL3, SR3) : null;
        const plumbX = hipsOk
          ? (px(HL3) + px(HR3)) / 2
          : neckPt && headX !== null
            ? px(neckPt)
            : null;

        const markerAt = (
          X: number,
          Y: number,
          coreR: number,
          core: string,
          ring: string,
        ) => {
          ctx.strokeStyle = ring;
          ctx.lineWidth = 1.6 * u;
          ctx.beginPath();
          ctx.arc(X, Y, coreR * 2.1 * u, 0, Math.PI * 2);
          ctx.stroke();
          ctx.fillStyle = core;
          ctx.shadowColor = core;
          ctx.shadowBlur = 12 * u;
          ctx.beginPath();
          ctx.arc(X, Y, coreR * u, 0, Math.PI * 2);
          ctx.fill();
          ctx.shadowBlur = 0;
        };

        // ─ symmetric reference: vertical plumb line through pelvis center ─
        if (plumbX !== null && hipsOk) {
          const hy2 = py(mid(HL3, HR3));
          const sy2 = neckPt ? py(neckPt) : hy2 - 100 * u;
          const bodyLen = Math.max(60 * u, hy2 - sy2);
          ctx.strokeStyle = 'rgba(250,204,21,0.5)';
          ctx.lineWidth = 1.6 * u;
          ctx.shadowColor = 'rgba(250,204,21,0.4)';
          ctx.shadowBlur = 5 * u;
          ctx.beginPath();
          ctx.moveTo(plumbX, hy2 + bodyLen * 0.18);
          ctx.lineTo(plumbX, hy2 - bodyLen * 2);
          ctx.stroke();
          ctx.shadowBlur = 0;
        }

        // ─ deviation connectors: horizontal distance to the plumb line ─
        const devLine = (
          X: number,
          Y: number,
          tol: number,
        ) => {
          if (plumbX === null) return;
          const d = Math.abs(X - plumbX);
          const ok2 = d <= tol;
          ctx.strokeStyle = ok2
            ? 'rgba(250,204,21,0.75)'
            : 'rgba(245,158,11,0.95)';
          ctx.lineWidth = 1.4 * u;
          ctx.shadowColor = ok2
            ? 'rgba(250,204,21,0.5)'
            : 'rgba(245,158,11,0.6)';
          ctx.shadowBlur = 5 * u;
          ctx.beginPath();
          ctx.moveTo(plumbX, Y);
          ctx.lineTo(X, Y);
          ctx.stroke();
          ctx.shadowBlur = 0;
        };

        if (neckPt) {
          const nx2 = px(neckPt);
          const ny2 = py(neckPt);
          if (plumbX !== null && hipsOk) devLine(nx2, ny2, 16 * u);
          markerAt(nx2, ny2, 3.6, '#facc15', 'rgba(250,204,21,0.5)');
        }
        if (headX !== null && headY !== null) {
          if (plumbX !== null && hipsOk) devLine(headX, headY, 26 * u);
          markerAt(headX, headY, 4.6, '#ffff2e', 'rgba(255,255,46,0.55)');
        }
      }

      // ── keypoints (body only) ──
      for (let i = 11; i < lms.length; i++) {
        const lm = lms[i];
        if (!isVis(lm)) continue;
        const isActive = active.has(i);
        const r = (isActive ? 5 : 3.2) * u;
        ctx.fillStyle = isActive ? '#ffff2e' : 'rgba(255,240,160,0.9)';
        ctx.shadowColor = isActive
          ? 'rgba(255,255,46,0.9)'
          : 'rgba(250,204,21,0.7)';
        ctx.shadowBlur = (isActive ? 12 : 6) * u;
        ctx.beginPath();
        ctx.arc(px(lm), py(lm), r, 0, Math.PI * 2);
        ctx.fill();
        ctx.shadowBlur = 0;
        if (isActive) {
          ctx.fillStyle = '#0a0800';
          ctx.beginPath();
          ctx.arc(px(lm), py(lm), r * 0.42, 0, Math.PI * 2);
          ctx.fill();
        }
      }

      // ── angle pill helper ──
      const drawPill = (
        x: number,
        y: number,
        text: string,
        tint: string,
        fontSize = 13,
        pad = 6,
      ) => {
        ctx.font = `600 ${fontSize * u}px "Space Grotesk", monospace`;
        const tw = ctx.measureText(text).width + pad * 2 * u;
        const ph = (fontSize + 6) * u;
        const rx = Math.min(Math.max(x, 4 * u), W - tw - 4 * u);
        const ry = Math.min(Math.max(y, 4 * u), H - ph - 4 * u);
        ctx.fillStyle = 'rgba(4,10,16,0.8)';
        ctx.strokeStyle = tint;
        ctx.lineWidth = 1 * u;
        ctx.beginPath();
        ctx.roundRect(rx, ry, tw, ph, 6 * u);
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = tint;
        ctx.fillText(text, rx + pad * u, ry + ph - (pad - 1) * u);
      };

      if (def.joints) {
        // ── exercise mode: arc on tracked joint ──
        const [ia, ib, ic] = def.joints;
        const A = lms[ia];
        const B = lms[ib];
        const C = lms[ic];
        if (isVis(A) && isVis(B) && isVis(C)) {
          const bx = px(B);
          const by = py(B);
          const a1 = Math.atan2(py(A) - by, px(A) - bx);
          const a2 = Math.atan2(py(C) - by, px(C) - bx);
          let delta = a2 - a1;
          while (delta > Math.PI) delta -= Math.PI * 2;
          while (delta < -Math.PI) delta += Math.PI * 2;
          ctx.strokeStyle = 'rgba(255,255,46,0.9)';
          ctx.lineWidth = 2.4 * u;
          ctx.setLineDash([5 * u, 5 * u]);
          ctx.beginPath();
          ctx.arc(bx, by, 44 * u, a1, a1 + delta, delta < 0);
          ctx.stroke();
          ctx.setLineDash([]);
          const shown = liveRef.current.primary;
          if (shown !== null) {
            drawPill(bx + 52 * u, by - 30 * u, `${shown}°`, '#ffff2e');
          }
        }
      } else {
        // ── free scan: smoothed 3D angle labels on limb joints ──
        const FREE_LABELS: { joint: number; key: AngleKey; tint: string }[] = [
          { joint: IDX.elbowL, key: 'elbowL', tint: '#fde047' },
          { joint: IDX.elbowR, key: 'elbowR', tint: '#fde047' },
          { joint: IDX.kneeL, key: 'kneeL', tint: '#ffff2e' },
          { joint: IDX.kneeR, key: 'kneeR', tint: '#ffff2e' },
        ];
        const sm = smoothRef.current;
        for (const f of FREE_LABELS) {
          const B = lms[f.joint];
          const v = sm[f.key];
          if (!isVis(B, 0.5) || v === null) continue;
          drawPill(px(B) + 10 * u, py(B) - 26 * u, `${Math.round(v)}°`, f.tint);
        }

        // torso lean label at the spine midpoint
        const post = liveRef.current.posture;
        const SL2 = lms[IDX.shoulderL];
        const SR2 = lms[IDX.shoulderR];
        const HL2 = lms[IDX.hipL];
        const HR2 = lms[IDX.hipR];
        if (
          post.torso !== null &&
          isVis(SL2, 0.5) &&
          isVis(SR2, 0.5) &&
          isVis(HL2, 0.5) &&
          isVis(HR2, 0.5)
        ) {
          const sm2 = mid(SL2, SR2);
          const hm2 = mid(HL2, HR2);
          const cxp = (px(sm2) + px(hm2)) / 2;
          const cyp = (py(sm2) + py(hm2)) / 2;
          drawPill(cxp + 12 * u, cyp - 10 * u, `تنه ${post.torso}°`, '#fbbf24');
        }
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  );

  const drawLive = useCallback(
    (lms: NormalizedLandmark[] | null, vw: number, vh: number) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
      drawScene(ctx, canvas.clientWidth, canvas.clientHeight, lms, vw, vh);
    },
    [canvasRef, drawScene],
  );

  // ── frame processing ────────────────────────────────────────
  const processFrame = useCallback(
    (lms: NormalizedLandmark[] | null, vw: number, vh: number, now: number) => {
      const live = liveRef.current;
      const visibleCount = lms
        ? lms.reduce((n, l) => n + (isVis(l) ? 1 : 0), 0)
        : 0;
      live.tracked = visibleCount;
      live.present = visibleCount >= 16;

      if (!lms || !live.present) {
        live.primary = null;
        live.angles = EMPTY_ANGLES;
        live.posture = EMPTY_POSTURE;
        const sm0 = smoothRef.current;
        (Object.keys(sm0) as AngleKey[]).forEach((k) => (sm0[k] = null));
        const sp0 = smoothPostureRef.current;
        (Object.keys(sp0) as (keyof LiveStats['posture'])[]).forEach(
          (k) => (sp0[k] = null),
        );
        smoothPrimaryRef.current = null;
        smoothMirrorRef.current = null;
        return;
      }

      // ── true 3D angles: gated by landmark visibility + EMA-smoothed ──
      const sm = smoothRef.current;
      const out: LiveStats['angles'] = {
        elbowL: null,
        elbowR: null,
        kneeL: null,
        kneeR: null,
        hip: null,
        shoulder: null,
        neck: null,
        neckR: null,
      };
      (Object.keys(ANGLE_DEFS) as AngleKey[]).forEach((key) => {
        const [ja, jb, jc] = ANGLE_DEFS[key];
        const J = lms[ja];
        const K = lms[jb];
        const L = lms[jc];
        let raw: number | null = null;
        if (isVis(J, 0.5) && isVis(K, 0.5) && isVis(L, 0.5)) {
          const v = angle3D(J, K, L, vw, vh);
          if (Number.isFinite(v)) raw = v;
        }
        if (raw === null) {
          sm[key] = null;
        } else {
          sm[key] = sm[key] === null ? raw : sm[key] + 0.35 * (raw - sm[key]);
          out[key] = Math.round(sm[key]!);
        }
      });
      live.angles = out;

      // ── segment-to-segment posture (shoulders ↔ hips) ──
      const sp = smoothPostureRef.current;
      const ease = (
        key: keyof LiveStats['posture'],
        raw: number | null,
      ): number | null => {
        if (raw === null || !Number.isFinite(raw)) {
          sp[key] = null;
          return null;
        }
        sp[key] = sp[key] === null ? raw : sp[key]! + 0.3 * (raw - sp[key]!);
        return Math.round(sp[key]!);
      };

      const SL = lms[IDX.shoulderL];
      const SR = lms[IDX.shoulderR];
      const HL = lms[IDX.hipL];
      const HR = lms[IDX.hipR];
      const shouldersOk = isVis(SL, 0.5) && isVis(SR, 0.5);
      const hipsOk = isVis(HL, 0.5) && isVis(HR, 0.5);

      live.posture = {
        torso:
          shouldersOk && hipsOk
            ? ease('torso', tiltFromVertical(mid(HL, HR), mid(SL, SR), vw, vh))
            : ease('torso', null),
        shoulderTilt: shouldersOk
          ? ease('shoulderTilt', tiltFromHorizontal(SL, SR, vw, vh))
          : ease('shoulderTilt', null),
        hipTilt: hipsOk
          ? ease('hipTilt', tiltFromHorizontal(HL, HR, vw, vh))
          : ease('hipTilt', null),
        twist:
          shouldersOk && hipsOk
            ? ease('twist', torsoTwist(SL, SR, HL, HR))
            : ease('twist', null),
      };

      const def = exerciseByKey(exerciseRef.current);

      // ── free scan: sparkline follows knee/elbow angle ──
      if (!def.joints) {
        live.primary = out.kneeL ?? out.elbowL;
        live.effort = 0;
        live.phase = null;
        const sample = out.kneeL ?? out.elbowL;
        if (sample !== null && now - lastSparkAtRef.current > 110) {
          lastSparkAtRef.current = now;
          sparkRef.current.push(sample);
          if (sparkRef.current.length > 70) sparkRef.current.shift();
          live.spark = [...sparkRef.current];
        }
        return;
      }

      // ── exercise mode: smoothed 3D primary-joint angle ──
      const [ia, ib, ic] = def.joints;
      const A = lms[ia];
      const B = lms[ib];
      const C = lms[ic];
      let rawP: number | null = null;
      if (isVis(A, 0.5) && isVis(B, 0.5) && isVis(C, 0.5)) {
        const v = angle3D(A, B, C, vw, vh);
        if (Number.isFinite(v)) rawP = v;
      }
      if (rawP === null) {
        smoothPrimaryRef.current = null;
        live.primary = null;
        return;
      }
      smoothPrimaryRef.current =
        smoothPrimaryRef.current === null
          ? rawP
          : smoothPrimaryRef.current + 0.35 * (rawP - smoothPrimaryRef.current);
      const ang = smoothPrimaryRef.current;
      live.primary = Math.round(ang);
      live.effort = effortProgress(ang, def);

      // mirror angle for symmetry scoring (also smoothed)
      let mir: number | null = null;
      if (def.mirror) {
        const [ma, mb, mc] = def.mirror;
        const MA = lms[ma];
        const MB = lms[mb];
        const MC = lms[mc];
        let rawM: number | null = null;
        if (isVis(MA, 0.5) && isVis(MB, 0.5) && isVis(MC, 0.5)) {
          const v = angle3D(MA, MB, MC, vw, vh);
          if (Number.isFinite(v)) rawM = v;
        }
        if (rawM === null) {
          smoothMirrorRef.current = null;
        } else {
          smoothMirrorRef.current =
            smoothMirrorRef.current === null
              ? rawM
              : smoothMirrorRef.current +
                0.35 * (rawM - smoothMirrorRef.current);
          mir = smoothMirrorRef.current;
        }
      }

      repRef.current ??= freshRepMachine();
      const ev = feedRep(repRef.current, ang, mir, def, now);
      live.phase = ev.machine.phase;
      live.reps = ev.machine.count;

      if (now - lastSparkAtRef.current > 110) {
        lastSparkAtRef.current = now;
        sparkRef.current.push(Math.round(ang));
        if (sparkRef.current.length > 70) sparkRef.current.shift();
        live.spark = [...sparkRef.current];
      }

      if (ev.rep) {
        live.lastQuality = ev.quality;
        setRepFlash((f) => f + 1);
        beep();
      }
    },
    [beep],
  );

  // ── main loop ───────────────────────────────────────────────
  const loop = useCallback(() => {
    if (!runningRef.current) return;
    const video = videoRef.current;
    const lm = landmarkerRef.current;
    if (video && lm && video.readyState >= 2) {
      if (video.currentTime !== lastVideoTimeRef.current) {
        lastVideoTimeRef.current = video.currentTime;
        const now = performance.now();
        try {
          const res = lm.detectForVideo(video, now);
          const lms = res.landmarks?.[0] ?? null;
          lastFrameRef.current = {
            lms,
            vw: video.videoWidth,
            vh: video.videoHeight,
          };
          processFrame(lms, video.videoWidth, video.videoHeight, now);
          drawLive(lms, video.videoWidth, video.videoHeight);
        } catch {
          /* transient frame errors are safe to skip */
        }
        // keep an always-fresh RGB copy of the frame for photo capture.
        // Paint it while the video element is definitely presenting (inside
        // rAF with a fresh currentTime) — on some iOS builds, ad-hoc draws
        // from the video element return black frames, but this path works.
        try {
          const vw2 = video.videoWidth;
          const vh2 = video.videoHeight;
          if (vw2 && vh2) {
            let fc = frameCanvasRef.current;
            if (!fc) {
              fc = document.createElement('canvas');
              frameCanvasRef.current = fc;
            }
            if (fc.width !== vw2 || fc.height !== vh2) {
              fc.width = vw2;
              fc.height = vh2;
              frameCtxRef.current = fc.getContext('2d', {
                willReadFrequently: true,
              });
            }
            const fctx = frameCtxRef.current;
            if (fctx) {
              fctx.drawImage(video, 0, 0, vw2, vh2);
              frameDirtyRef.current = now;
            }
          }
        } catch {
          /* frame snapshot is best-effort */
        }
        const marks = fpsRef.current.marks;
        marks.push(now);
        while (marks.length && marks[0] < now - 1000) marks.shift();
        liveRef.current.fps = marks.length;
      }
    }
    rafRef.current = requestAnimationFrame(loop);
  }, [drawLive, processFrame, videoRef]);

  // ── open camera stream with fallbacks ───────────────────────
  const openStream = useCallback(
    async (facingMode: FacingState): Promise<MediaStream> => {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error('INSECURE_CONTEXT');
      }
      const attempts: MediaStreamConstraints[] = [
        {
          video: {
            width: { ideal: 1280 },
            height: { ideal: 720 },
            facingMode: { ideal: facingMode },
          },
          audio: false,
        },
        { video: { facingMode: { ideal: facingMode } }, audio: false },
        { video: true, audio: false },
      ];
      let lastErr: unknown = null;
      for (const c of attempts) {
        try {
          return await navigator.mediaDevices.getUserMedia(c);
        } catch (e) {
          lastErr = e;
          if (e instanceof DOMException && e.name === 'NotAllowedError') break;
        }
      }
      throw lastErr ?? new Error('camera unavailable');
    },
    [],
  );

  const attachStream = useCallback(
    async (stream: MediaStream) => {
      const video = videoRef.current;
      if (!video) throw new Error('no video element');
      streamRef.current = stream;
      video.muted = true;
      video.playsInline = true;
      video.setAttribute('playsinline', '');
      video.setAttribute('webkit-playsinline', '');
      try {
        video.srcObject = stream;
      } catch {
        video.src = URL.createObjectURL(stream as unknown as Blob);
      }
      try {
        await video.play();
      } catch {
        await new Promise<void>((resolve) => {
          const done = () => {
            void video.play().finally(() => resolve());
          };
          video.addEventListener('loadedmetadata', done, { once: true });
          setTimeout(resolve, 2000);
        });
      }
    },
    [videoRef],
  );

  // ── camera control ──────────────────────────────────────────
  const start = useCallback(
    async (auto = false) => {
      if (
        modelState !== 'ready' ||
        camState === 'running' ||
        camState === 'starting'
      )
        return;
      setCamState('starting');
      setCamError('');
      try {
        const stream = await openStream(facingRef.current);
        await attachStream(stream);

        storage.set(GRANTED_KEY, '1'); // remember the grant for next visits

        sessionRef.current = {
          startedAt: performance.now(),
          carried: sessionRef.current.carried,
        };
        runningRef.current = true;
        lastVideoTimeRef.current = -1;
        setCamState('running');
        rafRef.current = requestAnimationFrame(loop);

        try {
          audioRef.current ??= new AudioContext();
          if (audioRef.current.state === 'suspended')
            void audioRef.current.resume();
        } catch {
          /* ignore */
        }
        if (soundRef.current) {
          beep(520, 0.08, 0.04);
          setTimeout(() => beep(760, 0.08, 0.04), 110);
        }
      } catch (e) {
        console.error(e);
        if (
          auto &&
          e instanceof DOMException &&
          (e.name === 'NotAllowedError' || e.name === 'PermissionDeniedError')
        ) {
          setCamState('idle'); // needs a fresh user gesture
          return;
        }
        setCamState('error');
        setCamError(cameraErrorMessage(e));
      }
    },
    [modelState, camState, openStream, attachStream, loop, beep],
  );

  // ── auto-start if permission was granted before ─────────────
  useEffect(() => {
    if (autoTriedRef.current || modelState !== 'ready') return;
    autoTriedRef.current = true;
    if (storage.get(GRANTED_KEY) === '1') void start(true);
  }, [modelState, start]);

  const stop = useCallback(() => {
    runningRef.current = false;
    cancelAnimationFrame(rafRef.current);
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    const video = videoRef.current;
    if (video) video.srcObject = null;
    if (sessionRef.current.startedAt) {
      sessionRef.current.carried +=
        performance.now() - sessionRef.current.startedAt;
      sessionRef.current.startedAt = 0;
    }
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (canvas && ctx)
      ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
    liveRef.current = {
      ...liveRef.current,
      fps: 0,
      present: false,
      tracked: 0,
      primary: null,
      phase: null,
      effort: 0,
      angles: EMPTY_ANGLES,
    };
    setCamState('idle');
  }, [videoRef, canvasRef]);

  const reset = useCallback(() => {
    repRef.current = null;
    sparkRef.current = [];
    sessionRef.current = {
      startedAt: runningRef.current ? performance.now() : 0,
      carried: 0,
    };
    liveRef.current = {
      ...liveRef.current,
      reps: 0,
      phase: null,
      lastQuality: null,
      spark: [],
      seconds: 0,
    };
    setStats((s) => ({
      ...s,
      reps: 0,
      lastQuality: null,
      phase: null,
      spark: [],
      seconds: 0,
    }));
  }, []);

  // ── flip front/back camera ──────────────────────────────────
  const [flipBusy, setFlipBusy] = useState(false);
  const flipCamera = useCallback(async () => {
    if (flipBusy) return;
    const next: FacingState =
      facingRef.current === 'user' ? 'environment' : 'user';
    facingRef.current = next;
    setFacing(next);
    if (!runningRef.current) return;
    setFlipBusy(true);
    try {
      const stream = await openStream(next);
      streamRef.current?.getTracks().forEach((t) => t.stop());
      await attachStream(stream);
      lastVideoTimeRef.current = -1;
    } catch (e) {
      console.error('flip failed', e);
    } finally {
      setFlipBusy(false);
    }
  }, [flipBusy, openStream, attachStream]);

  // ── photo capture: exactly what's on screen (frame + skeleton) ──
  // Uses several drawing strategies and validates pixels, because
  // direct video→canvas draws return black frames on some iOS builds.
  const capture = useCallback(async (): Promise<boolean> => {
    const video = videoRef.current;
    if (!video || !video.videoWidth) return false;
    const vw = video.videoWidth;
    const vh = video.videoHeight;
    const overlay = canvasRef.current;
    const cw = Math.max(1, overlay?.clientWidth ?? vw);
    const ch = Math.max(1, overlay?.clientHeight ?? vh);

    // visible region of the frame (matches the object-cover display)
    const sCover = Math.max(cw / vw, ch / vh);
    const sw = Math.min(vw, Math.max(1, Math.round(cw / sCover)));
    const sh = Math.min(vh, Math.max(1, Math.round(ch / sCover)));
    const sx = Math.max(0, Math.round((vw - sw) / 2));
    const sy = Math.max(0, Math.round((vh - sh) / 2));

    const c = document.createElement('canvas');
    c.width = sw;
    c.height = sh;
    const ctx = c.getContext('2d');
    if (!ctx) return false;

    const mirrorIt = facingRef.current === 'user';
    const paintMirrored = (src: CanvasImageSource) => {
      ctx.save();
      if (mirrorIt) {
        ctx.translate(sw, 0);
        ctx.scale(-1, 1);
      }
      ctx.drawImage(src, 0, 0, sw, sh);
      ctx.restore();
    };

    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, sw, sh);

    let painted = false;

    // method 0 — the fresh frame snapshot taken inside the render loop
    // (works where ad-hoc video draws return black on iOS)
    const fc = frameCanvasRef.current;
    if (
      fc &&
      frameCtxRef.current &&
      performance.now() - frameDirtyRef.current < 400 &&
      !frameAlmostBlack(frameCtxRef.current, fc.width, fc.height)
    ) {
      try {
        ctx.save();
        if (mirrorIt) {
          ctx.translate(sw, 0);
          ctx.scale(-1, 1);
        }
        ctx.drawImage(fc, sx, sy, sw, sh, 0, 0, sw, sh);
        ctx.restore();
        painted = !frameAlmostBlack(ctx, sw, sh);
      } catch {
        /* try next */
      }
    }

    // method 1 — createImageBitmap from the live video (most reliable on iOS)
    if (!painted && typeof createImageBitmap === 'function') {
      try {
        const bmp = await createImageBitmap(video, sx, sy, sw, sh);
        paintMirrored(bmp);
        bmp.close?.();
        painted = !frameAlmostBlack(ctx, sw, sh);
      } catch {
        /* try next */
      }
    }

    // method 2 — full frame via an intermediate canvas, then crop
    if (!painted) {
      try {
        const tmp = document.createElement('canvas');
        tmp.width = vw;
        tmp.height = vh;
        const tctx = tmp.getContext('2d');
        if (tctx) {
          tctx.drawImage(video, 0, 0, vw, vh);
          ctx.save();
          if (mirrorIt) {
            ctx.translate(sw, 0);
            ctx.scale(-1, 1);
          }
          ctx.drawImage(tmp, sx, sy, sw, sh, 0, 0, sw, sh);
          ctx.restore();
          painted = !frameAlmostBlack(ctx, sw, sh);
        }
      } catch {
        /* try next */
      }
    }

    // method 3 — classic direct cropped drawImage
    if (!painted) {
      try {
        ctx.save();
        if (mirrorIt) {
          ctx.translate(sw, 0);
          ctx.scale(-1, 1);
        }
        ctx.drawImage(video, sx, sy, sw, sh, 0, 0, sw, sh);
        ctx.restore();
      } catch {
        /* leave skeleton-only result */
      }
    }

    // skeleton overlay aligned to the same crop
    const lms = lastFrameRef.current?.lms ?? null;
    drawScene(ctx, sw, sh, lms, vw, vh, { sx, sy, sw, sh });

    // watermark
    ctx.font = '600 12px Tahoma, sans-serif';
    ctx.fillStyle = 'rgba(250,204,21,0.75)';
    const wm = 'آنالیز سعیدکلاب · SAEED CLUB';
    ctx.fillText(wm, sw - ctx.measureText(wm).width - 14, sh - 14);

    try {
      const url = c.toDataURL('image/jpeg', 0.92);
      const a = document.createElement('a');
      a.href = url;
      a.download = `pose-scan-${new Date()
        .toISOString()
        .replace(/[:.]/g, '-')}.jpg`;
      document.body.appendChild(a);
      a.click();
      a.remove();
    } catch (e) {
      console.error('capture failed', e);
      return false;
    }
    return true;
  }, [videoRef, canvasRef, drawScene]);

  // ── stats pump (8Hz) ────────────────────────────────────────
  useEffect(() => {
    const id = setInterval(() => {
      const sess = sessionRef.current;
      const secs = sess.startedAt
        ? sess.carried + (performance.now() - sess.startedAt)
        : sess.carried;
      liveRef.current.seconds = Math.floor(secs / 1000);
      setStats({ ...liveRef.current });
    }, 125);
    return () => clearInterval(id);
  }, []);

  // ── canvas resize handling ──────────────────────────────────
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const parent = canvas.parentElement;
    if (!parent) return;
    const ro = new ResizeObserver(() => {
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      canvas.width = Math.round(parent.clientWidth * dpr);
      canvas.height = Math.round(parent.clientHeight * dpr);
      const ctx = canvas.getContext('2d');
      ctx?.setTransform(dpr, 0, 0, dpr, 0, 0);
    });
    ro.observe(parent);
    return () => ro.disconnect();
  }, [canvasRef]);

  // ── resume camera when returning to the tab ─────────────────
  useEffect(() => {
    const onVis = () => {
      const video = videoRef.current;
      if (
        document.visibilityState === 'visible' &&
        runningRef.current &&
        video &&
        video.paused
      ) {
        lastVideoTimeRef.current = -1;
        void video.play().catch(() => undefined);
      }
    };
    document.addEventListener('visibilitychange', onVis);
    return () => document.removeEventListener('visibilitychange', onVis);
  }, [videoRef]);

  // ── teardown on unmount ─────────────────────────────────────
  useEffect(() => {
    return () => {
      runningRef.current = false;
      cancelAnimationFrame(rafRef.current);
      streamRef.current?.getTracks().forEach((t) => t.stop());
      void landmarkerRef.current?.close();
      void audioRef.current?.close().catch(() => undefined);
    };
  }, []);

  return {
    modelState,
    camState,
    camError,
    stats,
    repFlash,
    facing,
    flipBusy,
    start,
    stop,
    reset,
    flipCamera,
    capture,
    beep,
  };
}
