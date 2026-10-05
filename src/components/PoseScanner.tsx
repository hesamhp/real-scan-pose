import { useRef, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import {
  Camera,
  CameraOff,
  Check,
  Info,
  Loader2,
  Power,
  ScanLine,
  ShieldCheck,
  SwitchCamera,
  TriangleAlert,
  Volume2,
  VolumeX,
  Workflow,
  Zap,
} from 'lucide-react';
import { usePoseEngine, type LiveStats } from '../hooks/usePoseEngine';
import { StatusDot } from './ui';

const JOINT_CHIPS: { label: string; key: keyof LiveStats['angles'] }[] = [
  { label: 'آرنج چپ', key: 'elbowL' },
  { label: 'آرنج راست', key: 'elbowR' },
  { label: 'زانو چپ', key: 'kneeL' },
  { label: 'زانو راست', key: 'kneeR' },
  { label: 'باسن', key: 'hip' },
  { label: 'شانه', key: 'shoulder' },
];

/** segment-to-segment readouts (relations between major joints) */
const POSTURE_CHIPS: {
  label: string;
  key: keyof LiveStats['posture'];
  tint: string;
  /** good when |value| is below this */
  tol: number;
}[] = [
  { label: 'تنه', key: 'torso', tint: 'text-yellow-200', tol: 12 },
  { label: 'خط شانه', key: 'shoulderTilt', tint: 'text-amber-200', tol: 7 },
  { label: 'خط لگن', key: 'hipTilt', tint: 'text-orange-200', tol: 7 },
  { label: 'چرخش تنه', key: 'twist', tint: 'text-yellow-300', tol: 12 },
];

const fmtTime = (s: number) =>
  `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;

function MiniRing({ value }: { value: number | null }) {
  const r = 20;
  const c = 2 * Math.PI * r;
  const pct = value === null ? 0 : Math.min(1, value / 100);
  return (
    <div className="relative h-12 w-12 shrink-0">
      <svg viewBox="0 0 48 48" className="h-full w-full -rotate-90">
        <circle
          cx="24"
          cy="24"
          r={r}
          fill="rgba(10,8,0,0.6)"
          stroke="rgba(250,204,21,0.15)"
          strokeWidth="4"
        />
        <circle
          cx="24"
          cy="24"
          r={r}
          fill="none"
          stroke="#facc15"
          strokeWidth="4"
          strokeLinecap="round"
          strokeDasharray={c}
          strokeDashoffset={c * (1 - pct)}
          style={{
            transition: 'stroke-dashoffset 0.45s cubic-bezier(0.22,1,0.36,1)',
            filter: 'drop-shadow(0 0 5px rgba(250,204,21,0.5))',
          }}
        />
      </svg>
      <span className="absolute inset-0 grid place-items-center font-mono text-[11px] font-bold text-white tabular">
        {value === null ? '--' : value}
      </span>
    </div>
  );
}

function IconBtn({
  onClick,
  title,
  active,
  disabled,
  children,
}: {
  onClick: () => void;
  title: string;
  active?: boolean;
  disabled?: boolean;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      title={title}
      aria-label={title}
      disabled={disabled}
      className={`grid h-11 w-11 shrink-0 touch-manipulation place-items-center rounded-full border backdrop-blur-md transition-all active:scale-90 disabled:opacity-40 ${
        active
          ? 'border-yellow-300/60 bg-yellow-300/15 text-yellow-300'
          : 'border-white/15 bg-black/45 text-slate-200 hover:border-yellow-300/50'
      }`}
    >
      {children}
    </button>
  );
}

const scrollHide =
  'overflow-x-auto [scrollbar-width:none] [-ms-overflow-style:none] [&::-webkit-scrollbar]:hidden';

export default function PoseScanner() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [soundOn, setSoundOn] = useState(false);
  const [showHint, setShowHint] = useState(false);
  const [shotFlash, setShotFlash] = useState(0);
  const [shotAt, setShotAt] = useState<number | null>(null);
  const toastTimer = useRef<ReturnType<typeof setTimeout>>(undefined);

  const {
    modelState,
    camState,
    camError,
    stats,
    facing,
    flipBusy,
    start,
    stop,
    flipCamera,
    capture,
    beep,
  } = usePoseEngine(videoRef, canvasRef, 'free', soundOn);

  const running = camState === 'running';
  const coverage = stats.present ? Math.round((stats.tracked / 33) * 100) : 0;

  const onShutter = async () => {
    if (!running) return;
    try {
      if ('requestVideoFrameCallback' in HTMLVideoElement.prototype) {
        await new Promise<void>((r) => {
          videoRef.current?.requestVideoFrameCallback(() => r());
        });
      }
    } catch {
      /* ignore */
    }
    const ok = await capture();
    if (!ok) return;
    beep(1250, 0.06, 0.06);
    setShotFlash((f) => f + 1);
    setShotAt(Date.now());
    clearTimeout(toastTimer.current);
    toastTimer.current = setTimeout(() => setShotAt(null), 1800);
  };

  return (
    <div className="fixed inset-0 select-none overflow-hidden bg-black text-slate-100">
      {/* ── camera feed ── */}
      <video
        ref={videoRef}
        playsInline
        muted
        autoPlay
        className={`absolute inset-0 h-full w-full object-cover transition-transform duration-300 ${
          facing === 'user' ? '-scale-x-100' : ''
        }`}
      />
      <canvas
        ref={canvasRef}
        className="pointer-events-none absolute inset-0 h-full w-full"
      />

      {/* ── HUD deco ── */}
      {running && (
        <>
          <div className="pointer-events-none absolute inset-3 z-10 hud-corners opacity-60" />
          <div className="pointer-events-none absolute inset-0 z-10 bg-grid-faint opacity-30 [mask-image:linear-gradient(to_bottom,black_60%,transparent)]" />
          <div className="pointer-events-none absolute inset-x-0 z-10 h-28 animate-scan-y bg-gradient-to-b from-transparent via-yellow-300/15 to-transparent" />
        </>
      )}

      {/* ── shutter golden flash ── */}
      <AnimatePresence>
        {shotFlash > 0 && (
          <motion.div
            key={`shot-${shotFlash}`}
            initial={{ opacity: 0.85 }}
            animate={{ opacity: 0 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.45 }}
            className="pointer-events-none absolute inset-0 z-30 bg-yellow-200"
          />
        )}
      </AnimatePresence>

      {/* ── photo saved toast ── */}
      <AnimatePresence>
        {shotAt && (
          <motion.div
            key={shotAt}
            initial={{ opacity: 0, y: 16, scale: 0.92 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 10, scale: 0.95 }}
            transition={{ duration: 0.3 }}
            className="pointer-events-none absolute inset-x-0 top-[22%] z-30 flex justify-center"
          >
            <span className="flex items-center gap-2 rounded-full border border-yellow-300/40 bg-black/70 px-4 py-2 text-[12px] font-bold text-yellow-300 backdrop-blur-md shadow-[0_0_24px_rgba(250,204,21,0.2)]">
              <Check className="h-3.5 w-3.5" />
              عکس با اسکلت ذخیره شد
            </span>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ── top bar ── */}
      <div className="pointer-events-none absolute inset-x-0 top-0 z-20 bg-gradient-to-b from-black/75 via-black/35 to-transparent pb-10">
        <div className="px-4 pt-[calc(env(safe-area-inset-top,0px)+10px)]">
          <div className="flex items-center justify-between">
            <div className="pointer-events-auto flex items-center gap-2">
              <span className="grid h-8 w-8 place-items-center rounded-lg border border-yellow-300/30 bg-black/50 backdrop-blur-md">
                <ScanLine className="h-4 w-4 text-yellow-300" />
              </span>
              <span className="flex flex-col leading-none">
                <span className="text-[13px] font-extrabold text-white">
                  آنالیز سعیدکلاب
                </span>
                <span className="mt-0.5 flex items-center gap-1 text-[9px] text-slate-400">
                  <StatusDot
                    tone={
                      modelState === 'ready'
                        ? 'ok'
                        : modelState === 'loading'
                          ? 'warn'
                          : 'err'
                    }
                  />
                  {modelState === 'ready'
                    ? 'اسکن آزاد · مدل آماده'
                    : modelState === 'loading'
                      ? 'بارگذاری مدل…'
                      : 'خطای مدل'}
                </span>
              </span>
            </div>
            <div className="pointer-events-auto flex items-center gap-3 font-mono text-[10px] text-slate-300">
              {running && (
                <span className="flex items-center gap-1.5 tracking-widest text-yellow-300">
                  <span className="h-1.5 w-1.5 animate-blink rounded-full bg-rose-500" />
                  LIVE
                </span>
              )}
              <span className="tabular">
                <span className="text-slate-500">T </span>
                <b className="text-yellow-300">{fmtTime(stats.seconds)}</b>
              </span>
              <span className="tabular">
                <span className="text-slate-500">FPS </span>
                <b className={stats.fps > 24 ? 'text-yellow-300' : 'text-amber-300'}>
                  {running ? stats.fps : '--'}
                </b>
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* ── searching subject ── */}
      {running && !stats.present && (
        <div className="pointer-events-none absolute inset-0 z-20 flex items-center justify-center px-6">
          <div className="flex flex-col items-center gap-2.5 rounded-2xl border border-dashed border-yellow-300/40 bg-black/60 px-7 py-5 text-center backdrop-blur-md">
            <Workflow className="h-5 w-5 animate-pulse-soft text-yellow-300" />
            <span className="text-sm font-semibold text-yellow-200">
              در جست‌وجوی سوژه…
            </span>
            <span className="max-w-[240px] text-[11px] leading-5 text-slate-300">
              گوشی را ثابت بگیرید تا کل بدن فرد در کادر قرار گیرد
            </span>
          </div>
        </div>
      )}

      {/* ── bottom dock ── */}
      {running && (
        <div className="absolute inset-x-0 bottom-0 z-20 bg-gradient-to-t from-black/85 via-black/45 to-transparent pt-10">
          <div className="mx-auto w-full max-w-xl px-4 pb-[calc(env(safe-area-inset-bottom,0px)+14px)]">
            {/* hint popover */}
            <AnimatePresence>
              {showHint && (
                <motion.div
                  initial={{ opacity: 0, y: 12, scale: 0.97 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, y: 12, scale: 0.97 }}
                  transition={{ duration: 0.25 }}
                  className="mb-3 rounded-2xl border border-yellow-300/25 bg-black/75 p-3.5 text-[12px] leading-6 text-slate-200 backdrop-blur-lg"
                >
                  <b className="text-yellow-300">اسکن آزاد — </b>
                  کل بدن داخل کادر باشید. زاویه‌ی مفاصل، وضعیت تنه و خط لگن و شانه
                  به‌صورت زنده تحلیل می‌شود و با دکمه‌ی شاتر می‌توانید عکسی با
                  اسکلت ذخیره کنید.
                </motion.div>
              )}
            </AnimatePresence>

            {/* posture chips — relations between major joints */}
            <div className={`flex gap-2 pb-2 ${scrollHide}`}>
              {POSTURE_CHIPS.map((p) => {
                const v = stats.posture[p.key];
                const ok = v !== null && Math.abs(v) <= p.tol;
                return (
                  <span
                    key={p.key}
                    className={`flex shrink-0 items-center gap-1.5 rounded-full border px-3 py-1.5 backdrop-blur-md transition-colors ${
                      v === null
                        ? 'border-white/10 bg-black/45'
                        : ok
                          ? 'border-yellow-300/35 bg-yellow-300/10'
                          : 'border-amber-300/40 bg-amber-400/10'
                    }`}
                  >
                    <span className="text-[10px] text-slate-300">{p.label}</span>
                    <b
                      className={`font-mono text-[12px] tabular ${
                        v === null ? 'text-slate-500' : ok ? 'text-yellow-300' : 'text-amber-300'
                      }`}
                    >
                      {v === null ? '--' : `${Math.abs(v)}°`}
                    </b>
                    <span className={`text-[9px] ${p.tint}`}>
                      {v === null ? '' : ok ? '✓' : '!'}
                    </span>
                  </span>
                );
              })}
            </div>

            {/* joint chips */}
            <div className={`flex gap-2 pb-3 ${scrollHide}`}>
              {JOINT_CHIPS.map((j) => (
                <span
                  key={j.key}
                  className="flex shrink-0 items-center gap-1.5 rounded-full border border-yellow-300/15 bg-black/45 px-3 py-1.5 backdrop-blur-md"
                >
                  <span className="text-[10px] text-slate-400">{j.label}</span>
                  <b className="font-mono text-[12px] text-yellow-300 tabular">
                    {stats.angles[j.key] === null ? '--' : `${stats.angles[j.key]}°`}
                  </b>
                </span>
              ))}
            </div>

            {/* main stats row — body coverage */}
            <div className="flex items-center gap-4 rounded-2xl border border-white/10 bg-black/50 px-4 py-3 backdrop-blur-lg">
              <div className="flex items-end gap-1.5">
                <motion.span
                  key={stats.tracked}
                  animate={{ scale: 1 }}
                  className="font-mono text-[40px] font-bold leading-none text-yellow-100 tabular"
                >
                  {String(stats.tracked).padStart(2, '0')}
                </motion.span>
                <div className="flex flex-col pb-0.5">
                  <span className="font-mono text-[11px] text-slate-500">/33</span>
                  <span className="flex items-center gap-1 text-[9px] text-slate-500">
                    <StatusDot tone={stats.present ? 'ok' : 'idle'} />
                    نقاط بدن
                  </span>
                </div>
              </div>

              <div className="min-w-0 flex-1">
                <div className="mb-1 flex justify-between font-mono text-[8px] tracking-widest text-slate-500">
                  <span>پوشش اسکلت</span>
                  <span className="text-yellow-300">{coverage}%</span>
                </div>
                <div className="h-1.5 overflow-hidden rounded-full bg-white/10">
                  <div
                    className="h-full rounded-full bg-gradient-to-l from-yellow-300 to-amber-400 transition-all duration-300"
                    style={{ width: `${coverage}%` }}
                  />
                </div>
                <div className="mt-1.5 flex items-center justify-between font-mono text-[8px] tracking-widest text-slate-500">
                  <span>FREE·SCAN</span>
                  <span className="text-yellow-300/80">
                    {stats.angles.kneeL === null ? '--°' : `زانو ${stats.angles.kneeL}°`}
                  </span>
                </div>
              </div>

              <MiniRing value={stats.present ? coverage : null} />
            </div>

            {/* controls */}
            <div className="mt-3 flex items-center justify-center gap-3.5">
              <IconBtn
                onClick={() => setShowHint((v) => !v)}
                title="راهنما"
                active={showHint}
              >
                <Info className="h-4 w-4" />
              </IconBtn>
              <IconBtn
                onClick={() => setSoundOn((v) => !v)}
                title={soundOn ? 'بی‌صدا' : 'باصدا'}
                active={soundOn}
              >
                {soundOn ? (
                  <Volume2 className="h-4 w-4" />
                ) : (
                  <VolumeX className="h-4 w-4" />
                )}
              </IconBtn>

              {/* photo shutter */}
              <button
                onClick={onShutter}
                aria-label="گرفتن عکس"
                className="group relative grid h-16 w-16 touch-manipulation place-items-center rounded-full border-[3px] border-white/90 bg-white/10 shadow-[0_0_30px_rgba(255,255,255,0.25)] transition-all active:scale-90"
              >
                <span className="grid h-[48px] w-[48px] place-items-center rounded-full bg-white text-[#04070c] transition group-active:bg-lime-300">
                  <Camera className="h-5 w-5" />
                </span>
              </button>

              <IconBtn onClick={flipCamera} title="تعویض دوربین" disabled={flipBusy}>
                {flipBusy ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <SwitchCamera className="h-4 w-4" />
                )}
              </IconBtn>
              <IconBtn onClick={stop} title="خاموش کردن دوربین">
                <Power className="h-4 w-4 text-rose-300" />
              </IconBtn>
            </div>
          </div>
        </div>
      )}

      {/* ── overlays (idle / loading / error) ── */}
      <AnimatePresence>
        {!running && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.4 }}
            className="absolute inset-0 z-40 flex flex-col items-center justify-center bg-[#0a0800] p-6 text-center"
          >
            <div className="pointer-events-none absolute inset-0 bg-grid-faint opacity-30" />
            <div className="pointer-events-none absolute left-1/2 top-1/4 h-72 w-72 -translate-x-1/2 rounded-full bg-yellow-500/12 blur-[100px]" />

            {modelState === 'loading' || camState === 'starting' ? (
              <>
                <div className="relative mb-6 grid h-20 w-20 place-items-center rounded-full border border-yellow-300/25 bg-yellow-400/8">
                  <Loader2 className="h-8 w-8 animate-spin text-yellow-300" />
                </div>
                <p className="relative text-lg font-extrabold text-white">
                  {camState === 'starting'
                    ? 'در حال روشن شدن دوربین…'
                    : 'در حال بارگذاری مدل هوش مصنوعی'}
                </p>
                <p className="relative mt-2 font-mono text-[10px] tracking-[0.25em] text-yellow-300/70">
                  {camState === 'starting'
                    ? 'OPENING CAMERA…'
                    : 'LOADING NEURAL NETWORK…'}
                </p>
              </>
            ) : modelState === 'error' ? (
              <>
                <TriangleAlert className="relative mb-5 h-10 w-10 text-rose-400" />
                <p className="relative max-w-sm text-sm leading-7 text-slate-300">
                  بارگذاری مدل ناموفق بود. اتصال اینترنت را بررسی کنید و صفحه را
                  تازه‌سازی کنید.
                </p>
              </>
            ) : camState === 'error' ? (
              <>
                <CameraOff className="relative mb-5 h-10 w-10 text-rose-400" />
                <p className="relative mb-6 max-w-sm text-sm leading-7 text-slate-300">
                  {camError}
                </p>
                <button
                  onClick={() => void start()}
                  className="relative flex touch-manipulation items-center gap-2 rounded-2xl bg-yellow-400 px-7 py-3.5 text-sm font-extrabold text-[#0a0800] shadow-[0_0_30px_rgba(250,204,21,0.4)] transition active:scale-95"
                >
                  <Camera className="h-4 w-4" />
                  تلاش دوباره
                </button>
              </>
            ) : (
              <>
                <div className="relative mb-6">
                  <span className="absolute inset-0 animate-ping rounded-full bg-yellow-400/25" />
                  <span className="relative grid h-20 w-20 place-items-center rounded-full border border-yellow-300/30 bg-yellow-400/10 shadow-[0_0_50px_rgba(250,204,21,0.3)]">
                    <ScanLine className="h-8 w-8 text-yellow-300" />
                  </span>
                </div>
                <h1 className="relative text-3xl font-black text-white">
                  آنالیز سعیدکلاب
                </h1>
                <p className="relative mt-3 max-w-xs text-[13px] leading-6 text-slate-400">
                  اسکلت بدن، زاویه‌ی مفاصل و وضعیت تنه به‌صورت زنده تحلیل می‌شود
                  و با شاتر عکس می‌گیرید.
                </p>
                <div className="relative mt-5 flex items-center gap-2 text-[10px] text-slate-500">
                  <ShieldCheck className="h-3.5 w-3.5 text-yellow-300" />
                  پردازش ۱۰۰٪ محلی؛ هیچ تصویری ارسال نمی‌شود
                </div>
                <button
                  onClick={() => void start()}
                  className="relative mt-8 flex touch-manipulation items-center gap-2.5 rounded-2xl bg-yellow-400 px-9 py-4 text-base font-extrabold text-[#0a0800] shadow-[0_0_40px_rgba(250,204,21,0.5)] transition active:scale-95"
                >
                  <Zap className="h-5 w-5" />
                  شروع اسکن
                </button>
                <p className="relative mt-4 max-w-[260px] text-[10px] leading-5 text-slate-600">
                  فقط بار اول اجازه دوربین پرسیده می‌شود؛ دفعات بعد اسکن به‌صورت
                  خودکار شروع می‌شود.
                </p>
              </>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
