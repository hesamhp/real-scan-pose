import { useEffect, useRef, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import {
  Camera,
  Check,
  CircleStop,
  Loader2,
  Power,
  ScanLine,
  SwitchCamera,
  Video,
  Zap,
} from 'lucide-react';
import { usePoseEngine } from '../hooks/usePoseEngine';

const fmtTime = (s: number) =>
  `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;

export default function PoseScanner() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [shotFlash, setShotFlash] = useState(0);
  const [shotAt, setShotAt] = useState<number | null>(null);
  const [queuedStart, setQueuedStart] = useState(false);
  const toastTimer = useRef<ReturnType<typeof setTimeout>>(undefined);
  const [recSeconds, setRecSeconds] = useState(0);
  const recInterval = useRef<ReturnType<typeof setInterval>>(undefined);

  const {
    modelState, camState, stats, facing, flipBusy,
    start, stop, flipCamera, capture,
    recording, startRecording, stopRecording,
  } = usePoseEngine(videoRef, canvasRef, 'free', false);

  const running = camState === 'running';
  const coverage = stats.present ? Math.round((stats.tracked / 33) * 100) : 0;
  const busy = modelState === 'loading' || camState === 'starting';

  // auto-start queue
  useEffect(() => {
    if (queuedStart && modelState === 'ready' && camState === 'idle') {
      setQueuedStart(false);
      void start();
    }
  }, [queuedStart, modelState, camState, start]);

  const handleStart = () => {
    if (modelState === 'loading') { setQueuedStart(true); return; }
    if (modelState === 'ready') void start();
  };

  // recording timer
  useEffect(() => {
    if (recording) {
      setRecSeconds(0);
      recInterval.current = setInterval(() => setRecSeconds(s => s + 1), 1000);
    } else {
      clearInterval(recInterval.current);
      setRecSeconds(0);
    }
    return () => clearInterval(recInterval.current);
  }, [recording]);

  const onShutter = async () => {
    if (!running) return;
    const ok = await capture();
    if (!ok) return;
    setShotFlash(f => f + 1);
    setShotAt(Date.now());
    clearTimeout(toastTimer.current);
    toastTimer.current = setTimeout(() => setShotAt(null), 1800);
  };

  const toggleRecord = () => {
    if (recording) stopRecording();
    else startRecording();
  };

  return (
    <div className="fixed inset-0 select-none overflow-hidden bg-black text-slate-100 font-sans">
      <video ref={videoRef} playsInline muted autoPlay
        className={`absolute inset-0 h-full w-full object-cover ${facing === 'user' ? '-scale-x-100' : ''}`} />
      <canvas ref={canvasRef} className="pointer-events-none absolute inset-0 h-full w-full" />

      {/* ── Top HUD ── */}
      <div className="pointer-events-none absolute inset-x-0 top-0 z-20 bg-gradient-to-b from-black/50 to-transparent pb-8">
        <div className="flex items-center justify-between px-3 pt-[calc(env(safe-area-inset-top,0px)+8px)]">
          <button onClick={stop}
            className="pointer-events-auto flex items-center gap-1.5 rounded-lg bg-black/30 px-2.5 py-1.5 text-[11px] font-bold text-white backdrop-blur-md border border-white/5 active:scale-95">
            <Power className="h-3.5 w-3.5 text-rose-400" /> Exit
          </button>
          <div className="text-center">
            <div className="text-[12px] font-black text-white uppercase tracking-tight">Live Analysis</div>
            {running && !recording && (
              <div className="flex items-center justify-center gap-1 font-mono text-[8px] font-bold text-yellow-400">
                <span className="h-1 w-1 animate-pulse rounded-full bg-yellow-400" /> ACTIVE
              </div>
            )}
            {recording && (
              <div className="flex items-center justify-center gap-1.5 font-mono text-[9px] font-bold text-rose-400">
                <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-rose-500" />
                REC {fmtTime(recSeconds)}
              </div>
            )}
          </div>
          <button onClick={flipCamera} disabled={flipBusy}
            className="pointer-events-auto grid h-8 w-8 place-items-center rounded-lg bg-black/30 text-white backdrop-blur-md border border-white/5 active:scale-90 disabled:opacity-40">
            {flipBusy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <SwitchCamera className="h-3.5 w-3.5" />}
          </button>
        </div>
      </div>

      {/* ── Flash + Toast ── */}
      <AnimatePresence>
        {shotFlash > 0 && (
          <motion.div key={`f${shotFlash}`} initial={{ opacity: 0.8 }} animate={{ opacity: 0 }} exit={{ opacity: 0 }}
            transition={{ duration: 0.4 }} className="pointer-events-none absolute inset-0 z-[60] bg-yellow-50" />
        )}
      </AnimatePresence>
      <AnimatePresence>
        {shotAt && (
          <motion.div key={shotAt} initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 8 }}
            className="pointer-events-none absolute inset-x-0 top-[14%] z-50 flex justify-center">
            <span className="flex items-center gap-2 rounded-full bg-black/80 px-3 py-1.5 text-[10px] font-bold text-yellow-300 backdrop-blur-xl border border-yellow-300/30">
              <Check className="h-3 w-3" /> ذخیره شد
            </span>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ── Recording border pulse ── */}
      {recording && (
        <div className="pointer-events-none absolute inset-0 z-10 border-[3px] border-rose-500/60 animate-pulse" />
      )}

      {/* ── Bottom controls ── */}
      {running && (
        <div className="absolute inset-x-0 bottom-0 z-20 bg-gradient-to-t from-black/60 to-transparent pt-8">
          <div className="mx-auto w-full max-w-md px-4 pb-[calc(env(safe-area-inset-bottom,0px)+6px)]">

            {/* Micro stats */}
            <div className="flex items-center justify-between rounded-xl bg-black/30 px-3 py-1.5 backdrop-blur-xl border border-white/5 mb-3">
              <div className="flex items-center gap-3">
                <div className="flex flex-col">
                  <span className="text-[7px] font-black text-white/25 uppercase">Coverage</span>
                  <span className="font-mono text-[14px] font-black text-yellow-400 leading-none">{coverage}%</span>
                </div>
                <div className="h-4 w-px bg-white/5" />
                <div className="flex flex-col">
                  <span className="text-[7px] font-black text-white/25 uppercase">Time</span>
                  <span className="font-mono text-[14px] font-black text-white leading-none tabular-nums">{fmtTime(stats.seconds)}</span>
                </div>
              </div>
              <div className="flex items-center gap-1.5 rounded bg-black/30 px-1.5 py-0.5 border border-white/5">
                <span className="h-1 w-1 animate-pulse rounded-full bg-yellow-400" />
                <span className="font-mono text-[8px] font-bold text-yellow-400">{stats.fps}fps</span>
              </div>
            </div>

            {/* Shutter row: Record | Photo | (placeholder) */}
            <div className="flex items-center justify-center gap-6">
              
              {/* Record button */}
              <button onClick={toggleRecord}
                className={`relative grid h-12 w-12 place-items-center rounded-full border-[3px] transition-all active:scale-90 ${
                  recording
                    ? 'border-rose-500 bg-rose-500/20'
                    : 'border-white/40 bg-black/30'
                }`}>
                {recording ? (
                  <CircleStop className="h-5 w-5 text-rose-400" />
                ) : (
                  <Video className="h-5 w-5 text-white/70" />
                )}
                {recording && (
                  <span className="absolute -top-1 -right-1 h-3 w-3 rounded-full bg-rose-500 animate-pulse" />
                )}
              </button>

              {/* Photo shutter (center, larger) */}
              <button onClick={onShutter}
                className="group relative flex h-16 w-16 items-center justify-center active:scale-90 transition-transform">
                <div className="absolute inset-0 rounded-full border-[4px] border-white/80" />
                <div className="h-11 w-11 rounded-full bg-white group-active:bg-yellow-400 transition-colors shadow-xl" />
                <Camera className="absolute h-5 w-5 text-black" />
              </button>

              {/* Spacer for symmetry */}
              <div className="h-12 w-12" />
            </div>
          </div>
        </div>
      )}

      {/* ── Idle Overlay ── */}
      <AnimatePresence>
        {!running && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            className="absolute inset-0 z-40 flex flex-col items-center justify-center bg-[#0a0800] p-6 text-center">
            <div className="pointer-events-none absolute inset-0 bg-grid-faint opacity-30" />
            <div className="relative mb-6">
              <span className="absolute inset-0 animate-ping rounded-full bg-yellow-400/20" />
              <div className="relative grid h-20 w-20 place-items-center rounded-[24px] border-2 border-yellow-300/30 bg-yellow-400/10">
                {busy ? <Loader2 className="h-8 w-8 animate-spin text-yellow-300" /> : <ScanLine className="h-8 w-8 text-yellow-300" />}
              </div>
            </div>
            <h1 className="text-3xl font-black text-white">آنالیز سعیدکلاب</h1>
            <p className="mt-2 max-w-xs text-[12px] leading-5 text-white/35">
              {busy ? 'در حال آماده‌سازی مدل هوش مصنوعی…' : 'ابزار تخصصی آنالیز بیومکانیک و پاسچر بدن'}
            </p>
            <button onClick={handleStart} disabled={busy && !queuedStart}
              className="mt-8 flex h-14 w-full max-w-[260px] items-center justify-center gap-3 rounded-2xl bg-yellow-400 text-[16px] font-black text-black shadow-lg active:scale-95 transition-all disabled:opacity-50">
              {busy ? <Loader2 className="h-5 w-5 animate-spin" /> : <Zap className="h-5 w-5 fill-current" />}
              {busy ? 'صبر کنید…' : 'شروع آنالیز'}
            </button>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
