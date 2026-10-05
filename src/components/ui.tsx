import { motion } from 'framer-motion';

export function SectionHeading({
  kicker,
  title,
  desc,
  align = 'center',
}: {
  kicker: string;
  title: string;
  desc?: string;
  align?: 'center' | 'right';
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 28, filter: 'blur(6px)' }}
      whileInView={{ opacity: 1, y: 0, filter: 'blur(0px)' }}
      viewport={{ once: true, margin: '-80px' }}
      transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
      className={align === 'center' ? 'text-center' : 'text-right'}
    >
      <div
        className={`mb-4 flex items-center gap-3 ${align === 'center' ? 'justify-center' : ''}`}
      >
        <span className="h-px w-10 bg-gradient-to-l from-yellow-400/70 to-transparent" />
        <span className="font-mono text-[11px] font-medium tracking-[0.35em] text-yellow-300/90">
          {kicker}
        </span>
        <span className="h-px w-10 bg-gradient-to-r from-yellow-400/70 to-transparent" />
      </div>
      <h2 className="text-4xl font-black leading-[1.15] text-white sm:text-5xl">
        {title}
      </h2>
      {desc && (
        <p
          className={`mt-4 max-w-2xl text-[15px] leading-8 text-slate-400 ${align === 'center' ? 'mx-auto' : ''}`}
        >
          {desc}
        </p>
      )}
    </motion.div>
  );
}

export function RadialGauge({
  value,
  max = 100,
  label,
  unit,
  accent = '#facc15',
  size = 132,
}: {
  value: number | null;
  max?: number;
  label: string;
  unit?: string;
  accent?: string;
  size?: number;
}) {
  const r = 54;
  const c = 2 * Math.PI * r;
  const pct = value === null ? 0 : Math.min(1, Math.max(0, value / max));
  return (
    <div className="flex flex-col items-center gap-1">
      <div className="relative" style={{ width: size, height: size }}>
        <svg viewBox="0 0 132 132" className="h-full w-full -rotate-90">
          <circle
            cx="66"
            cy="66"
            r={r}
            fill="none"
            stroke="rgba(148,197,255,0.08)"
            strokeWidth="9"
          />
          <circle
            cx="66"
            cy="66"
            r={r}
            fill="none"
            stroke={accent}
            strokeWidth="9"
            strokeLinecap="round"
            strokeDasharray={c}
            strokeDashoffset={c * (1 - pct)}
            style={{
              transition: 'stroke-dashoffset 0.45s cubic-bezier(0.22,1,0.36,1)',
              filter: `drop-shadow(0 0 8px ${accent}66)`,
            }}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="font-mono text-[30px] font-semibold leading-none text-white tabular">
            {value === null ? '--' : Math.round(value)}
            {unit && (
              <span className="mr-0.5 text-[15px] text-slate-500">{unit}</span>
            )}
          </span>
        </div>
      </div>
      <span className="text-[11px] font-medium text-slate-400">{label}</span>
    </div>
  );
}

export function Sparkline({ data, className = 'h-11 w-full' }: { data: number[]; className?: string }) {
  const w = 240;
  const h = 44;
  const pts = data
    .map((v, i) => {
      const x = (i / Math.max(1, data.length - 1)) * (w - 4) + 2;
      const y = h - 4 - ((Math.min(180, Math.max(20, v)) - 20) / 160) * (h - 8);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(' ');
  const last = data.length ? pts.split(' ').pop()!.split(',') : null;
  return (
    <svg
      viewBox={`0 0 ${w} ${h}`}
      className={className}
      preserveAspectRatio="none"
    >
      <defs>
        <linearGradient id="sparkFill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#facc15" stopOpacity="0.35" />
          <stop offset="100%" stopColor="#facc15" stopOpacity="0" />
        </linearGradient>
      </defs>
      {data.length > 1 && (
        <>
          <polygon points={`2,${h} ${pts} ${w - 2},${h}`} fill="url(#sparkFill)" />
          <polyline
            points={pts}
            fill="none"
            stroke="#facc15"
            strokeWidth="1.6"
            strokeLinejoin="round"
            strokeLinecap="round"
          />
          {last && (
            <circle cx={last[0]} cy={last[1]} r="2.6" fill="#ffff2e">
              <animate attributeName="r" values="2.2;3.4;2.2" dur="1.4s" repeatCount="indefinite" />
            </circle>
          )}
        </>
      )}
    </svg>
  );
}

export function StatusDot({ tone }: { tone: 'ok' | 'warn' | 'idle' | 'err' }) {
  const map = {
    ok: 'bg-yellow-400',
    warn: 'bg-amber-400',
    idle: 'bg-slate-500',
    err: 'bg-rose-500',
  } as const;
  return (
    <span className="relative flex h-2 w-2">
      {tone === 'ok' && (
        <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-yellow-400 opacity-60" />
      )}
      <span className={`relative inline-flex h-2 w-2 rounded-full ${map[tone]}`} />
    </span>
  );
}
