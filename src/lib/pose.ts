// ─────────────────────────────────────────────────────────────
//  PoseScan — core biomechanics library
// ─────────────────────────────────────────────────────────────

export interface Lm {
  x: number;
  y: number;
  z?: number;
  visibility?: number;
}

/** MediaPipe Pose landmark indices */
export const IDX = {
  nose: 0,
  eyeInL: 1,
  eyeL: 2,
  eyeOutL: 3,
  eyeInR: 4,
  eyeR: 5,
  eyeOutR: 6,
  earL: 7,
  earR: 8,
  mouthL: 9,
  mouthR: 10,
  shoulderL: 11,
  shoulderR: 12,
  elbowL: 13,
  elbowR: 14,
  wristL: 15,
  wristR: 16,
  hipL: 23,
  hipR: 24,
  kneeL: 25,
  kneeR: 26,
  ankleL: 27,
  ankleR: 28,
} as const;

export type JointTriplet = [number, number, number];

/**
 * Angle (degrees) at joint `b` formed by points a-b-c.
 * Computed in pixel space (x scaled by w, y by h) so angles are geometrically correct.
 */
export function angleAt(
  a: Lm,
  b: Lm,
  c: Lm,
  w: number,
  h: number,
): number {
  const abx = (a.x - b.x) * w;
  const aby = (a.y - b.y) * h;
  const cbx = (c.x - b.x) * w;
  const cby = (c.y - b.y) * h;
  const dot = abx * cbx + aby * cby;
  const mag = Math.hypot(abx, aby) * Math.hypot(cbx, cby);
  if (mag < 1e-6) return NaN;
  const cos = Math.min(1, Math.max(-1, dot / mag));
  return (Math.acos(cos) * 180) / Math.PI;
}

export const VISIBLE = 0.45;

export function isVis(lm: Lm | undefined, min = VISIBLE): boolean {
  if (!lm) return false;
  return (lm.visibility ?? 1) >= min;
}

/**
 * True 3D angle (degrees) at joint `b`, using x/y/z.
 * MediaPipe's z uses roughly the same scale as x (image width),
 * so we scale z by `w` and y by `h` for a metric-consistent space.
 * This is rotation-invariant — correct even when the body is not side-on.
 */
export function angle3D(
  a: Lm,
  b: Lm,
  c: Lm,
  w: number,
  h: number,
): number {
  const abx = (a.x - b.x) * w;
  const aby = (a.y - b.y) * h;
  const abz = ((a.z ?? 0) - (b.z ?? 0)) * w;
  const cbx = (c.x - b.x) * w;
  const cby = (c.y - b.y) * h;
  const cbz = ((c.z ?? 0) - (b.z ?? 0)) * w;
  const dot = abx * cbx + aby * cby + abz * cbz;
  const mag = Math.hypot(abx, aby, abz) * Math.hypot(cbx, cby, cbz);
  if (mag < 1e-6) return NaN;
  const cos = Math.min(1, Math.max(-1, dot / mag));
  return (Math.acos(cos) * 180) / Math.PI;
}

// ─────────────────────────────────────────────────────────────
//  Segment-to-segment geometry (relations between major joints)
// ─────────────────────────────────────────────────────────────

export const mid = (a: Lm, b: Lm): Lm => ({
  x: (a.x + b.x) / 2,
  y: (a.y + b.y) / 2,
  z: ((a.z ?? 0) + (b.z ?? 0)) / 2,
  visibility: Math.min(a.visibility ?? 1, b.visibility ?? 1),
});

/**
 * Tilt of the segment a→b away from the vertical axis, in degrees.
 * 0° = perfectly upright, 90° = horizontal. Always positive.
 */
export function tiltFromVertical(a: Lm, b: Lm, w: number, h: number): number {
  const dx = (b.x - a.x) * w;
  const dy = (b.y - a.y) * h;
  const len = Math.hypot(dx, dy);
  if (len < 1e-6) return NaN;
  // vertical reference = (0, -1) in image space
  const cos = Math.min(1, Math.max(-1, -dy / len));
  return (Math.acos(cos) * 180) / Math.PI;
}

/**
 * Signed tilt of the segment a→b away from the horizontal axis.
 * 0° = level. Positive = the `b` end sits lower on screen.
 */
export function tiltFromHorizontal(
  a: Lm,
  b: Lm,
  w: number,
  h: number,
): number {
  const dx = (b.x - a.x) * w;
  const dy = (b.y - a.y) * h;
  if (Math.hypot(dx, dy) < 1e-6) return NaN;
  return (Math.atan2(dy, Math.abs(dx) < 1e-6 ? 1e-6 : dx) * 180) / Math.PI;
}

/**
 * Horizontal-plane rotation between the shoulder line and the hip line
 * (torso twist). 0° = shoulders square over hips.
 */
export function torsoTwist(
  shoulderL: Lm,
  shoulderR: Lm,
  hipL: Lm,
  hipR: Lm,
): number {
  // project both segments onto the transverse (x, z) plane
  const sA = Math.atan2(
    (shoulderR.z ?? 0) - (shoulderL.z ?? 0),
    shoulderR.x - shoulderL.x,
  );
  const hA = Math.atan2((hipR.z ?? 0) - (hipL.z ?? 0), hipR.x - hipL.x);
  let d = ((sA - hA) * 180) / Math.PI;
  while (d > 180) d -= 360;
  while (d < -180) d += 360;
  if (d > 90) d -= 180;
  if (d < -90) d += 180;
  return Math.abs(d);
}

/** Face-landmark indices that actually carry good visibility scores. */
export const HEAD_MAP_IDXS = [
  IDX.nose,
  IDX.eyeL,
  IDX.eyeR,
  IDX.earL,
  IDX.earR,
  IDX.mouthL,
  IDX.mouthR,
] as const;

/**
 * Normalized-space bounding box of the head (for the scan halo).
 * Returns min/max x/y or null when the head isn't confidently visible.
 */
export function headBounds(
  lms: Lm[],
  minVis = 0.45,
): { x0: number; y0: number; x1: number; y1: number } | null {
  let x0 = Infinity;
  let y0 = Infinity;
  let x1 = -Infinity;
  let y1 = -Infinity;
  let n = 0;
  for (const i of HEAD_MAP_IDXS) {
    const lm = lms[i];
    if (!lm || (lm.visibility ?? 1) < minVis) continue;
    if (lm.x < x0) x0 = lm.x;
    if (lm.x > x1) x1 = lm.x;
    if (lm.y < y0) y0 = lm.y;
    if (lm.y > y1) y1 = lm.y;
    n++;
  }
  if (n < 4) return null;
  return { x0, y0, x1, y1 };
}

/** Monitored joint angles, keyed by slot. */
export type AngleKey =
  | 'elbowL'
  | 'elbowR'
  | 'kneeL'
  | 'kneeR'
  | 'hip'
  | 'shoulder'
  | 'neck'
  | 'neckR';

/** Face skeleton connections (indices 0-10) for head scan rendering. */
export const FACE_CONNS: [number, number][] = [
  [IDX.eyeL, IDX.nose], // 2-0
  [IDX.eyeInL, IDX.nose], // 1-0
  [IDX.eyeR, IDX.nose], // 5-0
  [IDX.eyeInR, IDX.nose], // 4-0
  [IDX.eyeInL, IDX.eyeL], // 1-2
  [IDX.eyeL, IDX.eyeOutL], // 2-3
  [IDX.eyeOutL, IDX.earL], // 3-7
  [IDX.eyeInR, IDX.eyeR], // 4-5
  [IDX.eyeR, IDX.eyeOutR], // 5-6
  [IDX.eyeOutR, IDX.earR], // 6-8
  [IDX.earL, IDX.mouthL], // 7-9
  [IDX.earR, IDX.mouthR], // 8-10
  [IDX.mouthL, IDX.mouthR], // 9-10
];

export const ANGLE_DEFS: Record<AngleKey, JointTriplet> = {
  neck: [IDX.earL, IDX.shoulderL, IDX.hipL],
  neckR: [IDX.earR, IDX.shoulderR, IDX.hipR],
  elbowL: [IDX.shoulderL, IDX.elbowL, IDX.wristL],
  elbowR: [IDX.shoulderR, IDX.elbowR, IDX.wristR],
  kneeL: [IDX.hipL, IDX.kneeL, IDX.ankleL],
  kneeR: [IDX.hipR, IDX.kneeR, IDX.ankleR],
  hip: [IDX.shoulderL, IDX.hipL, IDX.kneeL],
  shoulder: [IDX.elbowL, IDX.shoulderL, IDX.hipL],
};

// ─────────────────────────────────────────────────────────────
//  Exercises
// ─────────────────────────────────────────────────────────────

export type ExerciseKey = 'squat' | 'pushup' | 'lunge' | 'jack' | 'free';

export interface ExerciseDef {
  key: ExerciseKey;
  fa: string;
  en: string;
  hint: string;
  view: string;
  muscles: string;
  level: string;
  /** primary tracked joint (a, b, c) — angle measured at b, left side */
  joints: JointTriplet | null;
  /** right-side counterpart used for symmetry scoring */
  mirror: JointTriplet | null;
  /** effort phase entered when angle crosses this */
  low: number;
  /** release phase entered when angle crosses this */
  high: number;
  /** ideal extremum for a "perfect" rep */
  target: number;
  /** effort = angle shrinks (flex) or grows (raise) */
  dir: 'flex' | 'raise';
  kcalPerRep: number;
  image?: string;
}

export const EXERCISES: ExerciseDef[] = [
  {
    key: 'squat',
    fa: 'اسکوات',
    en: 'SQUAT',
    hint: 'پاها به عرض شانه؛ تا رسیدن زاویه زانو به زیر ۱۱۰ درجه پایین بروید و بدون توقف برگردید.',
    view: 'نمای کناری',
    muscles: 'چهارسر ران، باسن، همسترینگ',
    level: 'مبتدی تا پیشرفته',
    joints: [IDX.hipL, IDX.kneeL, IDX.ankleL],
    mirror: [IDX.hipR, IDX.kneeR, IDX.ankleR],
    low: 115,
    high: 158,
    target: 96,
    dir: 'flex',
    kcalPerRep: 0.42,
    image: '/images/ex-squat.jpg',
  },
  {
    key: 'pushup',
    fa: 'شنای سوئدی',
    en: 'PUSH-UP',
    hint: 'بدن را کاملاً صاف نگه دارید؛ آرنج را تا حدود ۹۰ درجه خم کنید و سپس فشار دهید.',
    view: 'نمای کناری',
    muscles: 'سینه، سه‌سر بازو، مرکزی',
    level: 'متوسط',
    joints: [IDX.shoulderL, IDX.elbowL, IDX.wristL],
    mirror: [IDX.shoulderR, IDX.elbowR, IDX.wristR],
    low: 98,
    high: 152,
    target: 82,
    dir: 'flex',
    kcalPerRep: 0.55,
    image: '/images/ex-pushup.jpg',
  },
  {
    key: 'lunge',
    fa: 'لانج',
    en: 'LUNGE',
    hint: 'یک قدم بلند جلو بردارید؛ زانوی جلو را تا نزدیک ۹۰ درجه خم کنید و به حالت اول برگردید.',
    view: 'نمای کناری',
    muscles: 'پاها، باسن، تعادل',
    level: 'متوسط',
    joints: [IDX.hipL, IDX.kneeL, IDX.ankleL],
    mirror: [IDX.hipR, IDX.kneeR, IDX.ankleR],
    low: 118,
    high: 156,
    target: 98,
    dir: 'flex',
    kcalPerRep: 0.5,
    image: '/images/ex-lunge.jpg',
  },
  {
    key: 'jack',
    fa: 'جک پرشینگ',
    en: 'JUMPING JACK',
    hint: 'هم‌زمان با باز کردن پاها، دست‌ها را از کنار بدن تا بالای سر ببرید و دوباره برگردید.',
    view: 'نمای روبه‌رو',
    muscles: 'تمام‌بدن، هوازی',
    level: 'مبتدی',
    joints: [IDX.hipL, IDX.shoulderL, IDX.wristL],
    mirror: [IDX.hipR, IDX.shoulderR, IDX.wristR],
    low: 62,
    high: 138,
    target: 158,
    dir: 'raise',
    kcalPerRep: 0.32,
    image: '/images/ex-jack.jpg',
  },
  {
    key: 'free',
    fa: 'اسکن آزاد',
    en: 'FREE SCAN',
    hint: 'بدون شمارش؛ فقط زاویه زنده همه مفاصل را زیر نظر بگیرید و فرم خود را بررسی کنید.',
    view: 'هر نمایی',
    muscles: 'تحلیل کلی بدن',
    level: 'همه سطوح',
    joints: null,
    mirror: null,
    low: 0,
    high: 0,
    target: 0,
    dir: 'flex',
    kcalPerRep: 0,
  },
];

export const exerciseByKey = (k: ExerciseKey): ExerciseDef =>
  EXERCISES.find((e) => e.key === k) ?? EXERCISES[0];

// ─────────────────────────────────────────────────────────────
//  Rep-counting state machine (hysteresis + debounce)
// ─────────────────────────────────────────────────────────────

export interface RepMachine {
  phase: 'up' | 'down';
  count: number;
  extremum: number;
  symSum: number;
  symN: number;
  lastRepAt: number;
  phaseAt: number;
}

export const freshRepMachine = (): RepMachine => ({
  phase: 'up',
  count: 0,
  extremum: Infinity,
  symSum: 0,
  symN: 0,
  lastRepAt: 0,
  phaseAt: 0,
});

export interface RepEvent {
  rep: boolean;
  quality: number | null;
  machine: RepMachine;
}

/** Feed a new angle sample; returns whether a rep completed and its form score. */
export function feedRep(
  m: RepMachine,
  angle: number,
  mirrorAngle: number | null,
  def: ExerciseDef,
  now: number,
): RepEvent {
  if (!Number.isFinite(angle)) return { rep: false, quality: null, machine: m };

  const effortZone = def.dir === 'flex' ? angle < def.low : angle > def.high;
  const released = def.dir === 'flex' ? angle > def.high : angle < def.low;

  if (m.phase === 'up' && effortZone) {
    m.phase = 'down';
    m.phaseAt = now;
    m.extremum = angle;
    m.symSum = 0;
    m.symN = 0;
  } else if (m.phase === 'down') {
    m.extremum =
      def.dir === 'flex'
        ? Math.min(m.extremum, angle)
        : Math.max(m.extremum, angle);
    if (mirrorAngle !== null && Number.isFinite(mirrorAngle)) {
      m.symSum += Math.abs(angle - mirrorAngle);
      m.symN += 1;
    }
    if (released && now - m.phaseAt > 260) {
      const quality = scoreRep(m.extremum, m.symN ? m.symSum / m.symN : null, def);
      m.phase = 'up';
      m.count += 1;
      m.lastRepAt = now;
      return { rep: true, quality, machine: m };
    }
  }
  return { rep: false, quality: null, machine: m };
}

/** 0–100 form score: 70% depth/extremum, 30% left-right symmetry. */
export function scoreRep(
  extremum: number,
  symDiff: number | null,
  def: ExerciseDef,
): number {
  let depth: number;
  if (def.dir === 'flex') {
    const span = Math.max(1, def.low - def.target);
    depth = Math.min(1, Math.max(0, (def.low - extremum) / span));
  } else {
    const span = Math.max(1, def.target - def.high);
    depth = Math.min(1, Math.max(0, (extremum - def.high) / span));
  }
  const sym = symDiff === null ? 0.85 : Math.min(1, Math.max(0, 1 - symDiff / 24));
  return Math.round(depth * 70 + sym * 30);
}

/** Effort progress 0..1 between rest and effort thresholds (for micro progress bars). */
export function effortProgress(angle: number, def: ExerciseDef): number {
  if (!Number.isFinite(angle)) return 0;
  if (def.dir === 'flex') {
    const span = def.high - def.target;
    return Math.min(1, Math.max(0, (def.high - angle) / span));
  }
  const span = def.target - def.low;
  return Math.min(1, Math.max(0, (angle - def.low) / span));
}
