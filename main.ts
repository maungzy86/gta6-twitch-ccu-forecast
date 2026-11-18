// gta6-twitch-ccu-forecast-01 entry: hour-by-hour quantile forecast of GTA VI's launch-day Twitch viewers.
//
// Method: anchor on hour 1's print p1 and borrow the *shape* of a launch day from the case's own comparables
// (past big launches shipped inside the case file, so the final case's larger comparable set is used too —
// nothing is hardcoded). For each hour h = 2..24, take the median log-ratio m_h = median ln(viewers_h /
// viewers_1) across comparables and a per-hour spread s_h = IQR(log-ratio) / 1.349, then
//
//   q_tau(h) = p1 * exp(m_h + z_tau * sigma_h),  sigma_h = clamp(SPREAD_MULT * s_h, 0.20, 1.20).
//
// Why this shape: hour 1 (00:00-01:00 UTC) only captures the earliest unlocks (New Zealand, central Europe,
// the UK at 00:00Z). The US unlocks at 05:00Z (East) and 08:00Z (West) and evening prime time land later, so
// in the practice comparables the median ratio first dips (about 0.73x near hour 4, as the first-hour crowd
// thins before the US wakes up) and then climbs to a peak of roughly 2.1-2.4x around hours 19-20. The spread
// must widen through the day because launch sizes vary enormously between games (evening IQRs exceed 1.4 nats).
//
// SPREAD_MULT was tuned by leave-one-out validation on the practice comparables (each comparable in turn
// treated as a pseudo-case, forecast from the others only, scored with the contest's log-scale pinball loss):
//   multiplier 0.70 -> 0.3120 | 0.85 -> 0.3062 | 1.00 -> 0.3026 | 1.15 -> 0.3007 | 1.30 -> 0.2997
//   house persistence baseline on the same pseudo-cases -> 0.3465
// 1.30 was the grid's minimum (a probe at 1.45 scored 0.3002, so the optimum is flat there); it is fixed here
// as a constant so the program stays deterministic and offline. The per-hour spread s_h is smoothed with the
// same centred 3-hour moving average as the median curve: a single hour's IQR over ~34 curves is noisy while
// neighbouring hours' dispersion is genuinely alike, and smoothing it improved the LOO loss at every grid
// multiplier (at 1.30: 0.3001 unsmoothed -> 0.2997 smoothed).
//
// Fallbacks: an hour with fewer than 5 usable comparables borrows the nearest fitted hour; if no hour can be
// fitted at all (too few comparables), the forecast degrades to the house persistence baseline
// (spread 0.15 * sqrt(h - 1), median ratio 1).
//
// Deterministic: no randomness, no clock reads; input files are processed in sorted name order.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in row order. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** Comparables whose hour-1 audience is smaller than this are too noisy to shape the curve. */
const MIN_HOUR1 = 500;
/** An hour needs at least this many usable comparables before its own median/dispersion are trusted. */
const MIN_COUNT = 5;
/** Spread bounds in natural-log units: never pretend certainty, never blow past the observed dispersion. */
const SIGMA_FLOOR = 0.2;
const SIGMA_CAP = 1.2;
/** Global spread multiplier, tuned by leave-one-out on the practice comparables (see header). */
const SPREAD_MULT = 1.3;
/** House baseline spread per square-root hour, used only by the persistence fallback. */
const BASELINE_SIGMA = 0.15;

interface HourlyPoint {
  from: string;
  viewers: number;
}

interface Comparable {
  day_start: string;
  hourly?: HourlyPoint[];
}

interface CaseFile {
  known: Array<{ hour: number; viewers: number }>;
  forecast: Array<{ hour: number }>;
  comparables?: Comparable[];
}

interface HourFit {
  m: number;
  sigma: number;
}

/** x rounded to 2 decimals (half up). */
const round2 = (x: number): number => Math.round(x * 100) / 100;

/** Linear-interpolation quantile of an already-sorted array (deterministic; matches the tuning script). */
function quantileSorted(s: number[], t: number): number {
  const n = s.length;
  if (n === 1) return s[0];
  const pos = t * (n - 1);
  const lo = Math.floor(pos);
  const frac = pos - lo;
  return frac === 0 ? s[lo] : s[lo] * (1 - frac) + s[lo + 1] * frac;
}

/**
 * Mean viewers per release-day hour (1..24) of one comparable. Its hourly series starts 14 days before
 * release, so an hour is located by offset from day_start; several points can land in one hour (their mean
 * is the hour's figure, as in the case construction) and non-positive points carry no signal.
 */
function releaseDayHours(comp: Comparable): Map<number, number> {
  const t0 = Date.parse(comp.day_start);
  const acc = new Map<number, number[]>();
  for (const pt of comp.hourly ?? []) {
    const off = (Date.parse(pt.from) - t0) / 3600000;
    if (!Number.isInteger(off) || off < 0 || off > 23) continue;
    if (typeof pt.viewers !== 'number' || !(pt.viewers > 0)) continue;
    const h = off + 1;
    const arr = acc.get(h);
    if (arr) arr.push(pt.viewers);
    else acc.set(h, [pt.viewers]);
  }
  const out = new Map<number, number>();
  for (const [h, vs] of acc) out.set(h, vs.reduce((a, b) => a + b, 0) / vs.length);
  return out;
}

/**
 * Fit the per-hour median log-ratio curve and spread from the case's comparables.
 * Returns null when no hour reaches MIN_COUNT usable comparables (caller falls back to persistence).
 */
function fitCurve(comparables: Comparable[]): Map<number, HourFit> | null {
  const curves: Array<Map<number, number>> = [];
  for (const comp of comparables) {
    const hours = releaseDayHours(comp);
    if ((hours.get(1) ?? 0) >= MIN_HOUR1) curves.push(hours);
  }
  const raw = new Map<number, { m: number; s: number }>();
  for (let h = 2; h <= 24; h++) {
    const lrs: number[] = [];
    for (const hours of curves) {
      const v1 = hours.get(1);
      const vh = hours.get(h);
      if (v1 === undefined || vh === undefined) continue;
      lrs.push(Math.log(vh / v1));
    }
    if (lrs.length >= MIN_COUNT) {
      const s = [...lrs].sort((a, b) => a - b);
      raw.set(h, { m: quantileSorted(s, 0.5), s: (quantileSorted(s, 0.75) - quantileSorted(s, 0.25)) / 1.349 });
    }
  }
  if (raw.size === 0) return null;
  // Hours with too little data borrow the nearest fitted hour (the lower neighbour wins ties).
  const filled = new Map<number, { m: number; s: number }>();
  for (let h = 2; h <= 24; h++) {
    const own = raw.get(h);
    if (own) {
      filled.set(h, own);
      continue;
    }
    for (let d = 1; d <= 22; d++) {
      const lo = raw.get(h - d);
      if (lo) {
        filled.set(h, lo);
        break;
      }
      const hi = raw.get(h + d);
      if (hi) {
        filled.set(h, hi);
        break;
      }
    }
  }
  // Smooth both the median curve and the spread with a centred 3-hour moving average in log space:
  // single-hour medians and IQRs are noisy, but the true intraday shape and dispersion change slowly;
  // edges average the neighbours that exist.
  const out = new Map<number, HourFit>();
  for (let h = 2; h <= 24; h++) {
    const mVals: number[] = [];
    const sVals: number[] = [];
    for (const hh of [h - 1, h, h + 1]) {
      if (hh < 2 || hh > 24) continue;
      const f = filled.get(hh);
      if (f) {
        mVals.push(f.m);
        sVals.push(f.s);
      }
    }
    const m = mVals.reduce((a, b) => a + b, 0) / mVals.length;
    const s = sVals.reduce((a, b) => a + b, 0) / sVals.length;
    const sigma = Math.min(SIGMA_CAP, Math.max(SIGMA_FLOOR, s * SPREAD_MULT));
    out.set(h, { m, sigma });
  }
  return out;
}

/** The five quantiles of one hour, clamped to (0, 1e9] and forced non-decreasing after rounding. */
function hourQuantiles(p1: number, h: number, fit: Map<number, HourFit> | null): number[] {
  let qs: number[];
  const f = fit?.get(h);
  if (!f) {
    // Persistence fallback: hour 1 repeats, with the house baseline's random-walk spread.
    qs = Z.map((z) => round2(p1 * Math.exp(z * BASELINE_SIGMA * Math.sqrt(h - 1))));
  } else {
    qs = Z.map((z) => round2(p1 * Math.exp(f.m + z * f.sigma)));
  }
  qs = qs.map((q) => Math.min(1e9, Math.max(0.01, q)));
  for (let j = 1; j < qs.length; j++) if (qs[j] < qs[j - 1]) qs[j] = qs[j - 1];
  return qs;
}

/** Internal check that a finished forecast obeys the output contract; used to catch fit bugs before writing. */
function isValid(hours: Array<{ hour: number; q: number[] }>): boolean {
  if (hours.length !== 23) return false;
  for (let i = 0; i < 23; i++) {
    const row = hours[i];
    if (row.hour !== i + 2 || row.q.length !== LEVELS.length) return false;
    for (let j = 0; j < row.q.length; j++) {
      const q = row.q[j];
      if (!Number.isFinite(q) || !(q > 0) || q > 1e9) return false;
      if (j > 0 && q < row.q[j - 1]) return false;
    }
  }
  return true;
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as CaseFile;
  const first = c.known.find((k) => k.hour === 1);
  if (first === undefined) throw new Error(`${file} has no hour-1 print`);
  const p1 = Math.max(1, first.viewers);
  const fit = fitCurve(c.comparables ?? []);
  const wanted = [...new Set(c.forecast.map(({ hour }) => hour))].sort((a, b) => a - b);
  const hours = wanted.map((hour) => ({ hour, q: hourQuantiles(p1, hour, fit) }));
  if (!isValid(hours)) {
    // Never write an invalid forecast: degrade the whole case to persistence and check again.
    const fallback = wanted.map((hour) => ({ hour, q: hourQuantiles(p1, hour, null) }));
    if (!isValid(fallback)) throw new Error(`${file}: could not build a valid forecast`);
    writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ hours: fallback })}\n`);
  } else {
    writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ hours })}\n`);
  }
}
