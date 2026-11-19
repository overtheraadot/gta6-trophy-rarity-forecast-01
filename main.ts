// GTA VI trophy earn-rate forecaster — make.fun gta6-trophy-rarity-forecast-01
// Account A (overtheraa). Node 22, standard library only, fully deterministic.
//
// Model, per trophy: a weighted least-squares line of logit(earn rate) against
// log-time, extrapolated to the case target, with an honest spread.
//
//   x(read) = ln(1 + hours from 2026-11-19T00:00:00Z to the read)
//   y(read) = logit(rate), rate clamped to [0.01, 99.99] before the logit
//
// Reads are weighted so recent ones count more (exponential decay, 36 h
// half-life-ish scale). The fitted line is read at the target's x. Trophies
// with fewer than 4 usable reads have their slope shrunk toward the median
// slope of their tier (platinum/gold/silver/bronze), so a trophy with a
// flat or very short history does not chase noise. The median's move away
// from the last read is capped, so one noisy slope cannot fling a rare
// trophy to 90% or a common one to 0.01% over a few days. The trend is
// also damped when the target sits far outside the observed log-time
// range, so launch-night slopes are not trusted at full strength 4.5
// days out.
//
// Spread (logit sigma) combines three honest sources of error:
//   * a random-walk term, 0.55 logit units per sqrt(day) of horizon;
//   * an Exophase-refresh term, 0.30 per sqrt(day): the house saw the page
//     sit unchanged for hours and then jump at a refresh, so a forecast
//     over days must budget for refreshes it has not seen;
//   * the fit's own residual scale, inflated by how far the target sits
//     outside the observed x-range (standard regression prediction error).
// A floor of 0.15 logit units keeps even a 90-minute horizon from claiming
// false precision. Quantiles are the median shifted by the standard-normal
// z-scores at the scored levels.
//
// Reads $MAKEFUN_INPUT_DIR (else /input/) for *.in cases and writes one
// <id>.out per case to $MAKEFUN_OUTPUT_DIR (else /output/).

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? "/input";
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? "/output";

/** Launch-day anchor: log-time is measured from here. */
const ANCHOR_MS = Date.parse("2026-11-19T00:00:00Z");
const HOUR_MS = 3_600_000;

/** Scored quantile levels, in row order. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** Standard-normal quantiles at LEVELS. */
const Z_SCORES = [-1.6448536269514722, -0.6744897501960817, 0, 0.6744897501960817, 1.6448536269514722];

/** Random-walk spread, logit units per sqrt(day) of horizon. */
const WALK_SIGMA = 0.55;
/** Extra spread per sqrt(day) for Exophase refreshes we have not seen. */
const REFRESH_SIGMA = 0.3;
/** Never claim a tighter spread than this, in logit units. */
const SIGMA_FLOOR = 0.15;
/** Cap on |median logit - last-read logit|, so a slope cannot run away. */
const MAX_MEDIAN_MOVE = 4.5;
/** Recency weight decay: a read this many ms old weighs 1/e of a fresh one. */
const WEIGHT_DECAY_MS = 36 * HOUR_MS;
/** Trophies with fewer usable reads than this shrink toward their tier trend. */
const SHORT_HISTORY = 4;

type Trophy = { id: number; tier: string };
type Read = { at: string; rates: Array<number | null> };
type CaseFile = {
  id: string;
  target: string;
  horizon_minutes: number;
  levels: number[];
  trophies: Trophy[];
  history: Read[];
};

const clampRate = (pct: number): number =>
  Number.isFinite(pct) ? Math.min(99.99, Math.max(0.01, pct)) : 1;
const toLogit = (pct: number): number => {
  const p = clampRate(pct) / 100;
  return Math.log(p / (1 - p));
};
const fromLogit = (l: number): number => 100 / (1 + Math.exp(-l));
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;
/** Hours from the anchor to an ISO timestamp, floored at 0. */
const hoursSinceAnchor = (iso: string): number =>
  Math.max(0, (Date.parse(iso) - ANCHOR_MS) / HOUR_MS);

const medianOf = (xs: number[]): number => {
  if (xs.length === 0) return 0;
  const s = [...xs].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 === 1 ? s[m] : (s[m - 1] + s[m]) / 2;
};

/**
 * Weighted least-squares line y ~ a + b*x. Returns slope, intercept,
 * weighted mean of x, weighted Sxx, residual RMS and effective count.
 */
function fitLine(points: Array<{ x: number; y: number; w: number }>): {
  slope: number;
  intercept: number;
  xBar: number;
  sxx: number;
  residSigma: number;
  nEff: number;
} {
  const wSum = points.reduce((s, p) => s + p.w, 0);
  const xBar = points.reduce((s, p) => s + p.w * p.x, 0) / wSum;
  const yBar = points.reduce((s, p) => s + p.w * p.y, 0) / wSum;
  let sxx = 0;
  let sxy = 0;
  for (const p of points) {
    sxx += p.w * (p.x - xBar) * (p.x - xBar);
    sxy += p.w * (p.x - xBar) * (p.y - yBar);
  }
  const slope = sxx > 1e-12 ? sxy / sxx : 0;
  const intercept = yBar - slope * xBar;
  let residSq = 0;
  for (const p of points) {
    const r = p.y - (intercept + slope * p.x);
    residSq += p.w * r * r;
  }
  const n = points.length;
  // Residual scale, with a small-sample correction; 0.05 floor keeps a
  // perfectly flat history from claiming zero uncertainty.
  const residSigma =
    n > 2
      ? Math.max(0.05, Math.sqrt(residSq / wSum) * Math.sqrt(n / (n - 2)))
      : 0.15;
  const wSq = points.reduce((s, p) => s + p.w * p.w, 0);
  const nEff = wSq > 0 ? (wSum * wSum) / wSq : n;
  return { slope, intercept, xBar, sxx, residSigma, nEff };
}

function forecastCase(c: CaseFile): Array<{ id: number; q: number[] }> {
  if (c.levels.length !== LEVELS.length || c.levels.some((t, i) => t !== LEVELS[i])) {
    throw new Error(`case ${c.id} asks for levels ${JSON.stringify(c.levels)}`);
  }
  const horizonDays = Math.max(0, c.horizon_minutes / 1440);
  const targetX = Math.log1p(hoursSinceAnchor(c.target));
  const lastRead = c.history[c.history.length - 1];
  if (!lastRead) throw new Error(`case ${c.id} has no history`);

  const lastAt = Date.parse(lastRead.at);

  // First pass: fit every trophy, to learn each tier's median slope.
  const fits = c.trophies.map((trophy, i) => {
    const points: Array<{ x: number; y: number; w: number }> = [];
    let lastRate: number | null = null;
    for (const read of c.history) {
      const rate = read.rates[i];
      if (rate === null || rate === undefined || !Number.isFinite(rate)) continue;
      lastRate = rate;
      points.push({
        x: Math.log1p(hoursSinceAnchor(read.at)),
        y: toLogit(rate),
        w: Math.exp(-Math.max(0, lastAt - Date.parse(read.at)) / WEIGHT_DECAY_MS),
      });
    }
    return {
      trophy,
      points,
      fit: points.length >= 2 ? fitLine(points) : null,
      lastRate,
    };
  });

  const tierSlope = new Map<string, number>();
  for (const tier of new Set(c.trophies.map((t) => t.tier))) {
    const slopes = fits
      .filter((f) => f.trophy.tier === tier && f.fit !== null && f.points.length >= SHORT_HISTORY)
      .map((f) => (f.fit as NonNullable<typeof f.fit>).slope);
    tierSlope.set(tier, medianOf(slopes));
  }
  // Fallback tier trend: median slope across all trophies with a real fit.
  const globalSlope = medianOf(
    fits.filter((f) => f.fit !== null).map((f) => (f.fit as NonNullable<typeof f.fit>).slope),
  );

  return fits.map((f) => {
    // Reference rate: last non-null read, else the tier's typical last rate, else 1%.
    let refRate = f.lastRate;
    if (refRate === null || refRate === undefined) {
      const peers = fits
        .filter((g) => g.trophy.tier === f.trophy.tier && g.lastRate !== null)
        .map((g) => g.lastRate as number);
      refRate = peers.length > 0 ? medianOf(peers) : 1;
    }
    const refLogit = toLogit(refRate);

    let medianLogit = refLogit;
    let sigma = Math.sqrt((WALK_SIGMA * WALK_SIGMA + REFRESH_SIGMA * REFRESH_SIGMA) * horizonDays);
    if (f.fit !== null && f.points.length >= 2) {
      const fit = f.fit;
      let slope = fit.slope;
      if (f.points.length < SHORT_HISTORY) {
        // Short history: shrink the slope toward the tier's median trend.
        const tier = tierSlope.get(f.trophy.tier) ?? globalSlope;
        const k = f.points.length / SHORT_HISTORY;
        slope = k * slope + (1 - k) * tier;
      }
      const lineAtTarget = fit.intercept + slope * targetX;
      // Damp the trend when the target sits far outside the observed
      // log-time range: a line fitted on launch night should not be
      // trusted at full strength 4.5 days out. The damping is 1 when the
      // target is inside the observed range and fades toward 0 as the
      // gap beyond the last observed x grows.
      const xMax = Math.max(...f.points.map((p) => p.x));
      const damp = 1 / (1 + 0.25 * Math.max(0, targetX - xMax));
      medianLogit = refLogit + (lineAtTarget - refLogit) * damp;
      // Prediction error grows as the target leaves the observed x-range.
      const extrap =
        fit.sxx > 1e-12
          ? Math.sqrt(1 + 1 / Math.max(1, fit.nEff) + ((targetX - fit.xBar) ** 2) / fit.sxx)
          : 2;
      const fitSigma = fit.residSigma * Math.min(extrap, 3);
      sigma = Math.sqrt(sigma * sigma + fitSigma * fitSigma);
    }
    // Keep the median within a sane move of the last read.
    medianLogit = Math.min(refLogit + MAX_MEDIAN_MOVE, Math.max(refLogit - MAX_MEDIAN_MOVE, medianLogit));
    sigma = Math.max(SIGMA_FLOOR, sigma);

    const q: number[] = [];
    let prev = 0.01;
    for (const z of Z_SCORES) {
      const v = clampRate(round4(fromLogit(medianLogit + z * sigma)));
      // Rounding must never break the non-decreasing order.
      const safe = Math.max(prev, v);
      q.push(safe);
      prev = safe;
    }
    return { id: f.trophy.id, q };
  });
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write identical bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith(".in")).sort();
for (const file of inputs) {
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), "utf8")) as CaseFile;
  const trophies = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${c.id}.out`), `${JSON.stringify({ trophies })}\n`);
}
