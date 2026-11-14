// Forecast for make.fun's polymarket-faa-attribution-02: the Yes price of every
// market, each minute of the 4 hours after a checkpoint, as five quantiles.
//
// Model (per case, all markets pooled): prediction-market prices are near-
// martingales that sit still most minutes and then jump, so one normal spread
// fits neither the quiet minutes nor the jumps. Each market's median stays at
// its last price (drift 0 — the drift in the practice data is an artefact of
// that event's losers decaying, and a drift estimate only ever hurt). The
// spread has two scales: a tight inner scale for the 25%/75% quantiles and a
// wide outer scale for the 5%/95% ones, both growing with sqrt(h). Both scales
// are read off the case's own recent history, pooled across its markets —
// the markets of one family share one news flow — as quantiles of the
// level-scaled one-minute moves u = |dp| / sqrt(p (1-p)) over the last 24
// hours: inner = q70(u)/1.036, outer = q95(u)/1.960, each times the market's
// own sqrt(x (1-x)), with floors and caps so a quiet window never promises
// a narrower band than the jumps this family has already shown (Iran fell
// 0.49 -> 0.275 and AI-escape 0.215 -> 0.02 within a minute), and a loud one
// never widens without bound. Everything is a pure function of the case
// file, so two runs write the same bytes.
//
//   q05(h) = x - so * 1.644854 * sqrt(h)
//   q25(h) = x - si * 0.674490 * sqrt(h)
//   q50(h) = x
//   q75(h) = x + si * 0.674490 * sqrt(h)
//   q95(h) = x + so * 1.644854 * sqrt(h)
// each clamped to [0, 1] and rounded to 6 decimals, rows non-decreasing.
//
// Node 22, standard library only, no network. Reads every <id>.in in the
// input directory (MAKEFUN_INPUT_DIR, else /input) and writes <id>.out to
// the output directory (MAKEFUN_OUTPUT_DIR, else /output).
//
//   mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? "/input";
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? "/output";

/** Minutes of history the pooled volatility looks back over. */
const LOOKBACK_MINUTES = 1440;
/** Normal quantiles for the inner (25/75) and outer (5/95) levels. */
const Z_INNER = 0.67449;
const Z_OUTER = 1.644854;
/** Divisors turning a |move| quantile into a per-minute sigma (normal law). */
const Q70_TO_SIGMA = 1.036433;
const Q95_TO_SIGMA = 1.959964;
/** Inner scale: floor, cap (price units per square-root minute). */
const INNER_FLOOR = 0.0003;
const INNER_CAP = 0.002;
/** Outer scale: floor, cap. The floor is the jump insurance: this family's
    markets have already jumped 0.08-0.22 in a single minute. */
const OUTER_FLOOR = 0.0025;
const OUTER_CAP = 0.008;
/** Fallback scales when a case holds no usable price move at all. */
const FALLBACK_INNER = 0.0005;
const FALLBACK_OUTER = 0.0025;

interface CaseMarket {
  id: string;
  prices: number[];
}

interface CaseFile {
  horizon_minutes: number;
  markets: CaseMarket[];
}

/** x clamped to [0, 1], rounded to 6 decimals (0, never -0). */
function price6(x: number): number {
  return Math.round(Math.min(1, Math.max(0, x)) * 1e6) / 1e6 + 0;
}

/** The q-th quantile of a sorted array, linear interpolation (numpy default). */
function quantile(sorted: number[], q: number): number {
  const n = sorted.length;
  if (n === 0) return 0;
  if (n === 1) return sorted[0] as number;
  const pos = q * (n - 1);
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  const a = sorted[lo] as number;
  const b = sorted[hi] as number;
  return a + (b - a) * (pos - lo);
}

/**
 * Pooled level-scaled one-minute moves of the case: for every market, the
 * last LOOKBACK_MINUTES moves, each |dp| divided by sqrt(p (1-p)) at the
 * minute before the move, so markets at different price levels share one
 * volatility scale. Returned sorted, ascending.
 */
function pooledMoves(c: CaseFile): number[] {
  const moves: number[] = [];
  for (const m of c.markets) {
    const p = m.prices;
    const take = Math.min(p.length, LOOKBACK_MINUTES + 1);
    for (let i = p.length - take + 1; i < p.length; i++) {
      const prev = Math.min(Math.max(p[i - 1] as number, 1e-4), 1 - 1e-4);
      const d = Math.abs((p[i] as number) - (p[i - 1] as number));
      moves.push(d / Math.sqrt(prev * (1 - prev)));
    }
  }
  moves.sort((a, b) => a - b);
  return moves;
}

/** The 240 (or horizon) forecast rows for one market at last price x,
    with the case's two scales already multiplied by this market's level. */
function forecastRows(x: number, si: number, so: number, horizon: number): number[][] {
  const rows: number[][] = [];
  for (let h = 1; h <= horizon; h++) {
    const s = Math.sqrt(h);
    const raw = [
      x - so * Z_OUTER * s,
      x - si * Z_INNER * s,
      x,
      x + si * Z_INNER * s,
      x + so * Z_OUTER * s,
    ];
    // Round, clamp, and force non-decreasing (rounding alone cannot invert
    // these gaps, but the guarantee is worth one pass).
    const row: number[] = [];
    let prev = 0;
    for (const v of raw) {
      const q = Math.max(prev, price6(v));
      row.push(q);
      prev = q;
    }
    rows.push(row);
  }
  return rows;
}

/** The forecast for one case: one row per market, median at the last price. */
function forecastCase(c: CaseFile): { id: string; q: number[][] }[] {
  const moves = pooledMoves(c);
  const innerU = moves.length > 0 ? quantile(moves, 0.7) / Q70_TO_SIGMA : 0;
  const outerU = moves.length > 0 ? quantile(moves, 0.95) / Q95_TO_SIGMA : 0;
  const horizon = c.horizon_minutes > 0 ? c.horizon_minutes : 240;
  return c.markets.map((m) => {
    try {
      if (!Array.isArray(m.prices) || m.prices.length === 0) throw new Error("no prices");
      const x = m.prices[m.prices.length - 1] as number;
      if (!Number.isFinite(x)) throw new Error("bad last price");
      // This market's level turns the pooled u-scales into price sigmas;
      // with no usable move in the whole case, fall back to fixed scales.
      const level = Math.sqrt(Math.max(x * (1 - x), 1e-6));
      const si = moves.length > 0
        ? Math.min(INNER_CAP, Math.max(INNER_FLOOR, innerU * level))
        : FALLBACK_INNER;
      const so = moves.length > 0
        ? Math.min(OUTER_CAP, Math.max(OUTER_FLOOR, outerU * level))
        : FALLBACK_OUTER;
      return { id: m.id, q: forecastRows(x, si, so, horizon) };
    } catch {
      // Never lose the case over one market: flat 0.5 quantiles keep the
      // file valid (a bad row, not an invalid case).
      return { id: m.id, q: Array.from({ length: horizon }, () => [0.5, 0.5, 0.5, 0.5, 0.5]) };
    }
  });
}

mkdirSync(OUTPUT_DIR, { recursive: true });
const inputs = readdirSync(INPUT_DIR)
  .filter((f) => f.endsWith(".in"))
  .sort();
for (const file of inputs) {
  const id = file.slice(0, -".in".length);
  try {
    const c = JSON.parse(readFileSync(join(INPUT_DIR, file), "utf8")) as CaseFile;
    const forecasts = forecastCase(c);
    writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ forecasts })}\n`);
  } catch {
    // A case file that cannot be read at all cannot be forecast; write
    // nothing rather than a file that names markets the case does not hold.
    continue;
  }
}
