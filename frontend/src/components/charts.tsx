import { Area, CartesianGrid, ComposedChart, Line, ReferenceLine, ResponsiveContainer, Scatter, Tooltip, XAxis, YAxis } from "recharts";
import type { DriverContribution, ForecastSeries, Observation, Variable } from "../api/types";
import { addDays, featureLabel, fmtDay, fmtSig, fmtSigned, parseDate } from "../lib/format";
import { BRAND, HAIRLINE, INK, INK_MUTED, PAPER } from "../lib/ramp";

// Chart conventions (design.md + dataviz): plain charts on paper, recessive hairline axes,
// 2px lines, no gridline decoration, one y-axis, tooltip on hover.
const AXIS = { stroke: HAIRLINE, tick: { fill: INK_MUTED, fontSize: 11, fontFamily: "IBM Plex Mono" }, tickLine: false };

/** Signed attribution colours: severity orange raises the forecast, brand blue lowers it.
 * Orange/blue survives the common colour-vision deficiencies; every bar also carries its
 * signed value as text and sits on its own side of zero. */
export const RAISE = "var(--exceed-3)";
export const LOWER = "var(--brand-500)";

function TooltipBox({ lines }: { lines: [string, string][] }) {
  return (
    <div className="t-dense rounded-md bg-surface px-2.5 py-1.5 shadow-[var(--shadow-float)]">
      {lines.map(([k, v]) => (
        <div key={k} className="flex justify-between gap-4">
          <span className="text-muted">{k}</span>
          <span className="t-value-sm">{v}</span>
        </div>
      ))}
    </div>
  );
}

interface FanRow {
  t: number;
  date: string;
  obs?: number | null;
  band?: [number, number] | null;
  p50?: number | null;
  thr?: number | null;
}

/** Observed history + 10-day forecast fan (P10–P90, P50) for one variable. Observations
 * are points: a missing date is a gap, never a line drawn across it. */
export function FanChart({
  variable,
  history,
  series,
  days = 90,
}: {
  variable: Variable;
  history: Observation[];
  series: ForecastSeries | undefined;
  days?: number;
}) {
  const issued = series?.days[0] ? addDays(series.days[0].target_date, -series.days[0].horizon) : history.at(-1)?.obs_date;
  if (!issued) return null;
  const start = addDays(issued, -days);
  const rows: FanRow[] = [];
  for (const o of history) {
    if (o.obs_date < start || o.quality_flag !== "OK") continue;
    const v = o[variable];
    if (v === null) continue;
    rows.push({ t: parseDate(o.obs_date).getTime(), date: o.obs_date, obs: v });
  }
  const thresholds = new Set<number>();
  for (const d of series?.days ?? []) {
    rows.push({
      t: parseDate(d.target_date).getTime(),
      date: d.target_date,
      band: d.p10 !== null && d.p90 !== null ? [d.p10, d.p90] : null,
      p50: d.p50,
      thr: d.threshold,
    });
    if (d.threshold !== null) thresholds.add(d.threshold);
  }
  rows.sort((a, b) => a.t - b.t);
  const thr = thresholds.size === 1 ? [...thresholds][0] : null;
  const nowT = parseDate(issued).getTime();
  const ticks = [start, addDays(issued, -60), addDays(issued, -30), issued, addDays(issued, 10)].map((d) => parseDate(d).getTime());

  return (
    <div className="h-44 w-full" role="img" aria-label={`Observed ${variable} for the last ${days} days and the 10-day forecast fan`}>
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={rows} margin={{ top: 8, right: 20, bottom: 0, left: 0 }}>
          <CartesianGrid vertical={false} stroke={HAIRLINE} strokeOpacity={0.5} />
          <XAxis
            dataKey="t"
            type="number"
            scale="time"
            domain={[parseDate(start).getTime(), parseDate(addDays(issued, 10)).getTime()]}
            ticks={ticks}
            tickFormatter={(t: number) => fmtDay(new Date(t).toISOString())}
            {...AXIS}
          />
          <YAxis width={44} tickFormatter={(v: number) => fmtSig(v, 2)} {...AXIS} axisLine={false} />
          <ReferenceLine x={nowT} stroke={INK} strokeWidth={1} label={{ value: "now", position: "insideTopRight", fill: INK_MUTED, fontSize: 10 }} />
          <Area dataKey="band" stroke="none" fill={BRAND} fillOpacity={0.18} isAnimationActive={false} connectNulls={false} />
          <Line dataKey="p50" stroke={BRAND} strokeWidth={2} dot={false} isAnimationActive={false} connectNulls={false} />
          <Line dataKey="thr" stroke={INK_MUTED} strokeWidth={1} strokeDasharray="4 3" dot={false} isAnimationActive={false} />
          <Scatter dataKey="obs" fill={INK} stroke={PAPER} strokeWidth={1} shape="circle" isAnimationActive={false} />
          <Tooltip
            cursor={{ stroke: HAIRLINE }}
            content={({ active, payload }) => {
              const r = active ? (payload?.[0]?.payload as FanRow | undefined) : undefined;
              if (!r) return null;
              const lines: [string, string][] = [["date", fmtDay(r.date, true)]];
              if (r.obs != null) lines.push(["observed", fmtSig(r.obs)]);
              if (r.p50 != null) lines.push(["median forecast", fmtSig(r.p50)]);
              if (r.band) lines.push(["P10–P90", `${fmtSig(r.band[0])}–${fmtSig(r.band[1])}`]);
              if (r.thr != null) lines.push(["threshold", fmtSig(r.thr)]);
              return <TooltipBox lines={lines} />;
            }}
          />
        </ComposedChart>
      </ResponsiveContainer>
      {thr === null && series?.days.some((d) => d.threshold !== null) && (
        <p className="t-dense text-muted">Threshold changes with the season inside the window.</p>
      )}
    </div>
  );
}

/** Horizontal signed bars, built in plain HTML: a text label, a bar either side of zero,
 * and the signed value in mono. Largest magnitude first (the API's order). */
export function AttributionBars({ items, units }: { items: DriverContribution[]; units?: string }) {
  if (!items.length) return <p className="t-ui text-muted">No driver moved this forecast from the model's baseline.</p>;
  const max = Math.max(...items.map((d) => Math.abs(d.contribution)));
  return (
    <div role="table" aria-label={`Driver contributions${units ? ` in ${units}` : ""}`}>
      {items.map((d) => {
        const w = max > 0 ? (Math.abs(d.contribution) / max) * 50 : 0;
        const pos = d.contribution > 0;
        return (
          <div role="row" key={d.feature} className="grid grid-cols-[9rem_1fr_4.5rem] items-center gap-2 py-0.5" title={d.value !== null ? `value ${fmtSig(d.value)}` : "value not recorded"}>
            <span role="cell" className="t-dense truncate">
              {featureLabel(d.feature)}
            </span>
            <span role="cell" className="relative h-2.5" aria-hidden>
              <span className="absolute inset-y-0 left-1/2 w-px bg-hairline" />
              <span
                className="absolute inset-y-0 rounded-[2px]"
                style={{
                  background: pos ? RAISE : LOWER,
                  left: pos ? "50%" : `${50 - w}%`,
                  width: `${w}%`,
                }}
              />
            </span>
            <span role="cell" className="t-value-sm text-right">
              {fmtSigned(d.contribution)}
            </span>
          </div>
        );
      })}
    </div>
  );
}

/** A small sparkline-free count strip: n per horizon etc. (used by the validation page). */
export function TooltipLines(lines: [string, string][]) {
  return <TooltipBox lines={lines} />;
}
