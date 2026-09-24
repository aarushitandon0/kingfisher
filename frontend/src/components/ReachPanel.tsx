import { useState } from "react";
import { api } from "../api/client";
import type { AttributionSeries, ReachDetail, Variable } from "../api/types";
import { daysBetween, exposureLabel, fmtDay, fmtDistance, fmtNum, fmtProb, fmtRange, fmtSig, humanize, VARIABLE_LABEL } from "../lib/format";
import { useApi } from "../lib/useApi";
import { useStore } from "../store";
import { AttributionBars, FanChart } from "./charts";
import { ErrorNote, Loading, ObservabilityBadge, Rule, SeverityLabel } from "./bits";

const VARS: Variable[] = ["turbidity_proxy", "ndci"];

export function ReachPanel({ id }: { id: string }) {
  const select = useStore((s) => s.select);
  const [variable, setVariable] = useState<Variable>("turbidity_proxy");
  const detail = useApi(`reach:${id}`, () => api.reach(id, 400));
  const forecast = useApi(`forecast:${id}`, () => api.forecast(id));
  const attribution = useApi(`attr:${id}`, () => api.attribution(id));

  const d = detail.data;
  const p = d?.properties;
  const series = forecast.data?.series.find((s) => s.variable === variable);
  const window = forecast.data?.series[0]?.days;

  return (
    <article className="flex flex-col" aria-label={`Reach ${id}`}>
      <div className="flex items-start justify-between gap-2">
        <div>
          <h2 className="t-display">{p?.name ?? (p ? "Unnamed channel" : id)}</h2>
          <p className="t-ui text-muted">
            {id}
            {p?.strahler_order != null && <> — stream order {p.strahler_order}</>}
            {p && <> — {fmtNum(p.length_m, 0)} m long</>}
          </p>
        </div>
        <button type="button" onClick={() => select(null)} className="t-ui text-muted hover:text-ink" aria-label="Close reach detail">
          Close
        </button>
      </div>

      {detail.loading && !d && <Loading what="reach" />}
      <ErrorNote error={detail.error} what="Reach detail" />

      {d && p && (
        <>
          <Reading d={d} window={window ? [window[0].target_date, window.at(-1)!.target_date] : null} />
          <div className="mt-3">
            <ObservabilityBadge observability={p.observability} medianPixels={p.median_water_pixels} />
          </div>

          <div className="mt-4 flex gap-1" role="radiogroup" aria-label="Variable">
            {VARS.map((v) => (
              <button
                key={v}
                type="button"
                role="radio"
                aria-checked={variable === v}
                onClick={() => setVariable(v)}
                className={`t-dense border px-2 py-0.5 ${variable === v ? "border-ink text-ink" : "border-hairline text-muted hover:text-ink"}`}
              >
                {VARIABLE_LABEL[v]}
              </button>
            ))}
          </div>

          <Rule>observed and 10-day forecast</Rule>
          <ErrorNote error={forecast.error} what="Forecast" />
          <FanChart variable={variable} history={p.history} series={series} />
          <p className="t-dense text-muted">
            Points: clear-sky readings. Band: P10–P90. Line: median. Dashed: seasonal threshold.
            {forecast.data && <> Weather: {forecast.data.weather === "LIVE" ? "ECMWF IFS run of the issue date" : forecast.data.weather.toLowerCase()}.</>}
          </p>
          {forecast.data?.notes.map((n) => (
            <p key={n} className="t-dense mt-1 text-muted">
              {n}
            </p>
          ))}

          <Rule>driver attribution</Rule>
          {attribution.loading && <Loading what="attribution" />}
          <ErrorNote error={attribution.error} what="Attribution" />
          {attribution.data && <Attribution s={attribution.data.series.find((s) => s.variable === variable)} basis={attribution.data.basis} />}

          <Rule>upstream catchment</Rule>
          <CatchmentCard d={d} />

          <Rule>exposure within {p.exposure?.buffer_m ?? "—"} m</Rule>
          <ExposureList d={d} />
        </>
      )}
    </article>
  );
}

function Reading({ d, window }: { d: ReachDetail; window: [string, string] | null }) {
  const p = d.properties;
  if (p.exceedance_status === "OK" && p.p_exceed_max !== null) {
    const which = (p.p_exceed_turbidity_proxy ?? -1) >= (p.p_exceed_ndci ?? -1) ? "turbidity_proxy" : "ndci";
    return (
      <div className="mt-4">
        <p className="t-reading">{fmtProb(p.p_exceed_max)}</p>
        <p className="t-ui">
          peak exceedance probability, {VARIABLE_LABEL[which]}
          {window && (
            <>
              , <span className="t-value-sm">{fmtRange(window[0], window[1])}</span>
            </>
          )}
        </p>
        {p.alert_severity && (
          <p className="t-ui mt-1">
            <SeverityLabel s={p.alert_severity} /> in the latest alert run
          </p>
        )}
      </div>
    );
  }
  // Insufficient evidence is a finding, not a failure: say what is missing and why.
  const last = p.history_end;
  const lastOk = [...p.history].reverse().find((o) => o.quality_flag === "OK")?.obs_date ?? null;
  const age = lastOk && last ? daysBetween(lastOk, last) : null;
  return (
    <div className="mt-4 hatch-border pl-3">
      <p className="t-reading text-muted">—</p>
      <p className="t-ui">No exceedance probability.</p>
      <p className="t-ui mt-1 text-muted">
        {p.observability === "DRIVER_ONLY"
          ? `Not enough observation to set a threshold. This reach is optically unobservable at 10 m (median ${p.median_water_pixels ?? 0} clean water pixels per pass), so it has no history of its own to exceed.`
          : p.exceedance_status === "NO_FORECAST"
            ? "No production forecast on record for this reach."
            : `Not enough past observations in this season to set a threshold${lastOk ? `; last usable reading ${fmtDay(lastOk, true)}${age !== null ? `, ${age} days before the latest pass` : ""}` : ""}.`}
      </p>
    </div>
  );
}

function Attribution({ s, basis }: { s: AttributionSeries | undefined; basis: string }) {
  if (!s) return <p className="t-ui text-muted">No attribution stored for this variable.</p>;
  return (
    <div>
      <p className="t-dense mb-2 text-muted">
        {s.selection === "PEAK_EXCEEDANCE" ? "At the day of peak exceedance probability" : "At the day of the highest P90 (no threshold on this reach)"},{" "}
        <span className="t-value-sm">{fmtDay(s.target_date)}</span> (day {s.horizon}). Contribution to the {s.quantile.toUpperCase()} forecast, index units.
      </p>
      <AttributionBars items={s.contributions} units="index units" />
      <p className="t-dense mt-2 text-muted">
        <span className="inline-block h-2 w-3 align-middle" style={{ background: "#9A6636" }} /> raises the forecast{" "}
        <span className="ml-2 inline-block h-2 w-3 align-middle" style={{ background: "#6F97A3" }} /> lowers it. {basis}.
      </p>
    </div>
  );
}

const ATTRS: [string, string, (v: number) => string][] = [
  ["catchment_area_km2", "area", (v) => `${fmtSig(v, 3)} km²`],
  ["imperviousness_pct", "imperviousness", (v) => `${fmtNum(v, 1)} %`],
  ["riparian_width_m", "riparian width", (v) => `${fmtNum(v, 0)} m`],
  ["riparian_ndvi_mean", "riparian NDVI", (v) => fmtNum(v, 2)],
  ["road_density_km_km2", "road density", (v) => `${fmtNum(v, 1)} km/km²`],
  ["population", "population", (v) => v.toLocaleString("en-GB")],
  ["alan_radiance", "night-time light", (v) => fmtSig(v)],
];

function CatchmentCard({ d }: { d: ReachDetail }) {
  const a = d.properties.catchment_attributes;
  const flags = (a?.flags ?? {}) as Record<string, string | string[]>;
  const values: Record<string, unknown> = { ...(a ?? {}), catchment_area_km2: d.properties.catchment_area_km2 };
  const catchFlags = Array.isArray(flags._catchment) ? flags._catchment : [];
  return (
    <div>
      <table className="w-full t-dense">
        <tbody>
          {ATTRS.map(([k, label, fmt]) => {
            const v = values[k];
            const flag = typeof flags[k] === "string" ? (flags[k] as string) : null;
            return (
              <tr key={k} className="hairline-b">
                <td className="py-1">{label}</td>
                <td className="py-1 text-right t-value-sm">{typeof v === "number" ? fmt(v) : "—"}</td>
                <td className="py-1 pl-2 text-right text-muted">{flag ? humanize(flag) : ""}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="t-dense mt-1 text-muted">
        Aggregated over the upstream contributing catchment, not a buffer around the reach.
        {catchFlags.includes("CATCHMENT_TRUNCATED") && " Catchment truncated at the edge of the hydrology window."}
        {a?.adapter ? ` Land cover: ${String(a.adapter).includes("worldcover") ? "ESA WorldCover 2021 (proxies flagged)" : String(a.adapter)}.` : ""}
      </p>
    </div>
  );
}

function ExposureList({ d }: { d: ReachDetail }) {
  const e = d.properties.exposure;
  if (!e) return <p className="t-ui text-muted">Exposure not computed for this reach. Run `make exposure`.</p>;
  const rows = Object.entries(e.features).sort(
    (a, b) => (a[1].nearest_distance_m ?? Infinity) - (b[1].nearest_distance_m ?? Infinity),
  );
  return (
    <div>
      <table className="w-full t-dense">
        <tbody>
          {rows.map(([k, f]) => (
            <tr key={k} className={`hairline-b ${f.count ? "" : "text-muted"}`}>
              <td className="py-1">{exposureLabel(k)}</td>
              <td className="py-1 text-right t-value-sm">{f.count ?? "—"}</td>
              <td className="py-1 text-right t-value-sm">{f.count ? fmtDistance(f.nearest_distance_m) : ""}</td>
            </tr>
          ))}
          <tr>
            <td className="py-1">residents</td>
            <td className="py-1 text-right t-value-sm">{e.population ?? "—"}</td>
            <td className="py-1 text-right text-muted">GHS-POP</td>
          </tr>
        </tbody>
      </table>
      <p className="t-dense mt-1 text-muted">{e.note}</p>
    </div>
  );
}
