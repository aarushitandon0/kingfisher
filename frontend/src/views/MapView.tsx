import { useMemo } from "react";
import type { City, ReachCollection } from "../api/types";
import { ErrorNote, Swatch } from "../components/bits";
import { HydrographRail } from "../components/HydrographRail";
import { ReachPanel } from "../components/ReachPanel";
import { addDays, fmtDay, fmtProb, fmtRange, plural } from "../lib/format";
import { legendStops, PROB_BREAKS, RATIO_BREAKS } from "../lib/ramp";
import { reachAtDate, type TimelineIndex } from "../lib/timeline";
import type { Loadable } from "../lib/useApi";
import { ReachMap, type ReachValue } from "../map/ReachMap";
import { useStore } from "../store";

export function MapView({ city, reaches, timeline }: { city: City | undefined; reaches: Loadable<ReachCollection>; timeline: Loadable<TimelineIndex> }) {
  const { selected, hovered, select, hover, railDate, showExposure, toggleExposure, observability, setObservability } = useStore();
  const fc = reaches.data;
  const ix = timeline.data;

  const mode: "now" | "past" | "future" = !railDate || !ix?.issued ? "now" : railDate > ix.issued ? "future" : "past";
  const values = useMemo(() => {
    const out = new Map<string, ReachValue>();
    if (!fc) return out;
    for (const f of fc.features) {
      if (mode === "now" || !ix || !railDate) out.set(f.id, { value: f.properties.p_exceed_max });
      else out.set(f.id, reachAtDate(ix, f.id, railDate));
    }
    return out;
  }, [fc, ix, railDate, mode]);

  const visible = useMemo(() => {
    if (!fc || observability === "all") return null;
    return new Set(fc.features.filter((f) => (observability === "observable" ? f.properties.observable === true : f.properties.observable !== true)).map((f) => f.id));
  }, [fc, observability]);

  const breaks = mode === "past" ? RATIO_BREAKS : PROB_BREAKS;
  const sounding = useMemo(() => (mode === "past" ? (v: number) => `${v.toFixed(2)}×` : (v: number) => fmtProb(v)), [mode]);

  if (city?.status === "NOT_BUILT")
    return (
      <div className="grid flex-1 place-items-center p-8">
        <div className="max-w-md t-body">
          <p className="t-title">{city.name} has no reaches yet.</p>
          <p className="mt-2 text-muted">
            The city is configured (config/cities/{city.city}.yaml) but its pipeline has not been run. Build it with{" "}
            <code className="t-value-sm">make l0 city={city.city}</code> and the stages after it.
          </p>
        </div>
      </div>
    );

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="flex min-h-0 flex-1 flex-col md:flex-row">
        <div className="relative min-h-[320px] flex-1">
          <ErrorNote error={reaches.error} what="Reaches" />
          {fc && (
            <ReachMap
              label={`${city?.name ?? fc.city} stream network`}
              reaches={fc}
              values={values}
              breaks={breaks}
              sounding={sounding}
              selected={selected}
              hovered={hovered}
              showExposure={showExposure}
              visible={visible}
              onClick={select}
              onHover={hover}
            />
          )}
          <div className="absolute left-2 top-2 z-10 flex flex-col gap-1 border border-hairline bg-paper/95 px-2 py-1.5">
            <label className="t-dense flex items-center gap-2">
              <input type="checkbox" checked={showExposure} onChange={toggleExposure} />
              Exposure features
            </label>
            <label className="t-dense flex items-center gap-2">
              <span className="text-muted">Show</span>
              <select value={observability} onChange={(e) => setObservability(e.target.value as typeof observability)} className="t-dense border-0 border-b border-hairline bg-transparent">
                <option value="all">all reaches</option>
                <option value="observable">optically observable</option>
                <option value="driver">driver-predicted</option>
              </select>
            </label>
          </div>
          {fc && <Legend mode={mode} breaks={breaks} fc={fc} />}
        </div>
        <aside className="max-h-[50vh] w-full shrink-0 overflow-y-auto border-t border-hairline p-4 md:max-h-none md:w-[400px] md:border-l md:border-t-0" aria-label="Detail">
          {selected ? <ReachPanel key={selected} id={selected} /> : fc && <CitySummary city={city} fc={fc} />}
        </aside>
      </div>
      <HydrographRail ix={ix} loading={timeline.loading} error={timeline.error?.detail ?? null} />
    </div>
  );
}

function Legend({ mode, breaks, fc }: { mode: "now" | "past" | "future"; breaks: number[]; fc: ReachCollection }) {
  const title =
    mode === "past"
      ? "Observed ÷ seasonal threshold"
      : mode === "future"
        ? "P(exceed) on the rendered day; halo width = forecast spread"
        : fc.forecast_issued_date
          ? `Peak P(exceed), forecast issued ${fmtDay(fc.forecast_issued_date)}`
          : "Peak P(exceed) — no production forecast yet";
  return (
    <div className="absolute bottom-7 left-2 z-10 max-w-[260px] border border-hairline bg-paper/95 px-2 py-1.5 t-dense">
      <p className="mb-1 text-ink">{title}</p>
      <div className="grid grid-cols-[auto_1fr] items-center gap-x-2 gap-y-0.5">
        {legendStops(breaks, (v) => (mode === "past" ? v.toFixed(1) : v.toFixed(1))).map((s) => (
          <div key={s.label} className="contents">
            <Swatch color={s.color} />
            <span className="t-value-sm">{s.label}</span>
          </div>
        ))}
        <Swatch hatch />
        <span>{mode === "past" ? "no clear-sky reading in the 10 days before" : "insufficient evidence — no threshold"}</span>
        <Swatch color="#5A605D" dashed />
        <span>driver-predicted (dashed)</span>
        <span className="inline-block h-2.5 w-2.5 justify-self-center rounded-full border border-paper bg-alert" aria-hidden />
        <span>alert, at the downstream end</span>
      </div>
    </div>
  );
}

function CitySummary({ city, fc }: { city: City | undefined; fc: ReachCollection }) {
  const select = useStore((s) => s.select);
  const ranked = fc.features
    .filter((f) => f.properties.p_exceed_max !== null)
    .sort((a, b) => (b.properties.p_exceed_max ?? 0) - (a.properties.p_exceed_max ?? 0));
  const alerts = fc.features.filter((f) => f.properties.alert_severity === "ALERT").length;
  const watches = fc.features.filter((f) => f.properties.alert_severity === "WATCH").length;
  const noThreshold = fc.features.filter((f) => f.properties.exceedance_status !== "OK").length;
  return (
    <div>
      <h2 className="t-display">{city?.name ?? fc.city}</h2>
      <p className="t-ui text-muted">
        {plural(fc.features.length, "reach", "reaches")}. {city?.optically_observable ?? "—"} optically observable at 10 m, {city?.driver_only ?? "—"} driver-predicted.
      </p>
      <p className="t-ui mt-3">
        {fc.alert_run === "NO_ALERT_RUN" ? (
          <span className="text-muted">No alert run on record.</span>
        ) : (
          <>
            <span className="text-alert">{plural(alerts, "alert")}</span>, {plural(watches, "watch", "watches")} in the run issued {fmtDay(fc.alert_run_issued_date, true)}.{" "}
            <span className="text-muted">{noThreshold} reaches carry no exceedance probability — hatched on the map.</span>
          </>
        )}
      </p>
      <p className="t-dense mt-1 text-muted">
        Forecast model <span className="t-value-sm">{fc.forecast_model_version ?? "none"}</span>
      </p>

      <div className="mt-5 mb-1 flex items-center gap-2">
        <span className="t-dense text-muted">reaches by peak exceedance probability</span>
        <span className="h-px flex-1 bg-hairline" />
      </div>
      {!fc.forecast_issued_date && (
        <p className="t-ui text-muted">
          No production forecast on record for this city, so no reach has an exceedance probability yet. Build one with{" "}
          <code className="t-value-sm">make evaluate city={fc.city}</code>.
        </p>
      )}
      {fc.forecast_issued_date && ranked.length === 0 && (
        <p className="t-ui text-muted">No reach has a seasonal threshold yet, so none can be ranked.</p>
      )}
      <ol className="t-dense">
        {ranked.map((f) => (
          <li key={f.id}>
            <button
              type="button"
              onClick={() => select(f.id)}
              onFocus={() => useStore.getState().hover(f.id)}
              onBlur={() => useStore.getState().hover(null)}
              className="grid w-full grid-cols-[4.5rem_1fr_3rem] items-baseline gap-2 border-b border-hairline py-1 text-left hover:bg-paper-alt"
            >
              <span>{f.id}</span>
              <span className="truncate text-muted">{f.properties.name ?? "unnamed channel"}</span>
              <span className="t-value-sm text-right">{fmtProb(f.properties.p_exceed_max)}</span>
            </button>
          </li>
        ))}
      </ol>
      {fc.forecast_issued_date && ranked.length > 0 && (
        <p className="t-dense mt-2 text-muted">
          Peak daily probability over <span className="t-value-sm">{fmtRange(addDays(fc.forecast_issued_date, 1), addDays(fc.forecast_issued_date, 10))}</span>. Driver-predicted reaches are not ranked: they have no threshold.
        </p>
      )}
    </div>
  );
}
