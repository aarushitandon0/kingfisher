import { useEffect, useMemo, useState } from "react";
import type { City, ReachCollection } from "../api/types";
import { Chevron, ErrorNote, Rule, Segmented, Stat, Swatch } from "../components/bits";
import { HydrographRail } from "../components/HydrographRail";
import { ReachPanel } from "../components/ReachPanel";
import { addDays, fmtDay, fmtProb, fmtRange, plural } from "../lib/format";
import { INK_MUTED, legendStops, PROB_BREAKS, RATIO_BREAKS, rampColor } from "../lib/ramp";
import { reachAtDate, type TimelineIndex } from "../lib/timeline";
import type { Loadable } from "../lib/useApi";
import { ReachMap, type ReachValue } from "../map/ReachMap";
import { useStore, type ObservabilityFilter } from "../store";

const wide = (): boolean => typeof window !== "undefined" && !!window.matchMedia?.("(min-width: 768px)").matches;

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

  // Escape closes the reach detail, unless the key belongs to a form control.
  useEffect(() => {
    if (!selected) return;
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (e.key !== "Escape" || (t && /^(INPUT|SELECT|TEXTAREA)$/.test(t.tagName))) return;
      select(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [selected, select]);

  if (city?.status === "NOT_BUILT")
    return (
      <div className="grid flex-1 place-items-center p-8">
        <div className="max-w-md t-body">
          <p className="t-title">{city.name} has no reaches yet.</p>
          <p className="mt-2 text-muted">
            The city is configured (config/cities/{city.city}.yaml) but its pipeline has not been run. Build it with{" "}
            <code className="t-value-sm text-ink">make l0 city={city.city}</code> and the stages after it.
          </p>
        </div>
      </div>
    );

  const hoveredFeature = hovered && fc ? fc.features.find((f) => f.id === hovered) : undefined;

  return (
    // Phones: map, rail (it drives the map), then the detail, as one scrolling column.
    // From 768px: map and detail side by side, the rail across the bottom.
    <div className="flex flex-col md:grid md:min-h-0 md:flex-1 md:grid-cols-[minmax(0,1fr)_340px] md:grid-rows-[minmax(0,1fr)_auto] lg:grid-cols-[minmax(0,1fr)_400px]">
      <div className="relative h-[58vh] min-h-[320px] shrink-0 md:col-start-1 md:row-start-1 md:h-auto md:min-h-0">
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
        {reaches.error && (
          <div className="absolute inset-x-3 top-3 z-20 md:right-auto md:max-w-md">
            <ErrorNote error={reaches.error} what="Reaches" />
          </div>
        )}
        <LayersPanel showExposure={showExposure} toggleExposure={toggleExposure} observability={observability} setObservability={setObservability} />
        {hoveredFeature && (
          <HoverReadout
            id={hoveredFeature.id}
            name={hoveredFeature.properties.name}
            value={values.get(hoveredFeature.id)?.value ?? null}
            breaks={breaks}
            sounding={sounding}
          />
        )}
        {fc && <Legend mode={mode} breaks={breaks} fc={fc} />}
      </div>
      <div className="order-2 md:order-none md:col-span-2 md:row-start-2">
        <HydrographRail ix={ix} loading={timeline.loading} error={timeline.error?.detail ?? null} />
      </div>
      <aside
        className="order-3 border-t border-hairline px-4 pt-5 pb-8 md:order-none md:col-start-2 md:row-start-1 md:min-h-0 md:overflow-y-auto md:border-t-0 md:border-l md:pt-4"
        aria-label="Detail"
      >
        {selected ? <ReachPanel key={selected} id={selected} /> : fc && <CitySummary city={city} fc={fc} />}
      </aside>
    </div>
  );
}

function LayersPanel({
  showExposure,
  toggleExposure,
  observability,
  setObservability,
}: {
  showExposure: boolean;
  toggleExposure: () => void;
  observability: ObservabilityFilter;
  setObservability: (f: ObservabilityFilter) => void;
}) {
  return (
    <div className="map-panel absolute top-3 left-3 z-10 flex max-w-[calc(100%-4.5rem)] flex-col gap-2 p-2">
      <Segmented
        dense
        label="Show reaches"
        value={observability}
        onChange={setObservability}
        options={[
          { value: "all", label: "All reaches" },
          { value: "observable", label: "Observable" },
          { value: "driver", label: "Driver-predicted" },
        ]}
      />
      <label className="t-dense flex cursor-pointer items-center gap-2 px-0.5">
        <input type="checkbox" checked={showExposure} onChange={toggleExposure} />
        Exposure features
      </label>
    </div>
  );
}

/** What the pointer is over, read out in one line - the map's cursor readout. */
function HoverReadout({ id, name, value, breaks, sounding }: { id: string; name: string | null; value: number | null; breaks: number[]; sounding: (v: number) => string }) {
  return (
    <div className="map-panel t-dense pointer-events-none absolute top-3 right-14 z-10 hidden max-w-[40%] items-center gap-2 px-2.5 py-1.5 lg:flex" aria-hidden>
      {value !== null && Number.isFinite(value) ? <Swatch color={rampColor(value, breaks)} /> : <Swatch hatch />}
      <span className="text-ink">{id}</span>
      <span className="truncate text-muted">{name ?? "unnamed channel"}</span>
      <span className="t-value-sm text-ink">{value !== null && Number.isFinite(value) ? sounding(value) : "no value"}</span>
    </div>
  );
}

function Legend({ mode, breaks, fc }: { mode: "now" | "past" | "future"; breaks: number[]; fc: ReachCollection }) {
  const [open, setOpen] = useState(wide);
  const title =
    mode === "past"
      ? "Observed ÷ seasonal threshold"
      : mode === "future"
        ? "P(exceed) on the rendered day; halo width = forecast spread"
        : fc.forecast_issued_date
          ? `Peak P(exceed), forecast issued ${fmtDay(fc.forecast_issued_date)}`
          : "Peak P(exceed) — no production forecast yet";
  return (
    <div className="map-panel t-dense absolute bottom-8 left-3 z-10 max-w-[260px]">
      <button type="button" aria-expanded={open} aria-controls="map-legend" onClick={() => setOpen(!open)} className="flex w-full items-start gap-1.5 px-2 py-1.5 text-left text-ink hover:bg-paper-alt">
        <Chevron className="mt-0.5" />
        <span>{title}</span>
      </button>
      {open && (
        <div id="map-legend" className="grid grid-cols-[auto_1fr] items-center gap-x-2 gap-y-0.5 px-2 pb-2">
          {legendStops(breaks, (v) => v.toFixed(1)).map((s) => (
            <div key={s.label} className="contents">
              <Swatch color={s.color} />
              <span className="t-value-sm">{s.label}</span>
            </div>
          ))}
          <Swatch hatch />
          <span>{mode === "past" ? "no clear-sky reading in the 10 days before" : "insufficient evidence — no threshold"}</span>
          <Swatch color={INK_MUTED} dashed />
          <span>driver-predicted (dashed)</span>
          <span className="inline-block h-2.5 w-2.5 justify-self-center rounded-full border border-paper bg-alert" aria-hidden />
          <span>alert, at the downstream end</span>
        </div>
      )}
    </div>
  );
}

function CitySummary({ city, fc }: { city: City | undefined; fc: ReachCollection }) {
  const select = useStore((s) => s.select);
  const hovered = useStore((s) => s.hovered);
  const ranked = fc.features
    .filter((f) => f.properties.p_exceed_max !== null)
    .sort((a, b) => (b.properties.p_exceed_max ?? 0) - (a.properties.p_exceed_max ?? 0));
  const alerts = fc.features.filter((f) => f.properties.alert_severity === "ALERT").length;
  const watches = fc.features.filter((f) => f.properties.alert_severity === "WATCH").length;
  const noThreshold = fc.features.filter((f) => f.properties.exceedance_status !== "OK").length;
  const hoverTo = (id: string | null) => useStore.getState().hover(id);
  return (
    <div>
      <h2 className="t-display">{city?.name ?? fc.city}</h2>
      <p className="t-ui mt-1 text-muted">
        {plural(fc.features.length, "reach", "reaches")}. {city?.optically_observable ?? "—"} optically observable at 10 m, {city?.driver_only ?? "—"} driver-predicted.
      </p>

      {fc.alert_run === "NO_ALERT_RUN" ? (
        <p className="t-ui mt-4 text-muted">No alert run on record. This is not "no alerts": nothing has been assessed.</p>
      ) : (
        <>
          <div className="mt-4 grid grid-cols-3 gap-2">
            <Stat label="Alerts" value={alerts} severity="ALERT" />
            <Stat label="Watches" value={watches} severity="WATCH" />
            <Stat label="No probability" value={noThreshold} severity="INSUFFICIENT_EVIDENCE" />
          </div>
          <p className="t-dense mt-2 text-muted">
            In the alert run issued <span className="t-value-sm text-ink">{fmtDay(fc.alert_run_issued_date, true)}</span>. Reaches with no exceedance probability are hatched on the map.
          </p>
        </>
      )}
      <p className="t-dense mt-1 break-all text-muted">
        Forecast model <span className="t-value-sm">{fc.forecast_model_version ?? "none"}</span>
      </p>

      <Rule>Reaches by peak exceedance probability</Rule>
      {!fc.forecast_issued_date && (
        <p className="t-ui text-muted">
          No production forecast on record for this city, so no reach has an exceedance probability yet. Build one with{" "}
          <code className="t-value-sm text-ink">make evaluate city={fc.city}</code>.
        </p>
      )}
      {fc.forecast_issued_date && ranked.length === 0 && (
        <p className="t-ui text-muted">No reach has a seasonal threshold yet, so none can be ranked.</p>
      )}
      <ol className="t-dense" onMouseLeave={() => hoverTo(null)}>
        {ranked.map((f) => {
          const p = f.properties.p_exceed_max;
          const isHover = hovered === f.id;
          return (
            <li key={f.id}>
              <button
                type="button"
                onClick={() => select(f.id)}
                onFocus={() => hoverTo(f.id)}
                onBlur={() => hoverTo(null)}
                onMouseEnter={() => hoverTo(f.id)}
                className={`grid min-h-8 w-full grid-cols-[4.5rem_1fr_auto] items-center gap-2 border-b border-hairline px-1 py-1 text-left hover:bg-paper-alt ${isHover ? "bg-paper-alt" : ""}`}
              >
                <span className="text-ink">{f.id}</span>
                <span className="truncate text-muted">{f.properties.name ?? "unnamed channel"}</span>
                <span className="flex items-center gap-2">
                  {f.properties.alert_severity === "ALERT" && <span className="text-alert">Alert</span>}
                  {f.properties.alert_severity === "WATCH" && <span className="text-watch-text">Watch</span>}
                  <span aria-hidden className="inline-block h-[3px] w-4" style={{ background: rampColor(p, PROB_BREAKS) }} />
                  <span className="t-value-sm w-9 text-right">{fmtProb(p)}</span>
                </span>
              </button>
            </li>
          );
        })}
      </ol>
      {fc.forecast_issued_date && ranked.length > 0 && (
        <p className="t-dense mt-2 text-muted">
          Peak daily probability over <span className="t-value-sm">{fmtRange(addDays(fc.forecast_issued_date, 1), addDays(fc.forecast_issued_date, 10))}</span>. Driver-predicted reaches are not ranked: they have no threshold.
        </p>
      )}
    </div>
  );
}
