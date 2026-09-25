import type { Map as MLMap } from "maplibre-gl";
import { useEffect, useMemo, useRef, useState } from "react";
import type { City, ReachCollection } from "../api/types";
import { Chevron, ErrorNote, Rule, Segmented, SeverityLabel, Skeleton, Stat, Swatch } from "../components/bits";
import { HydrographRail } from "../components/HydrographRail";
import { Splitter, usePanelSize } from "../components/Splitter";
import { useShortcut } from "../lib/shortcuts";
import { ReachPanel } from "../components/ReachPanel";
import { addDays, fmtDay, fmtProb, fmtRange, plural } from "../lib/format";
import { INK_MUTED, legendStops, PROB_BREAKS, RATIO_BREAKS, rampColor } from "../lib/ramp";
import { reachAtDate, type TimelineIndex } from "../lib/timeline";
import type { Loadable } from "../lib/useApi";
import { featureBounds, ReachMap, type ReachValue } from "../map/ReachMap";
import { useStore, type ObservabilityFilter } from "../store";

const wide = (): boolean => typeof window !== "undefined" && !!window.matchMedia?.("(min-width: 768px)").matches;

export function MapView({ city, reaches, timeline }: { city: City | undefined; reaches: Loadable<ReachCollection>; timeline: Loadable<TimelineIndex> }) {
  const { selected, hovered, select, hover, railDate, showExposure, toggleExposure, observability, setObservability } = useStore();
  const fc = reaches.data;
  const ix = timeline.data;
  const [map, setMap] = useState<MLMap | null>(null);
  const [rail, setRail] = usePanelSize("map.rail", null);
  const [chart, setChart] = usePanelSize("map.timeline", null);
  // Collapsing remembers nothing but "closed"; reopening goes back to the default size.
  const toggleRail = () => setRail(rail === 0 ? null : 0);
  const toggleTimeline = () => setChart(chart === 0 ? null : 0);
  useShortcut("b", toggleRail, { ctrl: true });
  useShortcut("j", toggleTimeline, { ctrl: true });

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
        <div className="card card-pad t-body max-w-md">
          <p className="t-title">{city.name} has no reaches yet.</p>
          <p className="mt-2 text-muted">
            The city is configured (config/cities/{city.city}.yaml) but its pipeline has not been run. Build it with{" "}
            <code className="t-value-sm text-ink">make l0 city={city.city}</code> and the stages after it.
          </p>
        </div>
      </div>
    );

  const hoveredFeature = hovered && fc ? fc.features.find((f) => f.id === hovered) : undefined;

  const railW = rail ?? RAIL_DEFAULT;
  const chartH = chart ?? CHART_DEFAULT;
  const maxRail = Math.max(RAIL_MIN, Math.min(900, (typeof window !== "undefined" ? window.innerWidth : 1440) - 480));

  return (
    // Phones: map, rail (it drives the map), then the detail, as one scrolling column.
    // From 768px: VS Code-style panes - the map, a resizable timeline panel under it, and a
    // resizable (collapsible) detail sidebar on the right. The gaps are the drag handles.
    <div
      className="flex flex-col gap-4 p-3 md:min-h-0 md:flex-1 md:flex-row md:gap-0 md:p-4"
      style={{ ["--rail-w" as string]: `${railW}px` }}
    >
      <div className="flex min-w-0 flex-col gap-4 md:min-h-0 md:flex-1 md:gap-0">
        <div className="map-frame h-[62vh] min-h-[360px] shrink-0 md:h-auto md:min-h-[240px] md:flex-1">
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
              onMap={setMap}
            />
          )}
          {!fc && !reaches.error && (
            <div className="absolute inset-0 grid place-items-center" role="status">
              <span className="map-panel t-dense flex items-center gap-2.5 px-3.5 py-2 text-muted">
                <span aria-hidden className="h-2 w-2 animate-pulse rounded-full bg-water" />
                Loading the stream network
              </span>
            </div>
          )}
          {reaches.error && (
            <div className="absolute inset-x-3 top-16 z-20 md:right-auto md:max-w-md">
              <ErrorNote error={reaches.error} what="Reaches" />
            </div>
          )}
          <div className="absolute top-3 left-3 z-20 flex w-[min(340px,calc(100%-4.5rem))] flex-col gap-2">
            {fc && <MapSearch fc={fc} map={map} onReach={select} />}
            <LayersPanel showExposure={showExposure} toggleExposure={toggleExposure} observability={observability} setObservability={setObservability} />
          </div>
          {hoveredFeature && (
            <HoverReadout
              id={hoveredFeature.id}
              name={hoveredFeature.properties.name}
              value={values.get(hoveredFeature.id)?.value ?? null}
              severity={hoveredFeature.properties.alert_severity}
              breaks={breaks}
              sounding={sounding}
              observable={hoveredFeature.properties.observable === true}
            />
          )}
          {fc && <Legend mode={mode} breaks={breaks} fc={fc} />}
          <PaneToggles railOpen={railW > 0} timelineOpen={chartH > 0} toggleRail={toggleRail} toggleTimeline={toggleTimeline} />
        </div>
        <Splitter
          axis="y"
          reverse
          collapsible
          size={chartH}
          onSize={setChart}
          min={70}
          max={Math.max(120, (typeof window !== "undefined" ? window.innerHeight : 900) - 420)}
          label="Timeline panel"
          className="hidden md:flex"
        />
        <div className={`card order-2 overflow-hidden md:order-none ${chartH === 0 ? "md:hidden" : ""}`}>
          <HydrographRail ix={ix} loading={timeline.loading} error={timeline.error?.detail ?? null} height={chartH || CHART_DEFAULT} />
        </div>
      </div>
      <Splitter axis="x" reverse collapsible size={railW} onSize={setRail} min={RAIL_MIN} max={maxRail} label="Detail sidebar" className="hidden md:flex" />
      <aside
        className={`card order-3 px-5 pt-5 pb-8 md:order-none md:min-h-0 md:w-(--rail-w) md:shrink-0 md:overflow-y-auto ${railW === 0 ? "md:hidden" : ""}`}
        aria-label="Detail"
      >
        {selected ? <ReachPanel key={selected} id={selected} /> : fc ? <CitySummary city={city} fc={fc} /> : !reaches.error && <SummarySkeleton />}
      </aside>
    </div>
  );
}

/** The city summary's outline while the reach network loads. */
function SummarySkeleton() {
  return (
    <div aria-hidden>
      <span className="skeleton block h-3 w-28" />
      <span className="skeleton mt-3 block h-8 w-44" />
      <span className="skeleton mt-3 block h-3.5 w-56" />
      <div className="mt-6 grid grid-cols-3 gap-3">
        {[0, 1, 2, 3, 4, 5].map((i) => (
          <span key={i} className="skeleton block h-[104px] !rounded-[var(--radius-md)]" />
        ))}
      </div>
      <Skeleton label="stream network" lines={6} className="mt-8" />
    </div>
  );
}

const RAIL_DEFAULT = 400;
const RAIL_MIN = 300;
const CHART_DEFAULT = 140;

/** Toggle buttons for the two side panes, bottom-right of the map (Ctrl+B / Ctrl+J). */
function PaneToggles({ railOpen, timelineOpen, toggleRail, toggleTimeline }: { railOpen: boolean; timelineOpen: boolean; toggleRail: () => void; toggleTimeline: () => void }) {
  const btn = (on: boolean) => `grid h-9 w-9 place-items-center rounded-[var(--radius-sm)] ${on ? "text-brand-300" : "text-muted hover:text-ink"} hover:bg-raised`;
  return (
    <div className="map-panel absolute top-[146px] right-3 z-10 hidden flex-col gap-0.5 p-0.5 md:flex">
      <button type="button" className={btn(timelineOpen)} onClick={toggleTimeline} aria-pressed={timelineOpen} title="Toggle timeline panel (Ctrl+J)" aria-label="Toggle timeline panel">
        <svg aria-hidden viewBox="0 0 16 16" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.5">
          <rect x="1.75" y="2.75" width="12.5" height="10.5" rx="1.5" />
          <path d="M1.75 9.5h12.5" />
          {timelineOpen && <path d="M2.5 10.2h11v2.3h-11z" fill="currentColor" stroke="none" />}
        </svg>
      </button>
      <button type="button" className={btn(railOpen)} onClick={toggleRail} aria-pressed={railOpen} title="Toggle detail sidebar (Ctrl+B)" aria-label="Toggle detail sidebar">
        <svg aria-hidden viewBox="0 0 16 16" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.5">
          <rect x="1.75" y="2.75" width="12.5" height="10.5" rx="1.5" />
          <path d="M10 2.75v10.5" />
          {railOpen && <path d="M10.7 3.5h2.8v9h-2.8z" fill="currentColor" stroke="none" />}
        </svg>
      </button>
    </div>
  );
}

interface SearchHit {
  kind: "reach" | "river" | "place";
  key: string;
  label: string;
  sub: string;
  go: () => void;
}

/** Search reach IDs, stream names and basemap place names; Enter or click flies there. */
function MapSearch({ fc, map, onReach }: { fc: ReachCollection; map: MLMap | null; onReach: (id: string) => void }) {
  const [q, setQ] = useState("");
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const box = useRef<HTMLDivElement>(null);

  const rivers = useMemo(() => {
    const m = new Map<string, string[]>();
    for (const f of fc.features) {
      const n = f.properties.name;
      if (n) m.set(n, [...(m.get(n) ?? []), f.id]);
    }
    return m;
  }, [fc]);

  const hits = useMemo<SearchHit[]>(() => {
    const t = q.trim().toLowerCase();
    if (!t) return [];
    const out: SearchHit[] = [];
    for (const [name, ids] of rivers)
      if (name.toLowerCase().includes(t))
        out.push({
          kind: "river",
          key: `r:${name}`,
          label: name,
          sub: plural(ids.length, "reach", "reaches"),
          go: () => {
            if (!map) return;
            const b = featureBounds({ ...fc.features[0], geometry: { type: "MultiLineString", coordinates: fc.features.filter((f) => ids.includes(f.id)).map((f) => f.geometry.coordinates) } });
            if (b) map.fitBounds(b, { padding: 80, maxZoom: 15, duration: 700 });
          },
        });
    const rv = out.slice(0, 5);
    const reachHits = fc.features
      .filter((f) => f.id.toLowerCase().includes(t))
      .slice(0, 6)
      .map<SearchHit>((f) => ({ kind: "reach", key: `id:${f.id}`, label: f.id, sub: f.properties.name ?? "unnamed channel", go: () => onReach(f.id) }));
    const places: SearchHit[] = [];
    if (map && t.length >= 2) {
      const seen = new Set<string>();
      try {
        for (const f of map.querySourceFeatures("openmaptiles", { sourceLayer: "place" })) {
          const name = String(f.properties?.name ?? f.properties?.["name:latin"] ?? "");
          if (!name || seen.has(name) || !name.toLowerCase().includes(t) || f.geometry.type !== "Point") continue;
          seen.add(name);
          const c = f.geometry.coordinates as [number, number];
          places.push({ kind: "place", key: `p:${name}`, label: name, sub: String(f.properties?.class ?? "place"), go: () => map.flyTo({ center: c, zoom: 14.5, duration: 800 }) });
          if (places.length >= 5) break;
        }
      } catch {
        /* basemap unavailable: reaches and streams are still searchable */
      }
    }
    return [...reachHits, ...rv, ...places];
  }, [q, rivers, fc, map, onReach]);

  useEffect(() => setActive(0), [q]);
  useEffect(() => {
    const onDoc = (e: MouseEvent) => {
      if (box.current && !box.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, []);

  const pick = (h: SearchHit) => {
    h.go();
    setQ(h.kind === "reach" ? h.label : h.label);
    setOpen(false);
  };

  return (
    <div ref={box} className="relative">
      <div className="search map-panel !rounded-full">
        <svg aria-hidden viewBox="0 0 16 16" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round">
          <circle cx="7" cy="7" r="4.5" />
          <path d="m10.5 10.5 3 3" />
        </svg>
        <input
          type="search"
          role="combobox"
          aria-expanded={open && hits.length > 0}
          aria-controls="map-search-list"
          aria-activedescendant={open && hits[active] ? `hit-${active}` : undefined}
          aria-label="Search reach ID, stream or place" data-search
          placeholder="Search reach ID, stream or place…"
          value={q}
          onChange={(e) => {
            setQ(e.target.value);
            setOpen(true);
          }}
          onFocus={() => setOpen(true)}
          onKeyDown={(e) => {
            if (e.key === "ArrowDown") {
              e.preventDefault();
              setActive((a) => Math.min(hits.length - 1, a + 1));
            } else if (e.key === "ArrowUp") {
              e.preventDefault();
              setActive((a) => Math.max(0, a - 1));
            } else if (e.key === "Enter" && hits[active]) {
              e.preventDefault();
              pick(hits[active]);
            } else if (e.key === "Escape") {
              setOpen(false);
            }
          }}
        />
        {q && (
          <button type="button" aria-label="Clear search" onClick={() => setQ("")} className="-mr-1 grid h-7 w-7 place-items-center rounded-full text-muted hover:bg-raised hover:text-ink">
            <svg aria-hidden viewBox="0 0 12 12" width="11" height="11">
              <path d="M2.5 2.5l7 7M9.5 2.5l-7 7" stroke="currentColor" strokeWidth="1.5" />
            </svg>
          </button>
        )}
      </div>
      {open && q.trim() && (
        <ul id="map-search-list" role="listbox" className="map-panel absolute inset-x-0 top-[42px] z-30 max-h-[320px] overflow-y-auto p-1.5">
          {hits.length === 0 && <li className="t-dense px-2.5 py-2 text-muted">No reach, stream or loaded place matches "{q}".</li>}
          {hits.map((h, i) => (
            <li
              key={h.key}
              id={`hit-${i}`}
              role="option"
              aria-selected={i === active}
              onMouseEnter={() => setActive(i)}
              onMouseDown={(e) => {
                e.preventDefault();
                pick(h);
              }}
              className={`flex cursor-pointer items-center gap-2.5 rounded-md px-2.5 py-2 ${i === active ? "bg-raised" : ""}`}
            >
              <span aria-hidden className="grid h-6 w-6 shrink-0 place-items-center rounded-md bg-raised text-water">
                {h.kind === "place" ? (
                  <svg viewBox="0 0 16 16" width="13" height="13" fill="none" stroke="currentColor" strokeWidth="1.6">
                    <path d="M8 14s4.5-4.2 4.5-7.5a4.5 4.5 0 0 0-9 0C3.5 9.8 8 14 8 14Z" />
                    <circle cx="8" cy="6.5" r="1.5" />
                  </svg>
                ) : (
                  <svg viewBox="0 0 16 16" width="13" height="13" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round">
                    <path d="M2 10c2-2 4-2 6 0s4 2 6 0" />
                    {h.kind === "river" && <path d="M2 6c2-2 4-2 6 0s4 2 6 0" />}
                  </svg>
                )}
              </span>
              <span className="min-w-0 flex-1">
                <span className={`block truncate text-[13px] text-ink ${h.kind === "reach" ? "font-mono" : "font-medium"}`}>{h.label}</span>
                <span className="t-dense block truncate text-muted">{h.sub}</span>
              </span>
              <span className="t-eyebrow shrink-0">{h.kind === "river" ? "stream" : h.kind}</span>
            </li>
          ))}
        </ul>
      )}
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
    <div className="map-panel flex max-w-full flex-col gap-2.5 self-start p-2.5">
      <Segmented
        dense
        label="Show reaches"
        value={observability}
        onChange={setObservability}
        options={[
          { value: "all", label: <>All<span className="hidden sm:inline"> reaches</span></> },
          { value: "observable", label: "Observable" },
          { value: "driver", label: <>Driver<span className="hidden sm:inline">-predicted</span></> },
        ]}
      />
      <label className="flex cursor-pointer items-center gap-2 rounded-md px-1.5 py-0.5 text-[13px] hover:bg-raised">
        <input type="checkbox" checked={showExposure} onChange={toggleExposure} />
        Exposure features
        <span className="t-dense hidden text-muted sm:inline">(schools, parks, paths)</span>
      </label>
    </div>
  );
}

/** What the pointer is over, read out in one card - the map's cursor readout. */
function HoverReadout({
  id,
  name,
  value,
  severity,
  breaks,
  sounding,
  observable,
}: {
  id: string;
  name: string | null;
  value: number | null;
  severity: import("../api/types").Severity | null;
  breaks: number[];
  sounding: (v: number) => string;
  observable: boolean;
}) {
  const has = value !== null && Number.isFinite(value);
  return (
    <div className="map-panel pointer-events-none absolute top-3 left-[364px] z-10 hidden max-w-[calc(100%-364px-4.5rem)] items-center gap-3 px-3.5 py-2 xl:flex" aria-hidden>
      {has ? <Swatch color={rampColor(value, breaks)} width={4} /> : <Swatch hatch />}
      <span className="flex flex-col">
        <span className="flex items-center gap-2">
          <span className="font-mono text-[13px] font-medium text-ink">{id}</span>
          <span className="max-w-[200px] truncate text-[13px] text-muted">{name ?? "unnamed channel"}</span>
        </span>
        <span className="t-dense text-faint">{observable ? "optically observable" : "driver-predicted"}</span>
      </span>
      <span className="t-value text-ink">{has ? sounding(value) : "no value"}</span>
      {severity && severity !== "INSUFFICIENT_EVIDENCE" && <SeverityLabel s={severity} small />}
    </div>
  );
}

function Legend({ mode, breaks, fc }: { mode: "now" | "past" | "future"; breaks: number[]; fc: ReachCollection }) {
  const [open, setOpen] = useState(wide);
  const title =
    mode === "past"
      ? "Observed ÷ seasonal threshold"
      : mode === "future"
        ? "P(exceed) on the rendered day"
        : fc.forecast_issued_date
          ? `Peak P(exceed), next 10 days`
          : "Peak P(exceed): no forecast yet";
  const stops = legendStops(breaks, (v) => v.toFixed(1));
  return (
    <div className="map-panel absolute bottom-8 left-3 z-10 w-[268px] max-w-[calc(100%-1.5rem)]">
      <button type="button" aria-expanded={open} aria-controls="map-legend" onClick={() => setOpen(!open)} className="flex w-full items-center gap-2 rounded-[var(--radius-md)] px-4 py-2.5 text-left">
        <Chevron className="text-muted" />
        <span className="t-eyebrow text-muted">{title}</span>
      </button>
      {open && (
        <div id="map-legend" className="px-4 pb-3.5 text-[12px]">
          {/* The ramp as one bar: low (clear water) to high (silt), tick at each break. */}
          <div className="flex h-2 overflow-hidden rounded-full" aria-hidden>
            {stops.map((s) => (
              <span key={s.label} className="flex-1" style={{ background: s.color }} />
            ))}
          </div>
          <div className="relative mt-1 h-4 font-mono text-[10.5px] text-muted" aria-hidden>
            {breaks.map((b, i) => (
              <span key={b} className="absolute -translate-x-1/2" style={{ left: `${((i + 1) / stops.length) * 100}%` }}>
                {b.toFixed(1)}
              </span>
            ))}
          </div>
          <p className="sr-only">{stops.map((s) => s.label).join(", ")}</p>
          <div className="mt-2 grid grid-cols-[auto_1fr] items-center gap-x-2.5 gap-y-1.5">
            <Swatch hatch />
            <span className="text-muted">{mode === "past" ? "no clear-sky reading in 10 days" : "insufficient evidence (no threshold)"}</span>
            <Swatch color={INK_MUTED} dashed />
            <span className="text-muted">driver-predicted</span>
            <span className="relative inline-grid h-3 w-6 place-items-center" aria-hidden>
              <span className="h-2.5 w-2.5 rounded-full bg-[var(--severity-critical)] shadow-[0_0_0_3px_color-mix(in_srgb,var(--severity-critical)_35%,transparent)]" />
            </span>
            <span className="text-muted">alert, at the downstream end</span>
          </div>
        </div>
      )}
    </div>
  );
}

function CitySummary({ city, fc }: { city: City | undefined; fc: ReachCollection }) {
  const select = useStore((s) => s.select);
  const hovered = useStore((s) => s.hovered);
  const [filter, setFilter] = useState("");
  const ranked = fc.features
    .filter((f) => f.properties.p_exceed_max !== null)
    .sort((a, b) => (b.properties.p_exceed_max ?? 0) - (a.properties.p_exceed_max ?? 0));
  const shown = filter.trim()
    ? ranked.filter((f) => f.id.toLowerCase().includes(filter.trim().toLowerCase()) || (f.properties.name ?? "").toLowerCase().includes(filter.trim().toLowerCase()))
    : ranked;
  const alerts = fc.features.filter((f) => f.properties.alert_severity === "ALERT").length;
  const watches = fc.features.filter((f) => f.properties.alert_severity === "WATCH").length;
  const noThreshold = fc.features.filter((f) => f.properties.exceedance_status !== "OK").length;
  const hoverTo = (id: string | null) => useStore.getState().hover(id);
  return (
    <div>
      <p className="t-eyebrow">Stream network</p>
      <h2 className="t-display mt-1">{city?.name ?? fc.city}</h2>
      {fc.forecast_issued_date && (
        <p className="t-ui mt-1 text-muted">
          10-day forecast issued <span className="text-ink">{fmtDay(fc.forecast_issued_date, true)}</span>
        </p>
      )}
      <div className="mt-5 grid grid-cols-3 gap-3">
        <Stat className="!px-3 [&_.t-eyebrow]:tracking-[0.03em]" label="Reaches" value={fc.features.length} note="in the network" />
        <Stat className="!px-3 [&_.t-eyebrow]:tracking-[0.03em]" label="Observable" value={city?.optically_observable ?? "—"} note="at 10 m" tone="brand" />
        <Stat className="!px-3 [&_.t-eyebrow]:tracking-[0.03em]" label="Driver-only" value={city?.driver_only ?? "—"} note="weather + catchment" />
      </div>
      <p className="sr-only">
        {plural(fc.features.length, "reach", "reaches")}. {city?.optically_observable ?? "—"} optically observable at 10 m, {city?.driver_only ?? "—"} driver-predicted.
      </p>

      {fc.alert_run === "NO_ALERT_RUN" ? (
        <p className="t-ui mt-3 rounded-md bg-raised px-3 py-2 text-muted">No alert run on record. This is not "no alerts": nothing has been assessed.</p>
      ) : (
        <>
          <div className="mt-3 grid grid-cols-3 gap-3">
            <Stat className="!px-3 [&_.t-eyebrow]:tracking-[0.03em]" label="Alerts" value={alerts} severity="ALERT" />
            <Stat className="!px-3 [&_.t-eyebrow]:tracking-[0.03em]" label="Watches" value={watches} severity="WATCH" />
            <Stat className="!px-3 [&_.t-eyebrow]:tracking-[0.03em]" label="No prob." value={noThreshold} severity="INSUFFICIENT_EVIDENCE" note="no threshold" />
          </div>
          <p className="t-dense mt-3 text-muted">
            From the alert run issued <span className="text-ink">{fmtDay(fc.alert_run_issued_date, true)}</span>. Reaches with no exceedance probability are hatched on the map.
          </p>
        </>
      )}
      <p className="t-dense mt-1 break-all text-faint">
        Model <span className="font-mono">{fc.forecast_model_version ?? "none"}</span>
      </p>

      <Rule>Reaches by peak exceedance probability</Rule>
      {!fc.forecast_issued_date && (
        <p className="t-ui text-muted">
          No production forecast on record for this city, so no reach has an exceedance probability yet. Build one with{" "}
          <code className="t-value-sm text-ink">make evaluate city={fc.city}</code>.
        </p>
      )}
      {fc.forecast_issued_date && ranked.length === 0 && <p className="t-ui text-muted">No reach has a seasonal threshold yet, so none can be ranked.</p>}
      {ranked.length > 8 && (
        <div className="search mb-2 !h-8">
          <svg aria-hidden viewBox="0 0 16 16" width="13" height="13" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round">
            <circle cx="7" cy="7" r="4.5" />
            <path d="m10.5 10.5 3 3" />
          </svg>
          <input value={filter} onChange={(e) => setFilter(e.target.value)} placeholder={`Filter ${ranked.length} ranked reaches`} aria-label="Filter ranked reaches" />
        </div>
      )}
      <ol onMouseLeave={() => hoverTo(null)}>
        {shown.map((f, i) => {
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
                className={`group row-hover grid min-h-11 w-full grid-cols-[1.25rem_4.75rem_1fr_auto] items-center gap-2 rounded-md px-2 py-1.5 text-left ${isHover ? "bg-raised" : ""}`}
                title={`Fly to ${f.id} and open its detail`}
              >
                <span className="t-dense text-right text-faint">{filter ? "" : i + 1}</span>
                <span className="font-mono text-[12.5px] font-medium text-brand-text group-hover:underline">{f.id}</span>
                <span className="truncate text-[13px] text-muted">{f.properties.name ?? "unnamed channel"}</span>
                <span className="flex items-center gap-2">
                  {f.properties.alert_severity === "ALERT" && <SeverityLabel s="ALERT" small />}
                  {f.properties.alert_severity === "WATCH" && <SeverityLabel s="WATCH" small />}
                  <span aria-hidden className="inline-block h-1.5 w-5 rounded-full" style={{ background: rampColor(p, PROB_BREAKS) }} />
                  <span className="t-value-sm w-9 text-right text-ink">{fmtProb(p)}</span>
                </span>
              </button>
            </li>
          );
        })}
      </ol>
      {fc.forecast_issued_date && ranked.length > 0 && (
        <p className="t-dense mt-3 text-muted">
          Peak daily probability over <span className="text-ink">{fmtRange(addDays(fc.forecast_issued_date, 1), addDays(fc.forecast_issued_date, 10))}</span>. Driver-predicted reaches are not ranked: they have no threshold.
        </p>
      )}
    </div>
  );
}
