import type { Map as MLMap } from "maplibre-gl";
import { useEffect, useMemo, useRef, useState } from "react";
import { api, ApiError } from "../api/client";
import type { Citation, InterventionInfo, ReachCollection, ReachResult, ScenarioResult, Variable } from "../api/types";
import { Chevron, ErrorNote, Loading, Segmented, Swatch } from "../components/bits";
import type { LeverState } from "../store";
import { fmtDay, fmtMoney, fmtNum, fmtProb, humanize, VARIABLE_LABEL } from "../lib/format";
import { DAYS_BREAKS, legendStops, PROB_BREAKS, SCENARIO } from "../lib/ramp";
import { useApi, type Loadable } from "../lib/useApi";
import { ReachMap, type ReachValue } from "../map/ReachMap";
import { useStore } from "../store";

const CAVEAT = "Planning estimate. Intervention effect sizes from cited literature applied to a statistical model; not a causal experiment.";

const PATH_TEXT: Record<string, string> = {
  MODEL_PERTURBATION: "through the model",
  LITERATURE_DIRECT: "cited direct effect",
  NOT_ESTIMABLE: "not estimable here",
};

/** Can this lever move either variable on this city? (From the served response check.) */
const estimable = (i: InterventionInfo) => Object.values(i.paths).some((p) => p.path !== "NOT_ESTIMABLE");

/** The cited fractional change at the chosen extent, for scale_down levers; null otherwise. */
function citedShare(i: InterventionInfo, s: LeverState | undefined): number | null {
  if (i.operation !== "scale_down") return null;
  return i.magnitude * (s?.extent ?? defaultExtent(i) ?? 1);
}
/** Below this cited share a change in exceedance days rounds to zero. */
const SMALL_SHARE = 0.05;

function defaultExtent(i: InterventionInfo): number | null {
  if (i.operation === "floor") return null;
  if (i.operation === "scale_down") return 1;
  return 5;
}

export function ScenarioView({ city, reaches }: { city: string; reaches: Loadable<ReachCollection> }) {
  const { scenarioReaches, toggleScenarioReach, setScenarioReaches, levers, setLever, scenario, setScenario, hovered, hover, swipe, setSwipe } = useStore();
  const catalogue = useApi(`interventions:${city}`, () => api.interventions(city));
  const [lasso, setLasso] = useState(false);
  const [running, setRunning] = useState(false);
  const [runError, setRunError] = useState<ApiError | null>(null);
  const [variable, setVariable] = useState<Variable>("turbidity_proxy");
  const fc = reaches.data;

  // Enable nothing by default; initialise extents from the operation's meaning.
  useEffect(() => {
    for (const i of catalogue.data?.interventions ?? []) if (!levers[i.id]) setLever(i.id, { enabled: false, extent: defaultExtent(i) });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [catalogue.data]);

  const enabled = (catalogue.data?.interventions ?? []).filter((i) => levers[i.id]?.enabled);
  const canRun = scenarioReaches.length > 0 && scenarioReaches.length <= 100 && enabled.length > 0 && !running;

  async function run() {
    setRunning(true);
    setRunError(null);
    try {
      const res = await api.runScenario({
        name: `${enabled.map((i) => i.name).join(" + ")} on ${scenarioReaches.length} reaches`,
        interventions: enabled.map((i) => ({ type: i.id, reach_ids: scenarioReaches, extent: i.operation === "floor" ? null : levers[i.id]?.extent ?? defaultExtent(i) })),
      });
      setScenario(res);
      setSwipe(0.5);
    } catch (e) {
      setRunError(e instanceof ApiError ? e : new ApiError(0, String(e)));
    } finally {
      setRunning(false);
    }
  }

  const result = scenario?.result ?? null;
  const byReach = useMemo(() => {
    const m = new Map<string, ReachResult>();
    for (const r of result?.reaches ?? []) if (r.variable === variable) m.set(r.reach_id, r);
    return m;
  }, [result, variable]);

  const resultSet = useMemo(() => (result ? new Set(byReach.keys()) : null), [result, byReach]);
  const before = useMemo(() => {
    const m = new Map<string, ReachValue>();
    if (!fc) return m;
    for (const f of fc.features)
      m.set(f.id, result ? { value: byReach.get(f.id)?.baseline?.exceedance_days ?? null } : { value: f.properties.p_exceed_max });
    return m;
  }, [fc, result, byReach]);
  const after = useMemo(() => {
    const m = new Map<string, ReachValue>();
    for (const [id, r] of byReach) m.set(id, { value: r.scenario?.exceedance_days ?? null });
    return m;
  }, [byReach]);

  const daysSounding = useMemo(() => (v: number) => `${v.toFixed(0)} d`, []);
  const probSounding = useMemo(() => (v: number) => fmtProb(v), []);

  // Camera sync between the two stacked maps of the swipe.
  const maps = useRef<{ a: MLMap | null; b: MLMap | null }>({ a: null, b: null });
  function link(which: "a" | "b", m: MLMap) {
    maps.current[which] = m;
    const other = () => maps.current[which === "a" ? "b" : "a"];
    let syncing = false;
    m.on("move", () => {
      const o = other();
      if (!o || syncing) return;
      syncing = true;
      o.jumpTo({ center: m.getCenter(), zoom: m.getZoom() });
      syncing = false;
    });
    const o = other();
    if (o) m.jumpTo({ center: o.getCenter(), zoom: o.getZoom() });
  }

  const wrap = useRef<HTMLDivElement>(null);
  const drag = useRef(false);
  const onDivider = (e: React.PointerEvent) => {
    if (!drag.current || !wrap.current) return;
    const r = wrap.current.getBoundingClientRect();
    setSwipe((e.clientX - r.left) / r.width);
  };

  // Driver-only reaches have no seasonal threshold, so the engine returns
  // INSUFFICIENT_EVIDENCE for them. Say so before the run, not only after it.
  const observable = useMemo(() => new Set((fc?.features ?? []).filter((f) => f.properties.observable).map((f) => f.id)), [fc]);
  const driverOnlySelected = scenarioReaches.filter((id) => !observable.has(id));
  const cityName = fc?.city ? fc.city.charAt(0).toUpperCase() + fc.city.slice(1) : city;
  const usable = (catalogue.data?.interventions ?? []).filter(estimable);
  const expect = expectation({ enabled, levers, selected: scenarioReaches.length, observableSelected: scenarioReaches.length - driverOnlySelected.length, usable, cityName });

  if (!fc) return <div className="p-6">{reaches.error ? <ErrorNote error={reaches.error} what="Reaches" /> : <Loading what="reaches" />}</div>;

  const selectedCount = scenarioReaches.length;
  const clearAll = () => {
    setScenarioReaches([]);
    setScenario(null);
  };

  return (
    <div className="flex flex-col md:min-h-0 md:flex-1">
      <p className="t-ui flex items-center gap-2.5 border-b border-hairline bg-brand-50 px-4 py-2 text-ink" role="note">
        <svg aria-hidden viewBox="0 0 16 16" width="16" height="16" className="shrink-0 text-brand">
          <circle cx="8" cy="8" r="7" fill="none" stroke="currentColor" strokeWidth="1.5" />
          <path d="M8 7v4.5M8 4.6v.1" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
        </svg>
        <span>{CAVEAT}</span>
      </p>
      <div className="flex flex-col md:min-h-0 md:flex-1 md:flex-row">
        <div className="flex flex-col md:min-h-[360px] md:flex-1">
          <div className="t-dense flex flex-wrap items-center gap-x-3 gap-y-2 border-b border-hairline bg-surface px-3 py-2">
            <Step n={1} done={selectedCount > 0}>
              Select reaches
            </Step>
            <Segmented
              dense
              label="Selection tool"
              value={lasso ? "lasso" : "click"}
              onChange={(v) => setLasso(v === "lasso")}
              options={[
                { value: "click", label: "Click" },
                { value: "lasso", label: "Lasso" },
              ]}
            />
            <span aria-live="polite">
              <span className={`t-value-sm ${selectedCount ? "text-kf" : ""}`}>{selectedCount}</span> <span className="text-muted">selected</span>
            </span>
            <button
              type="button"
              className="btn btn-sm"
              onClick={() => setScenarioReaches([...observable].slice(0, 100))}
              disabled={observable.size === 0}
              title="Only optically observable reaches have a seasonal threshold, so only they return exceedance days"
            >
              <span>
                Select all observable (<span className="t-value-sm">{Math.min(100, observable.size)}</span>)
              </span>
            </button>
            <button type="button" onClick={clearAll} disabled={!selectedCount && !result} className="btn btn-ghost btn-sm">
              {result ? "Clear and start over" : "Clear"}
            </button>
            {result && (
              <div className="flex items-center gap-2 sm:ml-auto">
                <span className="text-muted">Colour by</span>
                <Segmented
                  dense
                  label="Colour by"
                  value={variable}
                  onChange={setVariable}
                  options={(["turbidity_proxy", "ndci"] as Variable[]).map((v) => ({ value: v, label: v === "ndci" ? "NDCI" : "Turbidity" }))}
                />
              </div>
            )}
          </div>
          <div ref={wrap} className="relative h-[55vh] min-h-[320px] overflow-hidden md:h-auto md:min-h-0 md:flex-1" onPointerMove={onDivider} onPointerUp={() => (drag.current = false)}>
            <ReachMap
              label="Scenario map, baseline"
              reaches={fc}
              values={before}
              breaks={result ? DAYS_BREAKS : PROB_BREAKS}
              sounding={result ? daysSounding : probSounding}
              highlighted={scenarioReaches}
              hovered={hovered}
              visible={resultSet}
              showPins={!result}
              lasso={lasso}
              onLasso={(ids) => setScenarioReaches([...new Set([...scenarioReaches, ...ids])])}
              onClick={(id) => id && toggleScenarioReach(id)}
              onHover={hover}
              onMap={(m) => link("a", m)}
            />
            {result && (
              <>
                <div className="pointer-events-none absolute inset-0" style={{ clipPath: `inset(0 0 0 ${swipe * 100}%)` }}>
                  <div className="pointer-events-auto h-full w-full" style={{ clipPath: `inset(0 0 0 ${swipe * 100}%)` }}>
                    <ReachMap
                      label="Scenario map, with interventions"
                      reaches={fc}
                      values={after}
                      breaks={DAYS_BREAKS}
                      sounding={daysSounding}
                      highlighted={scenarioReaches}
                      visible={resultSet}
                      showPins={false}
                      showCatchments={false}
                      fit={false}
                      onClick={(id) => id && toggleScenarioReach(id)}
                      onMap={(m) => link("b", m)}
                    />
                  </div>
                </div>
                <div
                  className="absolute inset-y-0 z-20 w-4 -translate-x-1/2 cursor-ew-resize"
                  style={{ left: `${swipe * 100}%` }}
                  role="slider"
                  tabIndex={0}
                  aria-label="Swipe between baseline and scenario"
                  aria-valuemin={0}
                  aria-valuemax={100}
                  aria-valuenow={Math.round(swipe * 100)}
                  onPointerDown={(e) => {
                    drag.current = true;
                    (e.target as HTMLElement).setPointerCapture(e.pointerId);
                  }}
                  onPointerMove={onDivider}
                  onPointerUp={() => (drag.current = false)}
                  onKeyDown={(e) => {
                    if (e.key === "ArrowLeft") setSwipe(swipe - 0.05);
                    if (e.key === "ArrowRight") setSwipe(swipe + 0.05);
                  }}
                >
                  <div className="mx-auto h-full w-0.5 bg-surface shadow-[0_0_0_1px_var(--border)]" />
                  <span aria-hidden className="absolute top-1/2 left-1/2 flex h-9 w-5 -translate-x-1/2 -translate-y-1/2 items-center justify-center gap-[3px] rounded-md bg-surface shadow-[var(--shadow-float)]">
                    <span className="h-3.5 w-px bg-muted" />
                    <span className="h-3.5 w-px bg-muted" />
                  </span>
                  <span className="map-panel t-dense absolute top-2 right-4 whitespace-nowrap px-2 py-1 font-medium">baseline</span>
                  <span className="map-panel t-dense absolute top-2 left-4 whitespace-nowrap px-2 py-1 font-medium" style={{ color: SCENARIO }}>
                    with interventions
                  </span>
                </div>
              </>
            )}
            {!result && selectedCount === 0 && (
              <p className="map-panel t-ui pointer-events-none absolute top-3 right-14 left-3 z-10 px-3 py-2 text-ink sm:right-auto sm:left-1/2 sm:w-max sm:max-w-[calc(100%-8rem)] sm:-translate-x-1/2 sm:text-center">
                {lasso ? "Draw around the reaches to include." : "Click reaches to add them to the scenario, switch to Lasso, or use Select all observable."} Only coloured (observable) reaches give exceedance days; hatched ones have no threshold.
              </p>
            )}
            <div className="map-panel t-dense absolute bottom-8 left-3 z-10 px-3 py-2.5">
              <p className="mb-1.5 font-medium">{result ? `Exceedance days per year, ${VARIABLE_LABEL[variable]}` : "Peak P(exceed), current forecast"}</p>
              <div className="grid grid-cols-[auto_1fr] items-center gap-x-2.5 gap-y-1">
                {legendStops(result ? DAYS_BREAKS : PROB_BREAKS, (v) => (result ? v.toFixed(0) : v.toFixed(1))).map((s) => (
                  <div key={s.label} className="contents">
                    <Swatch color={s.color} width={s.width} />
                    <span className="t-value-sm">{s.label}</span>
                  </div>
                ))}
                <Swatch hatch />
                <span>{result ? "not in the scenario, or no estimate" : "no threshold"}</span>
              </div>
            </div>
          </div>
        </div>
        <aside className="flex w-full shrink-0 flex-col border-t border-hairline bg-surface md:min-h-0 md:w-[360px] md:border-t-0 md:border-l lg:w-[420px]" aria-label="Interventions">
          <div className="px-4 pt-4 pb-4 md:min-h-0 md:flex-1 md:overflow-y-auto">
            <Step n={2} done={enabled.length > 0}>
              <span className="t-title">Interventions</span>
            </Step>
            {catalogue.loading && <Loading what="coefficient table" className="mt-2" />}
            <ErrorNote error={catalogue.error} what="Coefficient table" />
            {catalogue.data && (
              <>
                <p className="t-dense mt-1 text-muted">
                  Coefficient table v<span className="t-value-sm">{catalogue.data.coefficient_table_version}</span>. Effect sizes come only from the cited sources below; the model never supplies them.
                </p>
                {usable.length < catalogue.data.interventions.length && (
                  <p className="t-dense mt-2 rounded-md bg-paper-alt px-2.5 py-2 text-muted">
                    <span className="t-value-sm text-ink">{usable.length}</span> of <span className="t-value-sm text-ink">{catalogue.data.interventions.length}</span> levers can be estimated on {cityName}. They are listed first.
                  </p>
                )}
                <div className="mt-3 flex flex-col gap-2">
                  {[...catalogue.data.interventions]
                    .sort((a, b) => Number(estimable(b)) - Number(estimable(a)))
                    .map((i) => (
                      <Lever key={i.id} i={i} cityName={cityName} />
                    ))}
                  {catalogue.data.not_quantified.map((n) => (
                    <div key={n.id} className="rounded-lg border border-dashed border-hairline px-3 py-3 text-muted">
                      <p className="t-ui flex items-center gap-2.5">
                        <input type="checkbox" disabled aria-label={`${n.name} (not quantified)`} />
                        <span>{n.name} — not quantified</span>
                      </p>
                      <p className="t-dense mt-1.5">{n.reason}</p>
                      <Citations list={n.citations} />
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>
          {catalogue.data && (
            <div className="sticky bottom-0 z-10 border-t border-hairline bg-surface px-4 pt-3 pb-4 shadow-[0_-4px_12px_rgba(0,0,0,0.04)]">
              <Step n={3} done={!!result}>
                Run
              </Step>
              {expect && (
                <div className={`t-dense mt-2 rounded-md border-l-[3px] px-2.5 py-2 ${expect.tone === "none" ? "border-l-[var(--severity-watch)] bg-[color-mix(in_srgb,var(--severity-watch)_10%,transparent)]" : "border-l-brand bg-brand-50"}`} role="status">
                  <p className="font-semibold text-ink">{expect.title}</p>
                  <p className="mt-0.5 text-muted">{expect.body}</p>
                </div>
              )}
              <p className="t-dense mt-2 mb-2 text-muted" aria-live="polite">
                <span className={`t-value-sm ${selectedCount ? "text-ink" : ""}`}>{selectedCount}</span> {selectedCount === 1 ? "reach" : "reaches"},{" "}
                <span className={`t-value-sm ${enabled.length ? "text-ink" : ""}`}>{enabled.length}</span> {enabled.length === 1 ? "intervention" : "interventions"}
                {enabled.length > 0 && <span className="text-ink">: {enabled.map((i) => i.name).join(", ")}</span>}
              </p>
              <button type="button" onClick={run} disabled={!canRun} aria-busy={running} className="btn btn-primary h-10 w-full">
                {running ? "Running scenario" : "Run scenario"}
              </button>
              {running && <div className="progress mt-2" aria-hidden />}
              {running && (
                <p className="t-dense mt-1.5 text-muted">
                  Re-scoring a full reference year for {selectedCount} reaches, baseline and scenario, across the coefficient range. Roughly half a second per reach.
                </p>
              )}
              {selectedCount > 100 && <p className="t-dense mt-1.5 text-ink">A scenario covers at most 100 reaches; narrow the selection.</p>}
              {!selectedCount && <p className="t-dense mt-1.5 text-muted">Select reaches on the map first.</p>}
              {selectedCount > 0 && !enabled.length && <p className="t-dense mt-1.5 text-muted">Choose at least one intervention above.</p>}
              {driverOnlySelected.length > 0 && (
                <div className="t-dense mt-1.5 text-ink">
                  <p>
                    <span className="t-value-sm">{driverOnlySelected.length}</span> of <span className="t-value-sm">{selectedCount}</span> selected{" "}
                    {driverOnlySelected.length === 1 ? "reach is" : "reaches are"} driver-only: no satellite record, so no seasonal threshold, and the scenario will return
                    insufficient evidence for {driverOnlySelected.length === 1 ? "it" : "them"}. Exceedance days need one of the{" "}
                    <span className="t-value-sm">{observable.size}</span> optically observable reaches.
                  </p>
                  {driverOnlySelected.length < selectedCount && (
                    <button type="button" className="btn btn-ghost btn-sm mt-1" onClick={() => setScenarioReaches(scenarioReaches.filter((id) => observable.has(id)))}>
                      Keep the observable reaches only
                    </button>
                  )}
                </div>
              )}
              <ErrorNote error={runError} what="Scenario" />
            </div>
          )}
        </aside>
      </div>
      {result && <Results result={result} />}
    </div>
  );
}

function Lever({ i, cityName }: { i: InterventionInfo; cityName: string }) {
  const { levers, setLever } = useStore();
  const s = levers[i.id] ?? { enabled: false, extent: defaultExtent(i) };
  const allNotEstimable = !estimable(i);
  const share = citedShare(i, s);
  const small = !allNotEstimable && share !== null && share < SMALL_SHARE;
  const step = i.operation === "scale_down" ? 0.05 : 1;
  const digits = i.magnitude < 1 ? 3 : 1;
  return (
    <div
      className={`rounded-lg border px-3 py-3 transition-colors ${
        s.enabled ? "border-brand bg-brand-50 shadow-[0_0_0_1px_var(--brand-500)]" : "border-hairline hover:border-faint"
      } ${allNotEstimable && !s.enabled ? "opacity-75" : ""}`}
    >
      <label className="flex cursor-pointer items-start gap-2.5">
        <input type="checkbox" className="mt-[2px] shrink-0" checked={s.enabled} onChange={() => setLever(i.id, { enabled: !s.enabled })} />
        <span className="min-w-0 flex-1">
          <span className="flex flex-wrap items-center justify-between gap-x-2 gap-y-1">
            <span className="t-ui font-semibold text-ink">{i.name}</span>
            {allNotEstimable ? (
              <span className="pill pill-sm pill-insufficient">No estimate on {cityName}</span>
            ) : small ? (
              <span className="pill pill-sm pill-watch">Small cited effect</span>
            ) : (
              <span className="pill pill-sm pill-brand">Estimable</span>
            )}
          </span>
          <span className="t-dense mt-0.5 block text-muted">
            {humanize(i.direction)}s {i.feature.replaceAll("_", " ")}: cited magnitude <span className="t-value-sm text-ink">{fmtNum(i.magnitude, digits)}</span> (range{" "}
            <span className="t-value-sm">
              {fmtNum(i.uncertainty_range[0], digits)}–{fmtNum(i.uncertainty_range[1], digits)}
            </span>
            ), {i.effect_unit}.
          </span>
        </span>
      </label>
      <div className="pl-6">
        {i.extent_unit && (
          <div className="mt-3">
            <label className="t-dense flex items-center gap-3">
              <span className="shrink-0 text-muted">Extent</span>
              <input
                type="range"
                min={step}
                max={i.extent_max ?? 1}
                step={step}
                value={s.extent ?? defaultExtent(i) ?? 0}
                onChange={(e) => setLever(i.id, { extent: Number(e.target.value) })}
                disabled={!s.enabled}
                className="min-w-0 flex-1 disabled:opacity-50"
                aria-valuetext={`${s.extent} ${i.extent_unit}`}
              />
              <span className={`t-value-sm w-12 text-right ${s.enabled ? "text-ink" : "text-muted"}`}>{i.operation === "scale_down" ? fmtNum(s.extent, 2) : fmtNum(s.extent, 0)}</span>
            </label>
            <p className="t-dense mt-0.5 text-muted">
              {i.extent_unit}
              {i.operation === "subtract_treated_share" ? "; cannot exceed a reach's current imperviousness" : i.operation === "scale_down" ? "; 1 = the coverage of the cited study" : ""}.
            </p>
          </div>
        )}
        {i.operation === "floor" && <p className="t-dense mt-2 text-muted">Fixed at the cited design width; no extent to choose.</p>}
        <dl className="t-dense mt-3 grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5">
          {Object.entries(i.paths).map(([v, p]) => (
            <div key={v} className="contents">
              <dt className="text-muted">{v === "ndci" ? "NDCI" : "Turbidity"}</dt>
              <dd className={`flex items-center gap-1.5 ${p.path === "NOT_ESTIMABLE" ? "text-muted" : "text-ink"}`}>
                {p.path === "NOT_ESTIMABLE" && <span aria-hidden className="hatch inline-block h-2 w-3" />}
                {PATH_TEXT[p.path]}
              </dd>
            </div>
          ))}
        </dl>
        <details className="t-dense mt-1.5">
          <summary className="inline-flex cursor-pointer items-center gap-1 text-muted hover:text-ink">
            <Chevron />
            Why each path
          </summary>
          <div className="mt-1 flex flex-col gap-1 border-l border-hairline pl-2 text-muted">
            {Object.entries(i.paths).map(([v, p]) => (
              <p key={v} className="break-words">
                <span className="text-ink">{v === "ndci" ? "NDCI" : "Turbidity"}:</span> {p.reason}
              </p>
            ))}
          </div>
        </details>
        {allNotEstimable && <p className="t-dense mt-2 text-ink">Selecting it will not change the result on {cityName}: the model's response to it failed the check and no cited direct effect exists.</p>}
        {small && (
          <p className="t-dense mt-2 text-ink">
            The cited effect is <span className="t-value-sm">{fmtNum(share! * 100, 1)} %</span> at this extent, so expect a change close to 0 days.
          </p>
        )}
        <details className="t-dense mt-1.5">
          <summary className="inline-flex cursor-pointer items-center gap-1 text-muted hover:text-ink">
            <Chevron />
            Sources and cost
          </summary>
          <p className="mt-1.5 text-muted">Effect size from</p>
          <Citations list={i.citations} />
          <p className="mt-2 text-muted">
            Cost <span className="t-value-sm text-ink">{i.cost_per_unit.currency} {fmtNum(i.cost_per_unit.low, 0)}–{fmtNum(i.cost_per_unit.high, 0)}</span> per {i.cost_per_unit.unit} ({i.cost_per_unit.price_basis}).
          </p>
          <Citations list={i.cost_per_unit.citation} />
        </details>
      </div>
    </div>
  );
}

/** Citations are a feature: shown as text, never behind an icon or tooltip. */
function Citations({ list }: { list: Citation[] }) {
  return (
    <ul className="t-dense mt-1 flex flex-col gap-1.5">
      {list.map((c, k) => (
        <li key={k} className="border-l border-hairline pl-2 break-words">
          {c.authors} ({c.year}). {c.title}. <span className="italic">{c.source}</span>.{" "}
          {c.doi ? (
            <a className="link" href={`https://doi.org/${c.doi}`} target="_blank" rel="noreferrer">
              doi:{c.doi}
            </a>
          ) : c.url ? (
            <a className="link break-all" href={c.url} target="_blank" rel="noreferrer">
              {c.url}
            </a>
          ) : null}
          <span className="block text-muted">{c.locator}</span>
        </li>
      ))}
    </ul>
  );
}

function Results({ result }: { result: ScenarioResult }) {
  const [settled, setSettled] = useState(false);
  useEffect(() => {
    setSettled(false);
    const t = requestAnimationFrame(() => requestAnimationFrame(() => setSettled(true)));
    return () => cancelAnimationFrame(t);
  }, [result]);
  const ok = result.reaches.filter((r) => r.baseline && r.scenario);
  const other = result.reaches.filter((r) => !(r.baseline && r.scenario));
  const max = Math.max(1, ...ok.flatMap((r) => [r.baseline!.interval.high, r.scenario!.interval.high]));
  const costs = result.costs;
  const effectCites = result.citations.filter((c) => c.role !== "cost");
  return (
    <section className="shrink-0 border-t border-hairline bg-surface px-4 py-4 md:max-h-[42vh] md:overflow-y-auto" aria-label="Scenario results">
      <div className="flex flex-wrap items-baseline gap-x-4">
        <h2 className="t-title">Results</h2>
        <p className="t-dense text-muted">
          {result.label}, {result.units}; reference year {fmtDay(result.reference_start, true)} – {fmtDay(result.reference_end, true)}; {Math.round(result.ci_level * 100)}% intervals, widened ×{result.widening_factor} on the scenario; model{" "}
          <span className="t-value-sm">{result.model_version}</span>
        </p>
      </div>
      <Headline result={result} ok={ok} />
      <div className="grid gap-6 lg:grid-cols-[1.4fr_1fr]">
        <div>
          <div className={`overflow-x-auto ${ok.length === 0 ? "hidden" : ""}`}>
          <table className="mt-2 w-full min-w-[560px] t-dense">
            <thead>
              <tr className="hairline-b t-eyebrow text-left">
                <th className="py-1 font-normal">Reach</th>
                <th className="py-1 font-normal">Variable</th>
                <th className="w-[38%] py-1 font-normal">Baseline → with interventions</th>
                <th className="py-1 text-right font-normal">Days per year</th>
                <th className="py-1 text-right font-normal">Change</th>
              </tr>
            </thead>
            <tbody>
              {ok.map((r) => (
                <tr key={`${r.reach_id}-${r.variable}`} className="row-hover hairline-b align-middle">
                  <td className="t-value-sm py-1.5">{r.reach_id}</td>
                  <td className="py-1.5 text-muted">{r.variable === "ndci" ? "NDCI" : "turbidity"}</td>
                  <td className="py-1.5">
                    <div className="relative h-4" aria-hidden>
                      <div className="absolute top-0 h-1.5 rounded-full bg-heavy" style={{ width: `${(r.baseline!.exceedance_days / max) * 100}%` }} />
                      <div className="bar-descend absolute top-2 h-1.5 rounded-full" style={{ background: SCENARIO, width: `${((settled ? r.scenario! : r.baseline!).exceedance_days / max) * 100}%` }} />
                    </div>
                  </td>
                  <td className="py-1.5 text-right">
                    <span className="t-value-sm">{fmtNum(r.baseline!.exceedance_days, 1)} → {fmtNum(r.scenario!.exceedance_days, 1)}</span>
                    <span className="block t-value-sm text-muted">
                      [{fmtNum(r.baseline!.interval.low, 0)}–{fmtNum(r.baseline!.interval.high, 0)}] → [{fmtNum(r.scenario!.interval.low, 0)}–{fmtNum(r.scenario!.interval.high, 0)}]
                    </span>
                  </td>
                  <td className="py-1.5 text-right t-value-sm">
                    {r.delta_days !== null ? `${r.delta_days > 0 ? "+" : ""}${fmtNum(r.delta_days, 1)}` : "—"}
                    {!r.complete && <span className="block text-muted">some levers not estimated</span>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          </div>
          {ok.some((r) => r.delta_days !== null && Math.abs(r.delta_days) < 0.05) && (
            <p className="t-dense mt-2">
              A change of <span className="t-value-sm">0.0</span> is the estimate, not a failure: the cited effect sizes here are small (detention basins reduce peak flow by about 0.3 % in the cited study), or the lever could not be estimated on this reach.
            </p>
          )}
          {other.length > 0 && (
            <details className="mt-2 t-dense" open={ok.length === 0}>
              <summary className="inline-flex cursor-pointer items-center gap-1 text-muted hover:text-ink">
                <Chevron />
                {other.length} reach-variables without an estimate
              </summary>
              <ul className="mt-1">
                {other.map((r) => (
                  <li key={`${r.reach_id}-${r.variable}`} className="hatch-border hairline-b py-1 pl-2">
                    {r.reach_id} {r.variable === "ndci" ? "NDCI" : "turbidity"}: {humanize(r.status)}. <span className="text-muted">{r.reason}</span>
                  </li>
                ))}
              </ul>
            </details>
          )}
          <LeverFlags result={result} />
          <p className="t-dense mt-2 text-muted">{result.spatial_scope}</p>
          <p className="t-dense mt-1 text-muted">{result.interval_method}</p>
        </div>
        <div>
          <h3 className="t-eyebrow mt-2 mb-1">Cost</h3>
          <table className="w-full t-dense">
            <tbody>
              {costs.map((c, k) => (
                <tr key={k} className="hairline-b align-top">
                  <td className="py-1">
                    {c.reach_id} <span className="text-muted">{c.intervention_id.replaceAll("_", " ")}</span>
                  </td>
                  <td className="py-1 text-right">
                    {c.total_low !== null && c.total_high !== null ? (
                      <span className="t-value-sm">
                        {fmtMoney(c.total_low, c.currency)}–{fmtMoney(c.total_high, c.currency).replace(`${c.currency} `, "")}
                      </span>
                    ) : (
                      <span className="text-muted">
                        <span className="t-value-sm text-ink">{c.currency} {fmtNum(c.unit_cost_low, 0)}–{fmtNum(c.unit_cost_high, 0)}</span> per {c.unit}; no total: {c.reason}
                      </span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="t-dense mt-1 text-muted">In each source's currency and price year; nothing converted.</p>
          <h3 className="t-eyebrow mt-4 mb-1">Coefficients used</h3>
          <ul className="t-dense mt-1 flex flex-col gap-1.5">
            {result.levers
              .filter((l, k, arr) => arr.findIndex((x) => x.intervention_id === l.intervention_id) === k)
              .map((l) => (
                <li key={l.intervention_id}>
                  <span className="text-ink">{l.name}</span>: magnitude <span className="t-value-sm">{l.magnitude}</span> (range <span className="t-value-sm">{l.uncertainty_range[0]}–{l.uncertainty_range[1]}</span>), extent{" "}
                  <span className="t-value-sm">{l.extent ?? "—"}</span>
                  <ul>
                    {effectCites
                      .filter((c) => c.intervention_id === l.intervention_id)
                      .map((c, k) => (
                        <li key={k} className="border-l border-hairline pl-2 text-muted">
                          {c.authors} ({c.year}), {c.source}. {c.doi ? `doi:${c.doi}` : c.url} — {c.locator}
                        </li>
                      ))}
                  </ul>
                </li>
              ))}
          </ul>
        </div>
      </div>
    </section>
  );
}

function LeverFlags({ result }: { result: ScenarioResult }) {
  const seen = new Map<string, string>();
  for (const l of result.levers) seen.set(`${l.intervention_id}|${l.variable}`, `${l.name}, ${l.variable === "ndci" ? "NDCI" : "turbidity"}: ${PATH_TEXT[l.path]} — ${l.reason}`);
  const flags = new Set<string>();
  for (const r of result.reaches) for (const l of r.levers) for (const d of l.detail) flags.add(d);
  return (
    <div className="mt-3 t-dense">
      <p className="text-ink">How each lever was applied</p>
      <ul className="text-muted">
        {[...seen.values()].map((t) => (
          <li key={t}>{t}</li>
        ))}
      </ul>
      {flags.size > 0 && (
        <details className="mt-1">
          <summary className="inline-flex cursor-pointer items-center gap-1 text-muted hover:text-ink">
            <Chevron />
            {flags.size} support warnings
          </summary>
          <ul className="text-muted">
            {[...flags].map((f) => (
              <li key={f}>{f}</li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}

/** A numbered step label for the three-step workbench flow. */
function Step({ n, done, children }: { n: number; done: boolean; children: React.ReactNode }) {
  return (
    <span className="flex items-center gap-2">
      <span
        aria-hidden
        className={`inline-grid h-5 w-5 shrink-0 place-items-center rounded-full text-[11px] leading-none font-semibold ${
          done ? "bg-brand text-[var(--on-brand)]" : "bg-paper-alt text-muted shadow-[inset_0_0_0_1px_var(--border)]"
        }`}
      >
        {n}
      </span>
      <span className="font-medium text-ink">{children}</span>
    </span>
  );
}

interface Expectation {
  tone: "none" | "info";
  title: string;
  body: string;
}

/** What the run will return, said before it runs. Read off the served response-check paths
 * and the cited magnitudes; nothing is estimated here. */
function expectation(a: {
  enabled: InterventionInfo[];
  levers: Record<string, LeverState>;
  selected: number;
  observableSelected: number;
  usable: InterventionInfo[];
  cityName: string;
}): Expectation | null {
  if (!a.selected || !a.enabled.length) return null;
  const others = a.usable.filter((u) => !a.enabled.includes(u)).map((u) => u.name);
  if (a.observableSelected === 0)
    return {
      tone: "none",
      title: "No exceedance days will come back",
      body: "None of the selected reaches is optically observable, so none has a seasonal threshold to count days against. Use Select all observable.",
    };
  const est = a.enabled.filter(estimable);
  if (!est.length)
    return {
      tone: "none",
      title: "Baseline and scenario will be the same",
      body: `${a.enabled.map((i) => i.name).join(", ")} cannot be estimated on ${a.cityName}: the model failed the response check and the coefficient table has no cited direct effect.${
        others.length ? ` Levers that can be estimated here: ${others.join(", ")}.` : ""
      }`,
    };
  const small = est.filter((i) => {
    const sh = citedShare(i, a.levers[i.id]);
    return sh !== null && sh < SMALL_SHARE;
  });
  if (small.length === est.length)
    return {
      tone: "none",
      title: "Expect a change close to 0 days",
      body: `The cited effect of ${small.map((i) => `${i.name} (${fmtNum((citedShare(i, a.levers[i.id]) ?? 0) * 100, 1)} %)`).join(", ")} is small. A near-zero change is the honest estimate, not a failure.`,
    };
  return null;
}

/** One line that says what changed - or plainly why nothing did. */
function Headline({ result, ok }: { result: ScenarioResult; ok: ReachResult[] }) {
  const base = ok.reduce((t, r) => t + r.baseline!.exceedance_days, 0);
  const scen = ok.reduce((t, r) => t + r.scenario!.exceedance_days, 0);
  const moved = ok.some((r) => r.delta_days !== null && Math.abs(r.delta_days) >= 0.05);
  if (ok.length && moved)
    return (
      <div className="card mt-3 mb-4 flex flex-wrap items-baseline gap-x-3 gap-y-1 px-4 py-3">
        <span className="t-reading">
          {fmtNum(base, 1)} → <span style={{ color: SCENARIO }}>{fmtNum(scen, 1)}</span>
        </span>
        <span className="t-ui text-muted">
          exceedance days per year, summed over {ok.length} reach-{ok.length === 1 ? "variable" : "variables"} with an estimate
        </span>
      </div>
    );
  const notEst = [...new Set(result.levers.filter((l) => l.path === "NOT_ESTIMABLE").map((l) => l.name))];
  const est = [...new Set(result.levers.filter((l) => l.path !== "NOT_ESTIMABLE").map((l) => l.name))];
  const noThreshold = result.reaches.filter((r) => r.status === "INSUFFICIENT_EVIDENCE").length;
  const reasons: string[] = [];
  if (notEst.length) reasons.push(`${notEst.join(", ")}: not estimable here, for at least one variable (the model's response failed the check and no cited direct effect exists).`);
  if (noThreshold) reasons.push(`${noThreshold} reach-${noThreshold === 1 ? "variable has" : "variables have"} no seasonal threshold (driver-only reaches), so no days can be counted.`);
  if (ok.length && est.length) reasons.push(`${est.join(", ")}: estimated, but the cited effect is small, so the change rounds to 0 days.`);
  return (
    <div className="mt-3 mb-4 rounded-lg border-l-[3px] border-l-[var(--severity-watch)] bg-[color-mix(in_srgb,var(--severity-watch)_10%,transparent)] px-4 py-3" role="status">
      <p className="t-ui font-semibold text-ink">Why baseline and scenario look the same</p>
      <ul className="t-dense mt-1 list-disc pl-4 text-muted">
        {reasons.map((r) => (
          <li key={r}>{r}</li>
        ))}
      </ul>
      <p className="t-dense mt-1.5 text-muted">This is the estimate, not an error: no effect size is invented where the evidence does not support one.</p>
    </div>
  );
}
