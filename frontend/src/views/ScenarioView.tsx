import { useEffect, useMemo, useState } from "react";
import { api, ApiError, SNAPSHOT, type ScenarioPreset } from "../api/client";
import type { Citation, InterventionInfo, ReachCollection, ReachResult, ScenarioResult, Variable } from "../api/types";
import { Card, Chevron, ErrorNote, Loading, Segmented, Swatch } from "../components/bits";
import { Splitter, usePanelSize } from "../components/Splitter";
import { useShortcut } from "../lib/shortcuts";
import type { LeverState } from "../store";
import { fmtDay, fmtMoney, fmtNum, fmtProb, humanize } from "../lib/format";
import { CHANGE_BREAKS, changeColor, changeLegend, DAYS_BREAKS, legendStops, NEGLIGIBLE_DAYS, PROB_BREAKS } from "../lib/ramp";
import { useApi, type Loadable } from "../lib/useApi";
import { ReachMap, type ReachValue } from "../map/ReachMap";
import { useStore } from "../store";

const CAVEAT = "Planning estimate. Intervention effect sizes from cited literature applied to a statistical model; not a causal experiment.";

const PATH_TEXT: Record<string, string> = {
  MODEL_PERTURBATION: "through the model",
  LITERATURE_DIRECT: "cited direct effect",
  NOT_ESTIMABLE: "not estimable here",
};

type MapMode = "baseline" | "scenario" | "change";

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

const varShort = (v: string) => (v === "ndci" ? "NDCI" : "Turbidity");
/** Signed change in days; below the display precision it is plainly 0.0, not "+0.0". */
const signedDays = (v: number) => (Math.abs(v) < 0.05 ? "0.0 d" : `${v > 0 ? "+" : "−"}${fmtNum(Math.abs(v), 1)} d`);

/** What one reach-variable result says, in the words the results table already uses. */
function verdict(r: ReachResult): { tone: "none" | "small" | "moved"; text: string } {
  if (!(r.baseline && r.scenario) || r.delta_days === null)
    return { tone: "none", text: `${humanize(r.status)}${r.reason ? `: ${r.reason}` : ""}` };
  if (Math.abs(r.delta_days) < NEGLIGIBLE_DAYS) {
    const allNot = r.levers.length > 0 && r.levers.every((l) => l.path === "NOT_ESTIMABLE");
    return allNot
      ? { tone: "small", text: "Not estimable here: the model failed the response check and no cited direct effect exists." }
      : { tone: "small", text: "Cited effect too small to register: the change rounds to 0 days." };
  }
  return { tone: "moved", text: r.delta_days < 0 ? "Fewer exceedance days with the interventions." : "More exceedance days with the interventions." };
}

export function ScenarioView({ city, reaches }: { city: string; reaches: Loadable<ReachCollection> }) {
  const { scenarioReaches, toggleScenarioReach, setScenarioReaches, levers, setLever, scenario, setScenario, hovered, hover } = useStore();
  const catalogue = useApi(`interventions:${city}`, () => api.interventions(city));
  const [lasso, setLasso] = useState(false);
  const [running, setRunning] = useState(false);
  const [runError, setRunError] = useState<ApiError | null>(null);
  const [variable, setVariable] = useState<Variable>("turbidity_proxy");
  const [mode, setMode] = useState<MapMode>("change");
  const [panelW, setPanelW] = usePanelSize("scenario.panel", null);
  const [rowH, setRowH] = usePanelSize("scenario.height", null);
  useShortcut("b", () => setPanelW(panelW === 0 ? null : 0), { ctrl: true });
  const vw = typeof window !== "undefined" ? window.innerWidth : 1440;
  const vh = typeof window !== "undefined" ? window.innerHeight : 900;
  const pw = panelW ?? (vw >= 1280 ? 420 : 380);
  const rh = rowH ?? Math.max(520, vh - 190);
  const fc = reaches.data;

  // Enable nothing by default; initialise extents from the operation's meaning.
  useEffect(() => {
    for (const i of catalogue.data?.interventions ?? []) if (!levers[i.id]) setLever(i.id, { enabled: false, extent: defaultExtent(i) });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [catalogue.data]);

  const enabled = (catalogue.data?.interventions ?? []).filter((i) => levers[i.id]?.enabled);
  const presets = useApi(SNAPSHOT ? `presets:${city}` : null, () => api.presets(city));

  /** Snapshot build: put a precomputed scenario's exact selection and levers on screen, then
   * replay its stored engine output. */
  async function loadPreset(p: ScenarioPreset) {
    setScenarioReaches(p.interventions[0]?.reach_ids ?? []);
    for (const i of catalogue.data?.interventions ?? []) {
      const hit = p.interventions.find((x) => x.type === i.id);
      setLever(i.id, hit ? { enabled: true, extent: hit.extent ?? defaultExtent(i) } : { enabled: false });
    }
    setRunning(true);
    setRunError(null);
    try {
      setScenario(await api.runScenario({ name: p.name, interventions: p.interventions }));
      setMode("change");
    } catch (e) {
      setRunError(e instanceof ApiError ? e : new ApiError(0, String(e)));
    } finally {
      setRunning(false);
    }
  }
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
      // Change is the default view after a run: a difference, not two maps to compare by eye.
      setMode("change");
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
  const values = useMemo(() => {
    const m = new Map<string, ReachValue>();
    if (!fc) return m;
    for (const f of fc.features) {
      if (!result) {
        m.set(f.id, { value: f.properties.p_exceed_max });
        continue;
      }
      const r = byReach.get(f.id);
      const v = mode === "baseline" ? r?.baseline?.exceedance_days : mode === "scenario" ? r?.scenario?.exceedance_days : r?.baseline && r?.scenario ? r.delta_days : null;
      m.set(f.id, { value: v ?? null });
    }
    return m;
  }, [fc, result, byReach, mode]);

  const sounding = useMemo(
    () => (!result ? (v: number) => fmtProb(v) : mode === "change" ? (v: number) => signedDays(v) : (v: number) => `${v.toFixed(0)} d`),
    [result, mode],
  );

  // Driver-only reaches have no seasonal threshold, so the engine returns
  // INSUFFICIENT_EVIDENCE for them. Say so before the run, not only after it.
  const observable = useMemo(() => new Set((fc?.features ?? []).filter((f) => f.properties.observable).map((f) => f.id)), [fc]);
  const driverOnlySelected = scenarioReaches.filter((id) => !observable.has(id));
  const cityName = fc?.city ? fc.city.charAt(0).toUpperCase() + fc.city.slice(1) : city;
  const usable = (catalogue.data?.interventions ?? []).filter(estimable);
  const expect = expectation({ enabled, levers, selected: scenarioReaches.length, observableSelected: scenarioReaches.length - driverOnlySelected.length, usable, cityName });

  if (!fc) return <div className="page">{reaches.error ? <ErrorNote error={reaches.error} what="Reaches" /> : <Loading what="reaches" />}</div>;

  const selectedCount = scenarioReaches.length;
  const clearAll = () => {
    setScenarioReaches([]);
    setScenario(null);
  };
  const step = result ? 3 : enabled.length && selectedCount ? 3 : selectedCount ? 2 : 1;
  const hoveredResult = hovered && result ? byReach.get(hovered) : undefined;
  const summary = result ? mapSummary([...byReach.values()]) : null;

  return (
    <div className="page min-h-0 flex-1 overflow-y-auto !pt-5">
      <div className="card flex items-start gap-3 px-4 py-3 shadow-[inset_3px_0_0_var(--brand-500),var(--shadow-sm)]" role="note">
        <svg aria-hidden viewBox="0 0 16 16" width="18" height="18" className="mt-px shrink-0 text-brand-300">
          <circle cx="8" cy="8" r="7" fill="none" stroke="currentColor" strokeWidth="1.5" />
          <path d="M8 7v4.5M8 4.6v.1" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
        </svg>
        <p className="t-ui">
          <span className="font-semibold text-ink">Scenario workbench.</span> <span className="text-muted">{CAVEAT}</span>
        </p>
      </div>

      <div className="mt-4 flex flex-col gap-4 lg:flex-row lg:gap-0" style={{ ["--panel-w" as string]: `${pw}px`, ["--row-h" as string]: `${rh}px` }}>
        {/* ---- the map card: toolbar strip + map ---- */}
        <div className="map-frame flex h-[calc(100vh-190px)] min-h-[520px] min-w-0 flex-col lg:h-(--row-h) lg:min-h-[360px] lg:flex-1">
          <div className="flex flex-wrap items-center gap-x-3 gap-y-2 border-b border-hairline bg-surface px-3 py-2.5">
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
            <span aria-live="polite" className="text-[13px]">
              <span className={`t-value-sm ${selectedCount ? "text-brand-text" : "text-muted"}`}>{selectedCount}</span> <span className="text-muted">selected</span>
            </span>
            <button
              type="button"
              className="btn btn-sm"
              onClick={() => setScenarioReaches([...observable].slice(0, 100))}
              disabled={observable.size === 0}
              title="Only optically observable reaches have a seasonal threshold, so only they return exceedance days"
            >
              Select all observable ({Math.min(100, observable.size)})
            </button>
            <button type="button" onClick={clearAll} disabled={!selectedCount && !result} className="text-link text-link-danger disabled:opacity-40">
              {result ? "Clear and start over" : "Clear"}
            </button>
            {result && (
              <div className="flex flex-wrap items-center gap-2 min-[1400px]:ml-auto">
                <Segmented dense label="Variable" value={variable} onChange={setVariable} options={(["turbidity_proxy", "ndci"] as Variable[]).map((v) => ({ value: v, label: varShort(v) }))} />
                <Segmented
                  dense
                  label="Map shows"
                  value={mode}
                  onChange={setMode}
                  options={[
                    { value: "baseline", label: "Baseline" },
                    { value: "scenario", label: "With interventions" },
                    { value: "change", label: "Change" },
                  ]}
                />
              </div>
            )}
          </div>
          <div className="relative min-h-0 flex-1">
            <ReachMap
              label={`Scenario map, ${result ? (mode === "change" ? "change" : mode === "baseline" ? "baseline" : "with interventions") : "selection"}`}
              reaches={fc}
              values={values}
              breaks={!result ? PROB_BREAKS : mode === "change" ? CHANGE_BREAKS : DAYS_BREAKS}
              scheme={result && mode === "change" ? "change" : "exceed"}
              sounding={sounding}
              // After a run the result's own colours carry the map; the selection casing
              // would drown the neutral "negligible" line.
              highlighted={result ? undefined : scenarioReaches}
              hovered={hovered}
              visible={resultSet}
              showPins={!result}
              lasso={lasso}
              onLasso={(ids) => setScenarioReaches([...new Set([...scenarioReaches, ...ids])])}
              onClick={(id) => id && toggleScenarioReach(id)}
              onHover={hover}
            />
            {!result && selectedCount === 0 && (
              <p className="map-panel pointer-events-none absolute top-3 left-3 z-10 max-w-[min(440px,calc(100%-5rem))] px-4 py-3 text-[13px] text-ink">
                <span className="font-semibold">Step 1.</span> {lasso ? "Draw around the reaches to include." : "Click reaches to add them, switch to Lasso, or use Select all observable."}{" "}
                <span className="hidden text-muted sm:inline">Only coloured (observable) reaches return exceedance days; hatched ones have no threshold.</span>
              </p>
            )}
            {hoveredResult && <ChangeCard r={hoveredResult} />}
            {summary && <MapNotice s={summary} mode={mode} />}
            <MapLegend result={!!result} mode={mode} variable={variable} />
          </div>
        </div>

        <Splitter axis="x" reverse collapsible size={pw} onSize={setPanelW} min={320} max={Math.max(320, vw - 560)} label="Interventions panel" className="hidden lg:flex" />
        {/* ---- the steps panel ---- */}
        <aside
          className={`card flex h-[calc(100vh-190px)] min-h-[520px] flex-col overflow-hidden lg:h-(--row-h) lg:min-h-[360px] lg:w-(--panel-w) lg:shrink-0 ${pw === 0 ? "lg:hidden" : ""}`}
          aria-label="Interventions"
        >
          <div className="border-b border-hairline px-5 pt-5 pb-4">
            <Stepper step={step} done={[selectedCount > 0, enabled.length > 0, !!result]} />
          </div>
          <div className="min-h-0 flex-1 overflow-y-auto px-5 pt-4 pb-4">
            <h2 className="t-title">Interventions</h2>
            {catalogue.loading && <Loading what="coefficient table" className="mt-2" />}
            <ErrorNote error={catalogue.error} what="Coefficient table" />
            {catalogue.data && (
              <>
                <p className="t-dense mt-1 text-muted">
                  Coefficient table v<span className="font-mono">{catalogue.data.coefficient_table_version}</span>. Effect sizes come only from the cited sources; the model never supplies them.
                </p>
                {usable.length < catalogue.data.interventions.length && (
                  <p className="t-dense mt-3 rounded-md bg-raised px-3 py-2 text-muted">
                    <span className="font-semibold text-ink">{usable.length}</span> of <span className="font-semibold text-ink">{catalogue.data.interventions.length}</span> levers can be estimated on {cityName}. They are listed first.
                  </p>
                )}
                <div className="mt-4 flex flex-col gap-3">
                  {[...catalogue.data.interventions]
                    .sort((a, b) => Number(estimable(b)) - Number(estimable(a)))
                    .map((i) => (
                      <Lever key={i.id} i={i} cityName={cityName} />
                    ))}
                  {catalogue.data.not_quantified.map((n) => (
                    <div key={n.id} className="rounded-[var(--radius-md)] border border-dashed border-hairline-strong px-4 py-3 text-muted">
                      <p className="flex items-center gap-2.5 text-[13px]">
                        <input type="checkbox" disabled aria-label={`${n.name} (not quantified)`} />
                        <span className="font-semibold text-ink">{n.name}</span>
                        <span className="pill pill-sm pill-neutral ml-auto">Not quantified</span>
                      </p>
                      <p className="t-dense mt-2">{n.reason}</p>
                      <Citations list={n.citations} />
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>
          {catalogue.data && (
            <div className="border-t border-hairline bg-surface px-5 pt-4 pb-5 shadow-[0_-8px_16px_rgba(0,0,0,0.12)]">
              {expect && (
                <div
                  className={`mb-3 rounded-md border-l-4 px-3 py-2.5 text-[12px] leading-4 ${expect.tone === "none" ? "border-l-[var(--severity-watch)] bg-[var(--severity-watch-bg)]" : "border-l-brand bg-brand-50"}`}
                  role="status"
                >
                  <p className="font-semibold text-ink">{expect.title}</p>
                  <p className="mt-1 text-muted">{expect.body}</p>
                </div>
              )}
              {SNAPSHOT && (
                <div className="mb-3 rounded-md bg-raised px-3 py-2.5">
                  <p className="t-dense text-ink">
                    <span className="font-semibold">Hosted snapshot:</span> no live model here. These scenarios were run by the real engine when the snapshot was built; the full system runs any selection.
                  </p>
                  <div className="mt-2 flex flex-wrap gap-1.5">
                    {(presets.data ?? []).map((p) => (
                      <button key={p.id} type="button" className="btn h-8 px-2.5 text-[12px]" disabled={running} onClick={() => loadPreset(p)}>
                        {p.name}
                      </button>
                    ))}
                  </div>
                </div>
              )}
              <p className="t-dense mb-2 text-muted" aria-live="polite">
                <span className={`font-semibold ${selectedCount ? "text-ink" : ""}`}>{selectedCount}</span> {selectedCount === 1 ? "reach" : "reaches"},{" "}
                <span className={`font-semibold ${enabled.length ? "text-ink" : ""}`}>{enabled.length}</span> {enabled.length === 1 ? "intervention" : "interventions"}
                {enabled.length > 0 && <span className="text-ink">: {enabled.map((i) => i.name).join(", ")}</span>}
              </p>
              <button
                type="button"
                onClick={run}
                disabled={!canRun}
                aria-busy={running}
                className="btn btn-primary h-11 w-full text-[14px]"
                title={!selectedCount ? "Select reaches on the map first" : !enabled.length ? "Choose at least one intervention" : undefined}
              >
                {running ? "Running scenario…" : result ? "Run again" : "Run scenario"}
              </button>
              {running && <div className="progress mt-2" aria-hidden />}
              {running && <p className="t-dense mt-1.5 text-muted">Re-scoring a full reference year for {selectedCount} reaches, baseline and scenario, across the coefficient range. Roughly half a second per reach.</p>}
              {selectedCount > 100 && <p className="t-dense mt-1.5 text-ink">A scenario covers at most 100 reaches; narrow the selection.</p>}
              {!selectedCount && <p className="t-dense mt-1.5 text-muted">Select reaches on the map first.</p>}
              {selectedCount > 0 && !enabled.length && <p className="t-dense mt-1.5 text-muted">Choose at least one intervention above.</p>}
              {driverOnlySelected.length > 0 && (
                <div className="t-dense mt-2 text-ink">
                  <p>
                    <span className="font-semibold">{driverOnlySelected.length}</span> of {selectedCount} selected {driverOnlySelected.length === 1 ? "reach is" : "reaches are"} driver-only: no seasonal threshold, so the
                    scenario returns insufficient evidence for {driverOnlySelected.length === 1 ? "it" : "them"}.
                  </p>
                  {driverOnlySelected.length < selectedCount && (
                    <button type="button" className="text-link mt-1 !text-brand-text" onClick={() => setScenarioReaches(scenarioReaches.filter((id) => observable.has(id)))}>
                      Keep the {observable.size ? "observable" : ""} reaches only
                    </button>
                  )}
                </div>
              )}
              <ErrorNote error={runError} what="Scenario" />
            </div>
          )}
        </aside>
      </div>
      <Splitter axis="y" size={rh} onSize={setRowH} min={360} max={Math.max(400, vh * 1.5)} label="Map and panel height" className="mt-1 hidden lg:flex" />
      {result && <Results result={result} />}
    </div>
  );
}

interface Summary {
  total: number;
  better: number;
  worse: number;
  negligible: number;
  none: number;
}
function mapSummary(rs: ReachResult[]): Summary {
  const s: Summary = { total: rs.length, better: 0, worse: 0, negligible: 0, none: 0 };
  for (const r of rs) {
    if (!(r.baseline && r.scenario) || r.delta_days === null) s.none++;
    else if (Math.abs(r.delta_days) < NEGLIGIBLE_DAYS) s.negligible++;
    else if (r.delta_days < 0) s.better++;
    else s.worse++;
  }
  return s;
}

/** On the map itself: the run happened, and here is what it found - even when it is ~0. */
function MapNotice({ s, mode }: { s: Summary; mode: MapMode }) {
  const allFlat = s.better + s.worse === 0;
  return (
    <div className="map-panel absolute top-3 left-3 z-10 max-w-[min(420px,calc(100%-5rem))] px-4 py-3" role="status">
      <p className="t-eyebrow">Scenario ran on {s.total} reaches</p>
      <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[13px]">
        <span className="flex items-center gap-1.5">
          <span aria-hidden className="h-2 w-2 rounded-full" style={{ background: "var(--change-improve)" }} />
          <span className="t-value-sm text-ink">{s.better}</span> <span className="text-muted">better</span>
        </span>
        <span className="flex items-center gap-1.5">
          <span aria-hidden className="h-2 w-2 rounded-full" style={{ background: "var(--change-worsen)" }} />
          <span className="t-value-sm text-ink">{s.worse}</span> <span className="text-muted">worse</span>
        </span>
        <span className="flex items-center gap-1.5">
          <span aria-hidden className="h-2 w-2 rounded-full" style={{ background: "var(--change-neutral)" }} />
          <span className="t-value-sm text-ink">{s.negligible}</span> <span className="text-muted">Δ≈0</span>
        </span>
        <span className="flex items-center gap-1.5">
          <span aria-hidden className="hatch h-2 w-3 rounded-[2px]" />
          <span className="t-value-sm text-ink">{s.none}</span> <span className="text-muted">no estimate</span>
        </span>
      </div>
      {allFlat && (
        <p className="t-dense mt-2 text-muted">
          {mode === "change"
            ? "No reach moved by 0.05 days or more: grey dashed lines tagged Δ≈0 mark where the cited effect is too small to register."
            : "Baseline and with-interventions look alike because no reach moved by 0.05 days. Switch to Change to see it per reach."}
        </p>
      )}
    </div>
  );
}

/** Hover readout in the scenario map: baseline -> scenario, delta, and the verdict. */
function ChangeCard({ r }: { r: ReachResult }) {
  const v = verdict(r);
  return (
    <div className="map-panel pointer-events-none absolute bottom-8 left-3 z-20 w-[min(360px,calc(100%-1.5rem))] px-4 py-3 md:w-[min(360px,calc(100%-260px))]" aria-hidden>
      <div className="flex items-center justify-between gap-3">
        <span className="font-mono text-[13px] font-semibold text-ink">{r.reach_id}</span>
        <span className="t-dense text-muted">{varShort(r.variable)}</span>
      </div>
      {r.baseline && r.scenario ? (
        <div className="mt-1.5 flex items-baseline gap-2">
          <span className="t-value text-muted">{fmtNum(r.baseline.exceedance_days, 1)}</span>
          <span className="text-faint">→</span>
          <span className="t-value text-ink">{fmtNum(r.scenario.exceedance_days, 1)}</span>
          <span className="t-dense text-muted">days/yr</span>
          {r.delta_days !== null && (
            <span className="t-value ml-auto font-semibold" style={{ color: changeColor(r.delta_days) }}>
              <ChangeGlyph d={r.delta_days} /> {signedDays(r.delta_days)}
            </span>
          )}
        </div>
      ) : null}
      <p className={`t-dense mt-1.5 ${v.tone === "moved" ? "text-ink" : "text-muted"}`}>{v.text}</p>
    </div>
  );
}

function MapLegend({ result, mode, variable }: { result: boolean; mode: MapMode; variable: Variable }) {
  // Open by default where there is room; on a phone it would cover most of the map.
  const [open, setOpen] = useState(() => typeof window === "undefined" || !!window.matchMedia?.("(min-width: 768px)").matches);
  const title = !result ? "Peak P(exceed), current forecast" : mode === "change" ? `Change in exceedance days, ${varShort(variable)}` : `Exceedance days per year, ${varShort(variable)}`;
  return (
    <div className="map-panel absolute right-3 bottom-8 z-10 w-[220px] max-w-[calc(100%-1.5rem)] text-[12px]">
      <button type="button" aria-expanded={open} aria-controls="scenario-legend" onClick={() => setOpen(!open)} className="flex w-full items-start gap-2 rounded-[var(--radius-md)] px-4 py-2.5 text-left">
        <Chevron className="mt-px text-muted" />
        <span className="t-eyebrow">{title}</span>
      </button>
      {open && (
        <div id="scenario-legend" className="px-4 pb-3">
          <div className="grid grid-cols-[auto_1fr] items-center gap-x-2.5 gap-y-1.5">
            {result && mode === "change"
              ? changeLegend.map((s) => (
                  <div key={s.label} className="contents">
                    <Swatch color={s.color} width={4} />
                    <span className="text-muted">{s.label}</span>
                  </div>
                ))
              : legendStops(result ? DAYS_BREAKS : PROB_BREAKS, (v) => (result ? v.toFixed(0) : v.toFixed(1))).map((s) => (
                  <div key={s.label} className="contents">
                    <Swatch color={s.color} width={s.width - 1} />
                    <span className="t-value-sm">{s.label}</span>
                  </div>
                ))}
            <Swatch hatch />
            <span className="text-muted">{result ? "no estimate (no threshold)" : "no threshold"}</span>
          </div>
          {result && mode === "change" && <p className="t-dense mt-2 text-faint">Tags along each reach carry the signed change.</p>}
        </div>
      )}
    </div>
  );
}

function Stepper({ step, done }: { step: number; done: boolean[] }) {
  const labels = ["Select reaches", "Interventions", "Run"];
  return (
    <ol className="flex items-center">
      {labels.map((l, i) => {
        const n = i + 1;
        const on = done[i] || step === n;
        return (
          <li key={l} className="flex flex-1 items-center last:flex-none">
            <span className="flex items-center gap-2">
              <span
                aria-hidden
                className={`inline-grid h-7 w-7 shrink-0 place-items-center rounded-full text-[12px] font-semibold ${
                  done[i] ? "bg-brand text-white" : step === n ? "border-2 border-brand text-brand-text shadow-[0_0_0_4px_var(--brand-50)]" : "border-2 border-hairline-strong text-faint"
                }`}
              >
                {done[i] ? (
                  <svg viewBox="0 0 10 10" width="10" height="10">
                    <path d="M1.5 5.2 4 7.5l4.5-5" fill="none" stroke="currentColor" strokeWidth="1.8" />
                  </svg>
                ) : (
                  n
                )}
              </span>
              <span className={`text-[13px] whitespace-nowrap ${on ? "font-semibold text-ink" : "text-faint"} ${step === n ? "" : "max-[380px]:sr-only"}`} aria-current={step === n ? "step" : undefined}>
                {l}
              </span>
            </span>
            {n < 3 && <span aria-hidden className={`mx-2 h-px min-w-3 flex-1 ${done[i] ? "bg-brand" : "bg-hairline"}`} />}
          </li>
        );
      })}
    </ol>
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
  const max = i.extent_max ?? 1;
  const val = s.extent ?? defaultExtent(i) ?? 0;
  const fill = `${((val - step) / Math.max(1e-9, max - step)) * 100}%`;
  return (
    <div
      className={`rounded-[var(--radius-md)] px-4 py-4 transition-colors ${
        s.enabled ? "bg-brand-50 shadow-[inset_0_0_0_1.5px_var(--brand-500)]" : "bg-raised shadow-[inset_0_0_0_1px_var(--border)] hover:shadow-[inset_0_0_0_1px_var(--border-strong)]"
      }`}
    >
      <label className="flex cursor-pointer items-start gap-3">
        <input type="checkbox" className="mt-[3px] shrink-0" checked={s.enabled} onChange={() => setLever(i.id, { enabled: !s.enabled })} />
        <span className="min-w-0 flex-1">
          <span className="flex flex-wrap items-center justify-between gap-x-2 gap-y-1">
            <span className="t-h3 text-[15px]">{i.name}</span>
            <span className="pill pill-sm pill-neutral">{allNotEstimable ? `No estimate on ${cityName}` : small ? "Small cited effect" : "Estimable"}</span>
          </span>
          <span className="t-dense mt-1 block text-muted">
            {humanize(i.direction)}s {i.feature.replaceAll("_", " ")}: cited <span className="font-mono text-ink">{fmtNum(i.magnitude, digits)}</span> (range{" "}
            <span className="font-mono">
              {fmtNum(i.uncertainty_range[0], digits)}–{fmtNum(i.uncertainty_range[1], digits)}
            </span>
            ), {i.effect_unit}.
          </span>
        </span>
      </label>
      <div className="pl-7">
        {i.extent_unit && (
          <div className="mt-3">
            <label className="flex items-center justify-between text-[12px]">
              <span className="text-muted">Extent</span>
              <span className={`t-value-sm ${s.enabled ? "text-ink" : "text-faint"}`}>
                {i.operation === "scale_down" ? fmtNum(s.extent, 2) : fmtNum(s.extent, 0)} <span className="font-sans font-normal text-faint">{i.extent_unit.split(" ")[0]}</span>
              </span>
            </label>
            <input
              type="range"
              min={step}
              max={max}
              step={step}
              value={val}
              onChange={(e) => setLever(i.id, { extent: Number(e.target.value) })}
              disabled={!s.enabled}
              className="mt-1 w-full"
              style={{ ["--fill" as string]: fill }}
              aria-label={`${i.name} extent`}
              aria-valuetext={`${s.extent} ${i.extent_unit}`}
            />
            <p className="t-dense text-faint">
              {i.extent_unit}
              {i.operation === "subtract_treated_share" ? "; cannot exceed a reach's current imperviousness" : i.operation === "scale_down" ? "; 1 = the coverage of the cited study" : ""}.
            </p>
          </div>
        )}
        {i.operation === "floor" && <p className="t-dense mt-2 text-muted">Fixed at the cited design width; no extent to choose.</p>}
        <div className="mt-3 flex flex-wrap gap-1.5">
          {Object.entries(i.paths).map(([v, p]) => (
            <span
              key={v}
              className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[11px] leading-3 font-medium ${
                p.path === "NOT_ESTIMABLE" ? "bg-surface text-faint shadow-[inset_0_0_0_1px_var(--border)]" : "bg-brand-50 text-brand-300"
              }`}
            >
              {p.path === "NOT_ESTIMABLE" && <span aria-hidden className="hatch inline-block h-2 w-2.5 rounded-[1px]" />}
              {varShort(v)}: {PATH_TEXT[p.path]}
            </span>
          ))}
        </div>
        {allNotEstimable && <p className="t-dense mt-2 text-ink">Selecting it will not change the result on {cityName}: the model's response failed the check and no cited direct effect exists.</p>}
        {small && (
          <p className="t-dense mt-2 text-ink">
            The cited effect is <span className="font-mono">{fmtNum(share! * 100, 1)} %</span> at this extent, so expect a change close to 0 days.
          </p>
        )}
        <details className="mt-2 text-[12px]">
          <summary className="inline-flex cursor-pointer items-center gap-1 text-muted hover:text-ink">
            <Chevron />
            Why each path
          </summary>
          <div className="mt-1.5 flex flex-col gap-1.5 pl-4 text-muted">
            {Object.entries(i.paths).map(([v, p]) => (
              <p key={v} className="break-words">
                <span className="text-ink">{varShort(v)}:</span> {p.reason}
              </p>
            ))}
          </div>
        </details>
        <details className="mt-1 text-[12px]">
          <summary className="inline-flex cursor-pointer items-center gap-1 text-muted hover:text-ink">
            <Chevron />
            Sources and cost
          </summary>
          <div className="pl-4">
            <p className="mt-1.5 text-muted">Effect size from</p>
            <Citations list={i.citations} />
            <p className="mt-2 text-muted">
              Cost{" "}
              <span className="font-mono text-ink">
                {i.cost_per_unit.currency} {fmtNum(i.cost_per_unit.low, 0)}–{fmtNum(i.cost_per_unit.high, 0)}
              </span>{" "}
              per {i.cost_per_unit.unit} ({i.cost_per_unit.price_basis}).
            </p>
            <Citations list={i.cost_per_unit.citation} />
          </div>
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
        <li key={k} className="border-l-2 border-hairline pl-2 break-words">
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
          <span className="block text-faint">{c.locator}</span>
        </li>
      ))}
    </ul>
  );
}

function ChangeGlyph({ d }: { d: number }) {
  if (Math.abs(d) < NEGLIGIBLE_DAYS) return <span aria-label="no change">→</span>;
  return <span aria-label={d < 0 ? "fewer days" : "more days"}>{d < 0 ? "▼" : "▲"}</span>;
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
    <section className="mt-6" aria-label="Scenario results">
      <div className="mb-4 flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
        <h2 className="t-title">Results</h2>
        <p className="t-dense text-muted">
          {humanize(result.label)}, {result.units}; reference year {fmtDay(result.reference_start, true)} – {fmtDay(result.reference_end, true)}; {Math.round(result.ci_level * 100)}% intervals, widened ×{result.widening_factor} on the
          scenario; model <span className="font-mono">{result.model_version}</span>
        </p>
      </div>
      <Headline result={result} ok={ok} />
      <div className="mt-4 grid gap-4 xl:grid-cols-[minmax(0,1.5fr)_minmax(0,1fr)]">
        <Card title="Per reach" caption="Days per year the reach exceeds its seasonal threshold, baseline and with interventions" bodyClass="!px-0">
          {ok.length > 0 && (
            <div className="max-h-[520px] overflow-auto">
              <table className="dtable min-w-[620px] text-[13px]">
                <thead>
                  <tr>
                    <th>Reach</th>
                    <th>Variable</th>
                    <th className="w-[34%]">Baseline → with interventions</th>
                    <th className="!text-right">Days per year</th>
                    <th className="!text-right">Change</th>
                  </tr>
                </thead>
                <tbody>
                  {ok.map((r) => (
                    <tr key={`${r.reach_id}-${r.variable}`} className="row-hover" onMouseEnter={() => useStore.getState().hover(r.reach_id)} onMouseLeave={() => useStore.getState().hover(null)}>
                      <td className="font-mono text-[12.5px]">{r.reach_id}</td>
                      <td className="text-muted">{varShort(r.variable)}</td>
                      <td>
                        <div className="relative h-4" aria-hidden>
                          <div className="absolute top-0 h-1.5 rounded-full bg-[var(--text-tertiary)]" style={{ width: `${(r.baseline!.exceedance_days / max) * 100}%` }} />
                          <div className="bar-descend absolute top-2.5 h-1.5 rounded-full bg-brand" style={{ width: `${((settled ? r.scenario! : r.baseline!).exceedance_days / max) * 100}%` }} />
                        </div>
                      </td>
                      <td className="num">
                        <span className="text-ink">
                          {fmtNum(r.baseline!.exceedance_days, 1)} → {fmtNum(r.scenario!.exceedance_days, 1)}
                        </span>
                        <span className="block text-[11px] text-faint">
                          [{fmtNum(r.baseline!.interval.low, 0)}–{fmtNum(r.baseline!.interval.high, 0)}] → [{fmtNum(r.scenario!.interval.low, 0)}–{fmtNum(r.scenario!.interval.high, 0)}]
                        </span>
                      </td>
                      <td className="num font-semibold" style={{ color: changeColor(r.delta_days) }}>
                        {r.delta_days !== null ? (
                          <>
                            <ChangeGlyph d={r.delta_days} /> {signedDays(r.delta_days)}
                          </>
                        ) : (
                          "—"
                        )}
                        {!r.complete && <span className="block font-sans text-[11px] font-normal text-faint">some levers not estimated</span>}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          <div className="px-5">
            {ok.some((r) => r.delta_days !== null && Math.abs(r.delta_days) < NEGLIGIBLE_DAYS) && (
              <p className="t-dense mt-3 text-muted">
                A change of <span className="font-mono text-ink">0.0</span> is the estimate, not a failure: the cited effect sizes here are small (detention basins reduce peak flow by about 0.3 % in the cited study), or the lever
                could not be estimated on this reach.
              </p>
            )}
            {other.length > 0 && (
              <details className="mt-3 text-[12px]" open={ok.length === 0}>
                <summary className="inline-flex cursor-pointer items-center gap-1 text-muted hover:text-ink">
                  <Chevron />
                  {other.length} reach-variables without an estimate
                </summary>
                <ul className="mt-1.5 pl-4">
                  {other.map((r) => (
                    <li key={`${r.reach_id}-${r.variable}`} className="hatch-border border-b border-hairline py-1.5 pl-2">
                      <span className="font-mono">{r.reach_id}</span> {varShort(r.variable)}: {humanize(r.status)}. <span className="text-muted">{r.reason}</span>
                    </li>
                  ))}
                </ul>
              </details>
            )}
            <LeverFlags result={result} />
            <p className="t-dense mt-3 text-muted">{result.spatial_scope}</p>
            <p className="t-dense mt-1 text-muted">{result.interval_method}</p>
          </div>
        </Card>
        <div className="flex min-w-0 flex-col gap-4">
          <Card title="Cost" caption="In each source's currency and price year; nothing converted." bodyClass="!px-0">
            <div className="max-h-[300px] overflow-auto">
              <table className="dtable text-[13px]">
                <thead>
                  <tr>
                    <th>Lever and reach</th>
                    <th className="!text-right">Total</th>
                  </tr>
                </thead>
                <tbody>
                  {costs
                    .filter((c) => c.total_low !== null && c.total_high !== null)
                    .map((c, k) => (
                      <tr key={k} className="row-hover align-top">
                        <td>
                          <span className="font-mono text-[12.5px]">{c.reach_id}</span> <span className="text-muted">{c.intervention_id.replaceAll("_", " ")}</span>
                        </td>
                        <td className="num">
                          {fmtMoney(c.total_low!, c.currency)}–{fmtMoney(c.total_high!, c.currency).replace(`${c.currency} `, "")}
                        </td>
                      </tr>
                    ))}
                  {/* Rows with no total say the same thing per reach: one line per lever. */}
                  {noTotalGroups(costs).map((g) => (
                    <tr key={g.key} className="align-top">
                      <td colSpan={2} className="!whitespace-normal">
                        <span className="font-semibold text-ink">{g.c.intervention_id.replaceAll("_", " ")}</span>{" "}
                        <span className="text-muted">
                          on {g.n} {g.n === 1 ? "reach" : "reaches"}:{" "}
                          <span className="font-mono text-ink">
                            {g.c.currency} {fmtNum(g.c.unit_cost_low, 0)}–{fmtNum(g.c.unit_cost_high, 0)}
                          </span>{" "}
                          per {g.c.unit}. No total: {g.c.reason}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
          <Card title="Coefficients used">
            <ul className="flex flex-col gap-3 text-[12px]">
              {result.levers
                .filter((l, k, arr) => arr.findIndex((x) => x.intervention_id === l.intervention_id) === k)
                .map((l) => (
                  <li key={l.intervention_id}>
                    <span className="font-semibold text-ink">{l.name}</span>
                    <span className="text-muted">
                      : magnitude <span className="font-mono text-ink">{l.magnitude}</span> (range{" "}
                      <span className="font-mono">
                        {l.uncertainty_range[0]}–{l.uncertainty_range[1]}
                      </span>
                      ), extent <span className="font-mono">{l.extent ?? "—"}</span>
                    </span>
                    <ul className="mt-1">
                      {effectCites
                        .filter((c) => c.intervention_id === l.intervention_id)
                        .map((c, k) => (
                          <li key={k} className="border-l-2 border-hairline pl-2 text-muted">
                            {c.authors} ({c.year}), {c.source}.{" "}
                            {c.doi ? (
                              <a className="link" href={`https://doi.org/${c.doi}`} target="_blank" rel="noreferrer">
                                doi:{c.doi}
                              </a>
                            ) : (
                              c.url
                            )}{" "}
                            — {c.locator}
                          </li>
                        ))}
                    </ul>
                  </li>
                ))}
            </ul>
          </Card>
        </div>
      </div>
    </section>
  );
}

function noTotalGroups(costs: ScenarioResult["costs"]) {
  const m = new Map<string, { key: string; c: ScenarioResult["costs"][number]; n: number }>();
  for (const c of costs) {
    if (c.total_low !== null && c.total_high !== null) continue;
    const key = [c.intervention_id, c.currency, c.unit_cost_low, c.unit_cost_high, c.unit, c.reason].join("|");
    const g = m.get(key);
    if (g) g.n++;
    else m.set(key, { key, c, n: 1 });
  }
  return [...m.values()];
}

function LeverFlags({ result }: { result: ScenarioResult }) {
  const seen = new Map<string, string>();
  for (const l of result.levers) seen.set(`${l.intervention_id}|${l.variable}`, `${l.name}, ${varShort(l.variable)}: ${PATH_TEXT[l.path]} — ${l.reason}`);
  const flags = new Set<string>();
  for (const r of result.reaches) for (const l of r.levers) for (const d of l.detail) flags.add(d);
  return (
    <div className="mt-3 text-[12px]">
      <p className="font-semibold text-ink">How each lever was applied</p>
      <ul className="mt-1 flex flex-col gap-0.5 text-muted">
        {[...seen.values()].map((t) => (
          <li key={t}>{t}</li>
        ))}
      </ul>
      {flags.size > 0 && (
        <details className="mt-1.5">
          <summary className="inline-flex cursor-pointer items-center gap-1 text-muted hover:text-ink">
            <Chevron />
            {flags.size} support warnings
          </summary>
          <ul className="pl-4 text-muted">
            {[...flags].map((f) => (
              <li key={f}>{f}</li>
            ))}
          </ul>
        </details>
      )}
    </div>
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
  const moved = ok.some((r) => r.delta_days !== null && Math.abs(r.delta_days) >= NEGLIGIBLE_DAYS);
  const s = mapSummary(result.reaches);
  const tiles = (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
      {[
        ["Better", s.better, "var(--change-improve)"],
        ["Worse", s.worse, "var(--change-worsen)"],
        ["Negligible (Δ≈0)", s.negligible, "var(--change-neutral)"],
        ["No estimate", s.none, ""],
      ].map(([l, n, c]) => (
        <div key={l as string} className="card relative overflow-hidden p-4 pt-[18px]">
          <span aria-hidden className={`absolute inset-x-0 top-0 h-[3px] ${c ? "" : "hatch"}`} style={{ background: (c as string) || undefined }} />
          <span className="t-eyebrow">{l}</span>
          <span className="t-stat mt-1.5 block">{n as number}</span>
          <span className="t-dense text-muted">reach-variables</span>
        </div>
      ))}
    </div>
  );
  if (ok.length && moved)
    return (
      <div className="flex flex-col gap-4">
        <div className="card flex flex-wrap items-baseline gap-x-4 gap-y-1 px-5 py-4">
          <span className="t-reading">
            {fmtNum(base, 1)} <span className="text-faint">→</span> <span className="text-brand-300">{fmtNum(scen, 1)}</span>
          </span>
          <span className="t-ui text-muted">
            exceedance days per year, summed over {ok.length} reach-{ok.length === 1 ? "variable" : "variables"} with an estimate
          </span>
        </div>
        {tiles}
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
    <div className="flex flex-col gap-4">
      {tiles}
      <div className="rounded-[var(--radius-md)] border-l-4 border-l-[var(--severity-watch)] bg-[var(--severity-watch-bg)] px-5 py-4" role="status">
        <p className="t-h3">Why baseline and scenario look the same</p>
        <ul className="mt-2 flex list-disc flex-col gap-1 pl-5 text-[13px] text-muted">
          {reasons.map((r) => (
            <li key={r}>{r}</li>
          ))}
        </ul>
        <p className="t-dense mt-2 text-muted">This is the estimate, not an error: no effect size is invented where the evidence does not support one. On the map, Change mode marks these reaches Δ≈0.</p>
      </div>
    </div>
  );
}
