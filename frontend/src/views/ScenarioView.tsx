import type { Map as MLMap } from "maplibre-gl";
import { useEffect, useMemo, useRef, useState } from "react";
import { api, ApiError } from "../api/client";
import type { Citation, InterventionInfo, ReachCollection, ReachResult, ScenarioResult, Variable } from "../api/types";
import { ErrorNote, Loading, Swatch } from "../components/bits";
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

  if (!fc) return <div className="p-6">{reaches.error ? <ErrorNote error={reaches.error} what="Reaches" /> : <Loading what="reaches" />}</div>;

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <p className="t-ui border-b border-hairline bg-paper-alt px-4 py-1.5 text-muted" role="note">
        {CAVEAT}
      </p>
      <div className="flex min-h-0 flex-1 flex-col md:flex-row">
        <div className="flex min-h-[360px] flex-1 flex-col">
          <div className="flex flex-wrap items-center gap-3 border-b border-hairline px-3 py-1.5 t-dense">
            <span className="text-muted">Select reaches</span>
            <div role="radiogroup" aria-label="Selection tool" className="flex gap-1">
              {[
                [false, "Click"],
                [true, "Lasso"],
              ].map(([v, label]) => (
                <button key={String(label)} type="button" role="radio" aria-checked={lasso === v} onClick={() => setLasso(v as boolean)} className={`border px-2 ${lasso === v ? "border-ink text-ink" : "border-hairline text-muted"}`}>
                  {label as string}
                </button>
              ))}
            </div>
            <span className="t-value-sm">{scenarioReaches.length}</span>
            <span className="text-muted">selected</span>
            <button type="button" onClick={() => setScenarioReaches([])} disabled={!scenarioReaches.length} className="text-muted underline underline-offset-2 disabled:no-underline">
              Clear
            </button>
            {result && (
              <div className="ml-auto flex items-center gap-2">
                <span className="text-muted">Colour by</span>
                {(["turbidity_proxy", "ndci"] as Variable[]).map((v) => (
                  <button key={v} type="button" aria-pressed={variable === v} onClick={() => setVariable(v)} className={variable === v ? "text-ink underline underline-offset-4" : "text-muted"}>
                    {v === "ndci" ? "NDCI" : "turbidity"}
                  </button>
                ))}
              </div>
            )}
          </div>
          <div ref={wrap} className="relative min-h-0 flex-1 overflow-hidden" onPointerMove={onDivider} onPointerUp={() => (drag.current = false)}>
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
                  <div className="mx-auto h-full w-px bg-ink" />
                  <span className="t-dense absolute top-2 right-3 whitespace-nowrap bg-paper px-1">baseline</span>
                  <span className="t-dense absolute top-2 left-3 whitespace-nowrap bg-paper px-1" style={{ color: SCENARIO }}>
                    with interventions
                  </span>
                </div>
              </>
            )}
            <div className="absolute bottom-7 left-2 z-10 border border-hairline bg-paper/95 px-2 py-1.5 t-dense">
              <p className="mb-1">{result ? `Exceedance days per year, ${VARIABLE_LABEL[variable]}` : "Peak P(exceed), current forecast"}</p>
              <div className="grid grid-cols-[auto_1fr] items-center gap-x-2">
                {legendStops(result ? DAYS_BREAKS : PROB_BREAKS, (v) => (result ? v.toFixed(0) : v.toFixed(1))).map((s) => (
                  <div key={s.label} className="contents">
                    <Swatch color={s.color} />
                    <span className="t-value-sm">{s.label}</span>
                  </div>
                ))}
                <Swatch hatch />
                <span>{result ? "not in the scenario, or no estimate" : "no threshold"}</span>
              </div>
            </div>
          </div>
        </div>
        <aside className="max-h-[60vh] w-full shrink-0 overflow-y-auto border-t border-hairline p-4 md:max-h-none md:w-[420px] md:border-l md:border-t-0" aria-label="Interventions">
          <h2 className="t-title">Interventions</h2>
          {catalogue.loading && <Loading what="coefficient table" />}
          <ErrorNote error={catalogue.error} what="Coefficient table" />
          {catalogue.data && (
            <>
              <p className="t-dense text-muted">
                Coefficient table v<span className="t-value-sm">{catalogue.data.coefficient_table_version}</span>. Effect sizes come only from the cited sources below; the model never supplies them.
              </p>
              <div className="mt-3 flex flex-col">
                {catalogue.data.interventions.map((i) => (
                  <Lever key={i.id} i={i} />
                ))}
                {catalogue.data.not_quantified.map((n) => (
                  <div key={n.id} className="hairline-b py-3 text-muted">
                    <p className="t-ui">
                      <input type="checkbox" disabled aria-label={`${n.name} (not quantified)`} className="mr-2 align-middle" />
                      {n.name} — not quantified
                    </p>
                    <p className="t-dense mt-1">{n.reason}</p>
                    <Citations list={n.citations} />
                  </div>
                ))}
              </div>
              <button type="button" onClick={run} disabled={!canRun} className="t-ui mt-4 w-full border border-ink bg-ink py-2 text-paper disabled:border-hairline disabled:bg-paper-alt disabled:text-muted">
                {running ? "Running scenario" : "Run scenario"}
              </button>
              {running && (
                <p className="t-dense mt-1 text-muted">
                  Re-scoring a full reference year for {scenarioReaches.length} reaches, baseline and scenario, across the coefficient range. Roughly half a second per reach.
                </p>
              )}
              {scenarioReaches.length > 100 && <p className="t-dense mt-1 text-ink">A scenario covers at most 100 reaches; narrow the selection.</p>}
              {!scenarioReaches.length && <p className="t-dense mt-1 text-muted">Select reaches on the map first.</p>}
              {scenarioReaches.length > 0 && !enabled.length && <p className="t-dense mt-1 text-muted">Choose at least one intervention.</p>}
              <ErrorNote error={runError} what="Scenario" />
            </>
          )}
        </aside>
      </div>
      {result && <Results result={result} />}
    </div>
  );
}

function Lever({ i }: { i: InterventionInfo }) {
  const { levers, setLever } = useStore();
  const s = levers[i.id] ?? { enabled: false, extent: defaultExtent(i) };
  const allNotEstimable = Object.values(i.paths).every((p) => p.path === "NOT_ESTIMABLE");
  const step = i.operation === "scale_down" ? 0.05 : 1;
  return (
    <div className="hairline-b py-3">
      <label className="t-ui flex items-center gap-2">
        <input type="checkbox" checked={s.enabled} onChange={() => setLever(i.id, { enabled: !s.enabled })} />
        <span className="text-ink">{i.name}</span>
      </label>
      <p className="t-dense mt-1 text-muted">
        {humanize(i.direction)}s {i.feature.replaceAll("_", " ")}: cited magnitude <span className="t-value-sm text-ink">{fmtNum(i.magnitude, i.magnitude < 1 ? 3 : 1)}</span>{" "}
        (range <span className="t-value-sm">{fmtNum(i.uncertainty_range[0], i.magnitude < 1 ? 3 : 1)}–{fmtNum(i.uncertainty_range[1], i.magnitude < 1 ? 3 : 1)}</span>), {i.effect_unit}.
      </p>
      {i.extent_unit && (
        <div className="mt-2">
          <label className="t-dense flex items-center gap-2">
            <span className="w-24 shrink-0 text-muted">Extent</span>
            <input
              type="range"
              min={step}
              max={i.extent_max ?? 1}
              step={step}
              value={s.extent ?? defaultExtent(i) ?? 0}
              onChange={(e) => setLever(i.id, { extent: Number(e.target.value) })}
              disabled={!s.enabled}
              className="flex-1"
              aria-valuetext={`${s.extent} ${i.extent_unit}`}
            />
            <span className="t-value-sm w-12 text-right">{i.operation === "scale_down" ? fmtNum(s.extent, 2) : fmtNum(s.extent, 0)}</span>
          </label>
          <p className="t-dense text-muted">{i.extent_unit}{i.operation === "subtract_treated_share" ? "; cannot exceed a reach's current imperviousness" : i.operation === "scale_down" ? "; 1 = the coverage of the cited study" : ""}.</p>
        </div>
      )}
      {i.operation === "floor" && <p className="t-dense mt-1 text-muted">Fixed at the cited design width; no extent to choose.</p>}
      <div className="mt-2 t-dense">
        {Object.entries(i.paths).map(([v, p]) => (
          <p key={v} className={p.path === "NOT_ESTIMABLE" ? "text-muted" : ""}>
            <span className="text-ink">{v === "ndci" ? "NDCI" : "Turbidity"}:</span> {PATH_TEXT[p.path]}. <span className="text-muted">{p.reason}</span>
          </p>
        ))}
        {allNotEstimable && s.enabled && <p className="mt-1 text-ink">This lever will return no estimate on this city: the model's response to it failed the check and no cited direct effect exists.</p>}
      </div>
      <Citations list={i.citations} />
      <p className="t-dense mt-1 text-muted">
        Cost <span className="t-value-sm text-ink">{i.cost_per_unit.currency} {fmtNum(i.cost_per_unit.low, 0)}–{fmtNum(i.cost_per_unit.high, 0)}</span> per {i.cost_per_unit.unit} ({i.cost_per_unit.price_basis}).
      </p>
      <Citations list={i.cost_per_unit.citation} />
    </div>
  );
}

/** Citations are a feature: shown as text, never behind an icon or tooltip. */
function Citations({ list }: { list: Citation[] }) {
  return (
    <ul className="t-dense mt-1.5 flex flex-col gap-1">
      {list.map((c, k) => (
        <li key={k} className="border-l border-hairline pl-2">
          {c.authors} ({c.year}). {c.title}. <span className="italic">{c.source}</span>.{" "}
          {c.doi ? (
            <a className="text-kf underline underline-offset-2" href={`https://doi.org/${c.doi}`} target="_blank" rel="noreferrer">
              doi:{c.doi}
            </a>
          ) : c.url ? (
            <a className="text-kf underline underline-offset-2 break-all" href={c.url} target="_blank" rel="noreferrer">
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
    <section className="max-h-[42vh] shrink-0 overflow-y-auto border-t border-hairline px-4 py-3" aria-label="Scenario results">
      <div className="flex flex-wrap items-baseline gap-x-4">
        <h2 className="t-title">Results</h2>
        <p className="t-dense text-muted">
          {result.label}, {result.units}; reference year {fmtDay(result.reference_start, true)} – {fmtDay(result.reference_end, true)}; {Math.round(result.ci_level * 100)}% intervals, widened ×{result.widening_factor} on the scenario; model{" "}
          <span className="t-value-sm">{result.model_version}</span>
        </p>
      </div>
      <div className="grid gap-6 lg:grid-cols-[1.4fr_1fr]">
        <div>
          <table className="mt-2 w-full t-dense">
            <thead>
              <tr className="hairline-b text-left text-muted">
                <th className="py-1 font-normal">Reach</th>
                <th className="py-1 font-normal">Variable</th>
                <th className="w-[38%] py-1 font-normal">Baseline → with interventions</th>
                <th className="py-1 text-right font-normal">Days per year</th>
                <th className="py-1 text-right font-normal">Change</th>
              </tr>
            </thead>
            <tbody>
              {ok.map((r) => (
                <tr key={`${r.reach_id}-${r.variable}`} className="hairline-b align-middle">
                  <td className="py-1.5">{r.reach_id}</td>
                  <td className="py-1.5 text-muted">{r.variable === "ndci" ? "NDCI" : "turbidity"}</td>
                  <td className="py-1.5">
                    <div className="relative h-4" aria-hidden>
                      <div className="absolute top-0 h-1.5 bg-heavy" style={{ width: `${(r.baseline!.exceedance_days / max) * 100}%` }} />
                      <div className="bar-descend absolute top-2 h-1.5" style={{ background: SCENARIO, width: `${((settled ? r.scenario! : r.baseline!).exceedance_days / max) * 100}%` }} />
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
          {ok.some((r) => r.delta_days !== null && Math.abs(r.delta_days) < 0.05) && (
            <p className="t-dense mt-2">
              A change of <span className="t-value-sm">0.0</span> is the estimate, not a failure: the cited effect sizes here are small (detention basins reduce peak flow by about 0.3 % in the cited study), or the lever could not be estimated on this reach.
            </p>
          )}
          {other.length > 0 && (
            <details className="mt-2 t-dense" open={ok.length === 0}>
              <summary className="cursor-pointer text-muted">
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
          <h3 className="t-ui mt-2 text-ink">Cost</h3>
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
          <h3 className="t-ui mt-3 text-ink">Coefficients used</h3>
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
          <summary className="cursor-pointer text-muted">{flags.size} support warnings</summary>
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
