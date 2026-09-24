import { create } from "zustand";
import type { ScenarioResponse } from "./api/types";

export type View = "map" | "alerts" | "scenarios" | "validation";
export const VIEWS: { id: View; label: string }[] = [
  { id: "map", label: "Map" },
  { id: "alerts", label: "Alerts" },
  { id: "scenarios", label: "Scenarios" },
  { id: "validation", label: "Validation" },
];

export type ObservabilityFilter = "all" | "observable" | "driver";

export interface LeverState {
  enabled: boolean;
  extent: number | null;
}

interface State {
  view: View;
  city: string;
  /** Map selection (reach detail) and hover (catchment preview). */
  selected: string | null;
  hovered: string | null;
  /** Hydrograph rail head. null = "now": the map shows the peak over the forecast window. */
  railDate: string | null;
  railVariable: "turbidity_proxy" | "ndci";
  showExposure: boolean;
  observability: ObservabilityFilter;

  /** Scenario workbench. */
  scenarioReaches: string[];
  levers: Record<string, LeverState>;
  scenario: ScenarioResponse | null;
  swipe: number; // 0-1, fraction of the map width showing the baseline

  setView: (v: View) => void;
  setCity: (c: string) => void;
  select: (id: string | null) => void;
  hover: (id: string | null) => void;
  setRailDate: (d: string | null) => void;
  setRailVariable: (v: "turbidity_proxy" | "ndci") => void;
  toggleExposure: () => void;
  setObservability: (f: ObservabilityFilter) => void;
  toggleScenarioReach: (id: string) => void;
  setScenarioReaches: (ids: string[]) => void;
  setLever: (id: string, s: Partial<LeverState>) => void;
  setScenario: (s: ScenarioResponse | null) => void;
  setSwipe: (x: number) => void;
}

function readHash(): { view: View; city: string; reach: string | null } {
  const parts = window.location.hash.replace(/^#\/?/, "").split("/");
  const view = (VIEWS.some((v) => v.id === parts[0]) ? parts[0] : "map") as View;
  return { view, city: parts[1] || "coimbra", reach: parts[2] || null };
}

function writeHash(s: Pick<State, "view" | "city" | "selected">) {
  const h = `#/${s.view}/${s.city}${s.selected && s.view === "map" ? `/${s.selected}` : ""}`;
  if (window.location.hash !== h) window.history.replaceState(null, "", h);
}

const initial = readHash();

export const useStore = create<State>((set, get) => ({
  view: initial.view,
  city: initial.city,
  selected: initial.reach,
  hovered: null,
  railDate: null,
  railVariable: "turbidity_proxy",
  showExposure: false,
  observability: "all",
  scenarioReaches: [],
  levers: {},
  scenario: null,
  swipe: 0.5,

  setView: (view) => {
    set({ view });
    writeHash(get());
  },
  setCity: (city) => {
    if (city === get().city) return;
    set({ city, selected: null, hovered: null, railDate: null, scenarioReaches: [], scenario: null });
    writeHash(get());
  },
  select: (selected) => {
    set({ selected });
    writeHash(get());
  },
  hover: (hovered) => set({ hovered }),
  setRailDate: (railDate) => set({ railDate }),
  setRailVariable: (railVariable) => set({ railVariable }),
  toggleExposure: () => set({ showExposure: !get().showExposure }),
  setObservability: (observability) => set({ observability }),
  toggleScenarioReach: (id) => {
    const cur = get().scenarioReaches;
    set({ scenarioReaches: cur.includes(id) ? cur.filter((r) => r !== id) : [...cur, id] });
  },
  setScenarioReaches: (scenarioReaches) => set({ scenarioReaches }),
  setLever: (id, s) => {
    const cur = get().levers[id] ?? { enabled: false, extent: null };
    set({ levers: { ...get().levers, [id]: { ...cur, ...s } } });
  },
  setScenario: (scenario) => set({ scenario }),
  setSwipe: (swipe) => set({ swipe: Math.min(1, Math.max(0, swipe)) }),
}));

window.addEventListener("hashchange", () => {
  const h = readHash();
  const s = useStore.getState();
  if (h.view !== s.view) useStore.setState({ view: h.view });
  if (h.city !== s.city) s.setCity(h.city);
});
