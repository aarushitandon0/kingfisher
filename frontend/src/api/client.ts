import type {
  AlertDetail,
  AlertsResponse,
  AttributionResponse,
  CatchmentFeature,
  City,
  ExposureLayer,
  ForecastResponse,
  InterventionCatalogue,
  InterventionRequest,
  ReachCollection,
  ReachDetail,
  ScenarioResponse,
  Timeline,
  ValidationResponse,
} from "./types";

/** An API error carrying the server's own `detail` - shown to the user verbatim, because
 * the server's messages say what is missing and which command builds it. */
export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly detail: string,
  ) {
    super(detail);
  }
}

/** Static snapshot build (`npm run build:snapshot`, scripts/build_snapshot.py): there is no
 * server; every GET the UI makes was saved from the real API as a JSON file, and scenario
 * runs are the precomputed ones listed in snapshot/scenarios/index.json. */
export const SNAPSHOT = import.meta.env.VITE_SNAPSHOT === "1";

/** Where a GET lives in the snapshot: the path, with the query after a `~`. Must match
 * snapshot_file() in scripts/build_snapshot.py. */
export function snapshotFile(path: string): string {
  const [p, query] = path.split("?");
  return `./snapshot${p}${query ? `~${query}` : ""}.json`;
}

/** A link to an API resource (e.g. the FHIR bundle), valid in both builds. */
export const apiHref = (path: string) => (SNAPSHOT ? snapshotFile(path) : path);

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(SNAPSHOT ? snapshotFile(path) : path, init);
  } catch {
    throw new ApiError(0, "The Kingfisher API is not reachable. Start it with `make api`.");
  }
  if (SNAPSHOT && res.status === 404)
    throw new ApiError(404, "Not in this hosted snapshot. The full system (make api) serves it live.");
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (typeof body?.detail === "string") detail = body.detail;
      else if (Array.isArray(body?.detail)) detail = body.detail.map((d: { msg: string }) => d.msg).join("; ");
    } catch {
      /* non-JSON error body: keep the status line */
    }
    throw new ApiError(res.status, detail);
  }
  return (await res.json()) as T;
}

const q = (params: Record<string, string | number | undefined>) =>
  "?" +
  Object.entries(params)
    .filter(([, v]) => v !== undefined)
    .map(([k, v]) => `${k}=${encodeURIComponent(String(v))}`)
    .join("&");

export const api = {
  cities: () => request<{ cities: City[] }>("/api/cities").then((r) => r.cities),
  reaches: (city: string) => request<ReachCollection>(`/api/reaches${q({ city })}`),
  reach: (id: string, historyDays = 3650) =>
    request<ReachDetail>(`/api/reaches/${id}${q({ history_days: historyDays })}`),
  forecast: (id: string) => request<ForecastResponse>(`/api/reaches/${id}/forecast`),
  catchment: (id: string) => request<CatchmentFeature>(`/api/reaches/${id}/catchment`),
  attribution: (id: string) => request<AttributionResponse>(`/api/reaches/${id}/attribution`),
  exposure: (city: string) => request<ExposureLayer>(`/api/exposure${q({ city })}`),
  timeline: (city: string) => request<Timeline>(`/api/timeline${q({ city })}`),
  alerts: (city: string) => request<AlertsResponse>(`/api/alerts${q({ city, active: "true" })}`),
  alert: (id: string) => request<AlertDetail>(`/api/alerts/${id}`),
  interventions: (city: string) =>
    request<InterventionCatalogue>(`/api/scenarios/interventions${q({ city })}`),
  runScenario: (body: { name?: string; interventions: InterventionRequest[] }) =>
    SNAPSHOT
      ? snapshotScenario(body.interventions)
      : request<ScenarioResponse>("/api/scenarios", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        }),
  presets: (city: string) =>
    request<ScenarioPreset[]>("/api/scenarios/index").then((all) => all.filter((p) => p.city === city)),
  validation: (city: string) => request<ValidationResponse>(`/api/validation/metrics${q({ city })}`),
};

/** A scenario the real engine ran when the snapshot was built: its exact request, and the
 * file holding the response. */
export interface ScenarioPreset {
  id: string;
  city: string;
  name: string;
  interventions: InterventionRequest[];
  file: string;
}

/** Order-free identity of a scenario request, so a preset matches however it was selected. */
export function scenarioKey(interventions: InterventionRequest[]): string {
  return JSON.stringify(
    interventions
      .map((i) => ({ type: i.type, extent: i.extent ?? null, reach_ids: [...i.reach_ids].sort() }))
      .sort((a, b) => a.type.localeCompare(b.type)),
  );
}

/** In the snapshot a run can only replay a precomputed scenario - never a composed one. */
async function snapshotScenario(interventions: InterventionRequest[]): Promise<ScenarioResponse> {
  const all = await request<ScenarioPreset[]>("/api/scenarios/index");
  const key = scenarioKey(interventions);
  const hit = all.find((p) => scenarioKey(p.interventions) === key);
  if (!hit)
    throw new ApiError(
      0,
      "This hosted snapshot has no live model, so it can only show the precomputed scenarios listed above. Run the full system (make api) to try any selection.",
    );
  return request<ScenarioResponse>(hit.file);
}
