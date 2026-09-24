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

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, init);
  } catch {
    throw new ApiError(0, "The Kingfisher API is not reachable. Start it with `make api`.");
  }
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
    request<ScenarioResponse>("/api/scenarios", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  validation: (city: string) => request<ValidationResponse>(`/api/validation/metrics${q({ city })}`),
};
