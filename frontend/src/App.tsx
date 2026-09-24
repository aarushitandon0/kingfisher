import { useMemo } from "react";
import { api } from "./api/client";
import { ErrorNote } from "./components/bits";
import { TopBar } from "./components/TopBar";
import { buildIndex } from "./lib/timeline";
import { useApi } from "./lib/useApi";
import { useStore } from "./store";
import { AlertsView } from "./views/AlertsView";
import { MapView } from "./views/MapView";
import { ScenarioView } from "./views/ScenarioView";
import { ValidationView } from "./views/ValidationView";

export function App() {
  const { view, city } = useStore();
  const cities = useApi("cities", api.cities);
  const current = cities.data?.find((c) => c.city === city);
  const built = current?.status === "READY";
  const reaches = useApi(built ? `reaches:${city}` : null, () => api.reaches(city));
  const timelineRaw = useApi(built && view === "map" ? `timeline:${city}` : null, () => api.timeline(city));
  const timeline = useMemo(
    () => ({ ...timelineRaw, data: timelineRaw.data ? buildIndex(timelineRaw.data) : null }),
    [timelineRaw],
  );

  return (
    <div className="flex h-full flex-col">
      <a href="#main" className="sr-only focus:not-sr-only focus:absolute focus:left-2 focus:top-2 focus:z-50 focus:bg-paper focus:px-2">
        Skip to content
      </a>
      <TopBar cities={cities.data} />
      <main id="main" className="flex min-h-0 flex-1 flex-col">
        <ErrorNote error={cities.error} what="City list" />
        {cities.data && !current && <p className="t-ui p-6 text-muted">No city called "{city}" is configured.</p>}
        {current && view === "map" && <MapView key={city} city={current} reaches={reaches} timeline={timeline} />}
        {current && view === "alerts" && <AlertsView key={city} city={city} />}
        {current && view === "scenarios" && (built ? <ScenarioView key={city} city={city} reaches={reaches} /> : <NotBuilt name={current.name} city={city} />)}
        {current && view === "validation" && <ValidationView key={city} city={city} />}
      </main>
    </div>
  );
}

function NotBuilt({ name, city }: { name: string; city: string }) {
  return (
    <p className="t-body p-6 text-muted">
      {name} has no reaches yet. Build it with <code className="t-value-sm">make l0 city={city}</code> and the stages after it.
    </p>
  );
}
