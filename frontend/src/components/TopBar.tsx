import type { City } from "../api/types";
import { VIEWS, useStore } from "../store";

export function TopBar({ cities }: { cities: City[] | null }) {
  const { view, city, setView, setCity } = useStore();
  const current = cities?.find((c) => c.city === city);
  return (
    <header className="flex h-14 shrink-0 items-center gap-6 border-b border-hairline px-4">
      <div className="flex items-baseline gap-3">
        <span className="t-title">Kingfisher</span>
        <label className="sr-only" htmlFor="city">
          City
        </label>
        <select
          id="city"
          value={city}
          onChange={(e) => setCity(e.target.value)}
          className="t-ui border-0 border-b border-hairline bg-transparent pb-0.5 pr-1 text-ink"
        >
          {(cities ?? [{ city, name: city } as City]).map((c) => (
            <option key={c.city} value={c.city}>
              {c.name}
              {c.status === "NOT_BUILT" ? " (not built)" : ""}
            </option>
          ))}
        </select>
        {current && current.status === "READY" && (
          <span className="t-dense hidden text-muted md:inline">
            {current.reaches} reaches, {current.optically_observable} optically observable
          </span>
        )}
      </div>
      <nav className="ml-auto flex h-full items-stretch" aria-label="Views">
        {VIEWS.map((v) => (
          <button
            key={v.id}
            type="button"
            onClick={() => setView(v.id)}
            aria-current={view === v.id ? "page" : undefined}
            className={`t-ui -mb-px border-b-2 px-3 ${
              view === v.id ? "border-kf text-ink" : "border-transparent text-muted hover:text-ink"
            }`}
          >
            {v.label}
          </button>
        ))}
      </nav>
    </header>
  );
}
