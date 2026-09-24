import type { City } from "../api/types";
import { VIEWS, useStore } from "../store";

/** The favicon's mark: a stream line and the bird. */
function Mark() {
  return (
    <svg aria-hidden viewBox="0 0 32 32" width="22" height="22" className="shrink-0 self-center">
      <path d="M4 21c4-3 8-3 12 0s8 3 12 0" fill="none" stroke="var(--kingfisher)" strokeWidth="3" strokeLinecap="round" />
      <circle cx="22" cy="11" r="3" fill="var(--ink)" />
    </svg>
  );
}

export function TopBar({ cities }: { cities: City[] | null }) {
  const { view, city, setView, setCity } = useStore();
  const current = cities?.find((c) => c.city === city);
  return (
    <header className="flex shrink-0 flex-col border-b border-hairline md:h-14 md:flex-row md:items-stretch">
      <div className="flex h-12 min-w-0 items-center gap-3 px-4 md:h-auto">
        <span className="flex items-center gap-2">
          <Mark />
          <span className="t-title">Kingfisher</span>
        </span>
        <span aria-hidden className="h-5 w-px bg-hairline" />
        <label className="sr-only" htmlFor="city">
          City
        </label>
        <select
          id="city"
          value={city}
          onChange={(e) => setCity(e.target.value)}
          className="t-ui ctl h-8 max-w-[11rem] border border-transparent bg-transparent pr-1 pl-1.5 text-ink hover:border-hairline"
        >
          {(cities ?? [{ city, name: city } as City]).map((c) => (
            <option key={c.city} value={c.city}>
              {c.name}
              {c.status === "NOT_BUILT" ? " (not built)" : ""}
            </option>
          ))}
        </select>
        {current && current.status === "READY" && (
          <span className="t-dense hidden truncate text-muted lg:inline">
            <span className="t-value-sm text-ink">{current.reaches}</span> reaches, <span className="t-value-sm text-ink">{current.optically_observable}</span> optically observable
          </span>
        )}
      </div>
      <nav className="flex h-11 items-stretch border-t border-hairline md:ml-auto md:h-auto md:border-t-0 md:pr-2" aria-label="Views">
        {VIEWS.map((v) => (
          <button
            key={v.id}
            type="button"
            onClick={() => setView(v.id)}
            aria-current={view === v.id ? "page" : undefined}
            className={`t-ui -mb-px flex-1 border-b-2 px-3 md:flex-none ${
              view === v.id ? "border-kf text-ink" : "border-transparent text-muted hover:border-hairline hover:text-ink"
            }`}
          >
            {v.label}
          </button>
        ))}
      </nav>
    </header>
  );
}
