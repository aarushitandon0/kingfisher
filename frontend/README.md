# Kingfisher frontend

React + TypeScript + Vite + MapLibre GL + Zustand + Tailwind + Recharts. Visual language:
`design.md` (hydrographic chart: chart paper, hairlines, the clear-water-to-silt ramp).

```
make api            # FastAPI on :8000
make web-install    # once
make web            # Vite on http://localhost:5173, proxies /api to :8000
```

Heat Surgeon shipped only as a built bundle that draws its streets as SVG from OSM
snapshots - there was no MapLibre code to lift. The basemap here is OpenFreeMap's keyless
Positron style, recoloured to chart paper in `src/map/basemap.ts`.

## Layout

| Path | What |
|---|---|
| `src/store.ts` | Zustand: view, city, selection, hover, rail date, filters, scenario state; view/city/reach mirrored to the URL hash |
| `src/api/` | typed client; `types.ts` mirrors `api/schemas.py` |
| `src/map/ReachMap.tsx` | the map: sediment-ramped reaches, dashed driver-predicted, hatched insufficient evidence, catchment on hover, alert pins at the downstream end, exposure layer, lasso |
| `src/components/HydrographRail.tsx` | the rail: drag the head (or arrow keys) and the map re-renders at that date |
| `src/lib/timeline.ts` | per-reach value at any date from `/api/timeline` - gaps stay gaps |
| `src/views/` | Map, Alerts, Scenarios (swipe before/after), Validation |

## Rules the UI keeps

- No number is computed in the browser that the API did not serve, except colour lookup
  and the past-date ratio observed ÷ seasonal threshold (both stated in the legend).
- `INSUFFICIENT_EVIDENCE` is drawn and listed, never filtered out by default.
- Mono type is for measured or computed values only.
- Every scenario coefficient is shown with its citation as text.

MapLibre 6 is excluded from Vite's dependency pre-bundling (`vite.config.ts`): it resolves
its worker relative to `import.meta.url`.
