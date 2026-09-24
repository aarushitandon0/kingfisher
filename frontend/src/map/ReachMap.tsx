import * as maplibregl from "maplibre-gl";
import type { GeoJSONSource, Map as MLMap, MapLayerMouseEvent } from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
// MapLibre 6 finds its worker from a runtime URL the bundler cannot see, so the worker (and
// the shared chunk it imports) is bundled explicitly and handed over.
import workerUrl from "maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url";
import { useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import type { CatchmentFeature, ExposureLayer, ReachCollection, ReachFeature } from "../api/types";
import { cached } from "../lib/useApi";
import { ALERT, INK, INK_MUTED, KINGFISHER, PAPER, UNKNOWN, WATCH, rampExpression } from "../lib/ramp";
import { chartPaperStyle, hatchImage } from "./basemap";

export interface ReachValue {
  value: number | null;
  spread?: number | null;
}

export interface ReachMapProps {
  reaches: ReachCollection;
  /** Per-reach value to colour by. Missing / null -> hatched "no value" state. */
  values: Map<string, ReachValue>;
  breaks: number[];
  /** Number format for the soundings set along each reach. */
  sounding: (v: number) => string;
  highlighted?: string[]; // scenario selection
  selected?: string | null;
  hovered?: string | null;
  showPins?: boolean;
  showExposure?: boolean;
  showCatchments?: boolean;
  /** Reaches outside this set are drawn faint (observability filter). */
  visible?: Set<string> | null;
  lasso?: boolean;
  onLasso?: (ids: string[]) => void;
  onClick?: (id: string | null) => void;
  onHover?: (id: string | null) => void;
  onMap?: (m: MLMap) => void;
  interactive?: boolean;
  /** Fit the network on load and on a city change. False for a map whose camera is
   * driven by another (the scenario swipe). */
  fit?: boolean;
  label: string;
}

maplibregl.setWorkerUrl(workerUrl);

const EMPTY: GeoJSON.FeatureCollection = { type: "FeatureCollection", features: [] };

/** OSM waterways are drawn in the direction of flow: the last vertex is downstream. */
function downstreamEnd(f: ReachFeature): [number, number] | null {
  const g = f.geometry;
  if (g.type === "LineString") return (g.coordinates as [number, number][]).at(-1) ?? null;
  if (g.type === "MultiLineString") return (g.coordinates as [number, number][][]).at(-1)?.at(-1) ?? null;
  return null;
}

function bounds(fc: ReachCollection): maplibregl.LngLatBounds | null {
  const b = new maplibregl.LngLatBounds();
  let any = false;
  const add = (c: unknown) => {
    if (Array.isArray(c) && typeof c[0] === "number") {
      b.extend(c as [number, number]);
      any = true;
    } else if (Array.isArray(c)) c.forEach(add);
  };
  fc.features.forEach((f) => add(f.geometry.coordinates));
  return any ? b : null;
}

function reachData(p: ReachMapProps): GeoJSON.FeatureCollection {
  return {
    type: "FeatureCollection",
    features: p.reaches.features.map((f) => {
      const v = p.values.get(f.id);
      const value = v?.value ?? null;
      return {
        type: "Feature",
        id: f.id,
        geometry: f.geometry as GeoJSON.Geometry,
        properties: {
          reach_id: f.id,
          name: f.properties.name ?? "",
          observable: f.properties.observable === true,
          value,
          has_value: value !== null && Number.isFinite(value),
          sounding: value !== null && Number.isFinite(value) ? p.sounding(value) : "",
          spread: v?.spread ?? 0,
          faded: p.visible ? !p.visible.has(f.id) : false,
          highlighted: p.highlighted?.includes(f.id) ?? false,
        },
      };
    }),
  };
}

function pinData(fc: ReachCollection): GeoJSON.FeatureCollection {
  return {
    type: "FeatureCollection",
    features: fc.features.flatMap((f) => {
      const s = f.properties.alert_severity;
      if (s !== "ALERT" && s !== "WATCH") return [];
      const end = downstreamEnd(f);
      return end
        ? [{ type: "Feature" as const, geometry: { type: "Point" as const, coordinates: end }, properties: { reach_id: f.id, severity: s } }]
        : [];
    }),
  };
}

// Point-in-polygon for the lasso (ray casting, screen space).
function inside(pt: [number, number], poly: [number, number][]): boolean {
  let hit = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i];
    const [xj, yj] = poly[j];
    if (yi > pt[1] !== yj > pt[1] && pt[0] < ((xj - xi) * (pt[1] - yi)) / (yj - yi) + xi) hit = !hit;
  }
  return hit;
}

export function ReachMap(props: ReachMapProps) {
  const el = useRef<HTMLDivElement>(null);
  const mapRef = useRef<MLMap | null>(null);
  const [ready, setReady] = useState(false);
  const propsRef = useRef(props);
  propsRef.current = props;
  const [lassoPath, setLassoPath] = useState<[number, number][] | null>(null);

  // ---- create the map once --------------------------------------------------------
  useEffect(() => {
    let disposed = false;
    let map: MLMap | null = null;
    chartPaperStyle().then((style) => {
      if (disposed || !el.current) return;
      const b = bounds(propsRef.current.reaches);
      const mm: MLMap = new maplibregl.Map({
        container: el.current,
        style,
        bounds: propsRef.current.fit === false ? undefined : (b ?? undefined),
        fitBoundsOptions: { padding: 40 },
        attributionControl: { compact: true },
        dragRotate: false,
        pitchWithRotate: false,
        interactive: propsRef.current.interactive !== false,
      });
      map = mm;
      mm.touchZoomRotate.disableRotation();
      if (propsRef.current.interactive !== false) {
        mm.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
        mm.addControl(new maplibregl.ScaleControl({ maxWidth: 90 }), "bottom-right");
      }
      mapRef.current = mm;
      mm.on("error", (e) => console.warn("map:", e.error?.message ?? e));
      mm.on("load", () => {
        mm.addImage("hatch", hatchImage(UNKNOWN));
        setup(mm);
        setReady(true);
        propsRef.current.onMap?.(mm);
      });
    });
    return () => {
      disposed = true;
      map?.remove();
      mapRef.current = null;
    };
  }, []);

  function setup(map: MLMap) {
    map.addSource("catchment-hover", { type: "geojson", data: EMPTY });
    map.addSource("catchment-selected", { type: "geojson", data: EMPTY });
    map.addSource("reaches", { type: "geojson", data: reachData(propsRef.current), promoteId: "reach_id" });
    map.addSource("pins", { type: "geojson", data: pinData(propsRef.current.reaches) });
    map.addSource("exposure", { type: "geojson", data: EMPTY });
    map.addSource("lasso", { type: "geojson", data: EMPTY });

    map.addLayer({ id: "catchment-selected-fill", type: "fill", source: "catchment-selected", paint: { "fill-color": KINGFISHER, "fill-opacity": 0.08 } });
    map.addLayer({ id: "catchment-selected-line", type: "line", source: "catchment-selected", paint: { "line-color": KINGFISHER, "line-width": 1 } });
    map.addLayer({ id: "catchment-hover-fill", type: "fill", source: "catchment-hover", paint: { "fill-color": INK, "fill-opacity": 0.05 } });
    map.addLayer({ id: "catchment-hover-line", type: "line", source: "catchment-hover", paint: { "line-color": INK_MUTED, "line-width": 1, "line-dasharray": [3, 2] } });

    // Exposure: neutral ink, never data-coloured.
    map.addLayer({ id: "exposure-fill", type: "fill", source: "exposure", filter: ["==", ["geometry-type"], "Polygon"], layout: { visibility: "none" }, paint: { "fill-color": INK, "fill-opacity": 0.12, "fill-outline-color": INK_MUTED } });
    map.addLayer({ id: "exposure-line", type: "line", source: "exposure", filter: ["==", ["geometry-type"], "LineString"], layout: { visibility: "none" }, paint: { "line-color": INK_MUTED, "line-width": 1, "line-dasharray": [1, 1.5] } });
    map.addLayer({ id: "exposure-point", type: "circle", source: "exposure", filter: ["==", ["geometry-type"], "Point"], layout: { visibility: "none" }, paint: { "circle-radius": 3, "circle-color": PAPER, "circle-stroke-color": INK, "circle-stroke-width": 1.2 } });
    map.addLayer({
      id: "exposure-label",
      type: "symbol",
      source: "exposure",
      minzoom: 14,
      layout: { visibility: "none", "text-field": ["get", "label"], "text-size": 10, "text-font": ["Noto Sans Regular"], "text-offset": [0, 0.9], "text-anchor": "top" },
      paint: { "text-color": INK_MUTED, "text-halo-color": PAPER, "text-halo-width": 1 },
    });

    // Forecast spread: a soft edge under the line, wider where P10-P90 is wider.
    map.addLayer({
      id: "reach-spread",
      type: "line",
      source: "reaches",
      filter: ["all", ["get", "has_value"], [">", ["get", "spread"], 0]],
      layout: { "line-cap": "round", "line-join": "round" },
      paint: {
        "line-color": rampExpression("value", propsRef.current.breaks) as never,
        "line-width": ["interpolate", ["linear"], ["get", "spread"], 0, 3, 2, 14],
        "line-blur": ["interpolate", ["linear"], ["get", "spread"], 0, 2, 2, 10],
        "line-opacity": 0.35,
      },
    });
    // Selection / scenario casing: kingfisher around the line.
    map.addLayer({
      id: "reach-casing",
      type: "line",
      source: "reaches",
      filter: ["any", ["==", ["get", "reach_id"], ""], ["get", "highlighted"]],
      layout: { "line-cap": "round", "line-join": "round" },
      paint: { "line-color": KINGFISHER, "line-width": 6 },
    });
    // No value: insufficient evidence -> the unsurveyed hatch, not an opacity fade.
    map.addLayer({
      id: "reach-unknown-base",
      type: "line",
      source: "reaches",
      filter: ["!", ["get", "has_value"]],
      layout: { "line-cap": "butt", "line-join": "round" },
      paint: { "line-color": UNKNOWN, "line-width": 4, "line-opacity": ["case", ["get", "faded"], 0.2, 0.55] },
    });
    map.addLayer({
      id: "reach-unknown",
      type: "line",
      source: "reaches",
      filter: ["!", ["get", "has_value"]],
      layout: { "line-cap": "butt", "line-join": "round" },
      paint: { "line-pattern": "hatch", "line-width": 4, "line-opacity": ["case", ["get", "faded"], 0.2, 1] },
    });
    map.addLayer({
      id: "reach-observable",
      type: "line",
      source: "reaches",
      filter: ["all", ["get", "has_value"], ["get", "observable"]],
      layout: { "line-cap": "round", "line-join": "round" },
      paint: { "line-color": rampExpression("value", propsRef.current.breaks) as never, "line-width": 3, "line-opacity": ["case", ["get", "faded"], 0.2, 1] },
    });
    map.addLayer({
      id: "reach-driver",
      type: "line",
      source: "reaches",
      filter: ["all", ["get", "has_value"], ["!", ["get", "observable"]]],
      layout: { "line-cap": "butt", "line-join": "round" },
      paint: { "line-color": rampExpression("value", propsRef.current.breaks) as never, "line-width": 2, "line-dasharray": [3, 2], "line-opacity": ["case", ["get", "faded"], 0.2, 1] },
    });
    // Soundings: the current reading set along the feature, as on a survey chart.
    map.addLayer({
      id: "reach-sounding",
      type: "symbol",
      source: "reaches",
      minzoom: 13.5,
      filter: ["all", ["get", "has_value"], ["!", ["get", "faded"]]],
      layout: { "symbol-placement": "line", "text-field": ["get", "sounding"], "text-size": 10, "text-font": ["Noto Sans Regular"], "symbol-spacing": 220, "text-offset": [0, -0.9] },
      paint: { "text-color": INK, "text-halo-color": PAPER, "text-halo-width": 1.2 },
    });
    // Wide transparent hit area so a 2 px line is clickable.
    map.addLayer({ id: "reach-hit", type: "line", source: "reaches", paint: { "line-color": "#000", "line-opacity": 0, "line-width": 14 } });

    map.addLayer({
      id: "pins",
      type: "circle",
      source: "pins",
      paint: {
        "circle-radius": 5,
        "circle-color": ["match", ["get", "severity"], "ALERT", ALERT, WATCH],
        "circle-stroke-color": PAPER,
        "circle-stroke-width": 1.5,
      },
    });
    map.addLayer({ id: "lasso-fill", type: "fill", source: "lasso", paint: { "fill-color": KINGFISHER, "fill-opacity": 0.06 } });
    map.addLayer({ id: "lasso-line", type: "line", source: "lasso", paint: { "line-color": KINGFISHER, "line-width": 1, "line-dasharray": [2, 1] } });

    map.on("mousemove", "reach-hit", (e: MapLayerMouseEvent) => {
      const id = e.features?.[0]?.properties?.reach_id as string | undefined;
      map.getCanvas().style.cursor = propsRef.current.lasso ? "crosshair" : "pointer";
      if (id && id !== lastHover.current) {
        lastHover.current = id;
        propsRef.current.onHover?.(id);
      }
    });
    map.on("mouseleave", "reach-hit", () => {
      map.getCanvas().style.cursor = propsRef.current.lasso ? "crosshair" : "";
      lastHover.current = null;
      propsRef.current.onHover?.(null);
    });
    map.on("click", (e) => {
      if (propsRef.current.lasso) return;
      const f = map.queryRenderedFeatures(e.point, { layers: ["reach-hit", "pins"] })[0];
      propsRef.current.onClick?.((f?.properties?.reach_id as string) ?? null);
    });
    map.on("mousemove", "exposure-point", (e) => showExposurePopup(map, e));
    map.on("mousemove", "exposure-fill", (e) => showExposurePopup(map, e));
    map.on("mouseleave", "exposure-point", () => popup.current?.remove());
    map.on("mouseleave", "exposure-fill", () => popup.current?.remove());
  }

  const lastHover = useRef<string | null>(null);
  const popup = useRef<maplibregl.Popup | null>(null);
  function showExposurePopup(map: MLMap, e: MapLayerMouseEvent) {
    const p = e.features?.[0]?.properties;
    if (!p) return;
    popup.current ??= new maplibregl.Popup({ closeButton: false, closeOnClick: false, offset: 8 });
    popup.current
      .setLngLat(e.lngLat)
      .setText(`${p.label} — nearest of its type to ${p.reaches === 1 ? "1 reach" : `${p.reaches} reaches`}`)
      .addTo(map);
  }

  // ---- data updates ---------------------------------------------------------------
  const m = ready ? mapRef.current : null;

  useEffect(() => {
    if (!m) return;
    (m.getSource("reaches") as GeoJSONSource).setData(reachData(props));
    (m.getSource("pins") as GeoJSONSource).setData(pinData(props.reaches));
  }, [m, props.reaches, props.values, props.highlighted, props.visible, props.sounding]);

  useEffect(() => {
    if (!m) return;
    for (const id of ["reach-observable", "reach-driver", "reach-spread"])
      m.setPaintProperty(id, "line-color", rampExpression("value", props.breaks) as never);
  }, [m, props.breaks]);

  useEffect(() => {
    if (!m) return;
    m.setFilter("reach-casing", ["any", ["==", ["get", "reach_id"], props.selected ?? "__none__"], ["get", "highlighted"]]);
  }, [m, props.selected]);

  useEffect(() => {
    if (!m) return;
    m.setLayoutProperty("pins", "visibility", props.showPins === false ? "none" : "visible");
  }, [m, props.showPins]);

  // New city: refit.
  useEffect(() => {
    if (!m || props.fit === false) return;
    const b = bounds(props.reaches);
    if (b) m.fitBounds(b, { padding: 40, duration: 0 });
  }, [m, props.reaches.city]);

  // Catchments: hover preview + selected. Fetched on demand, cached for the session.
  useEffect(() => {
    if (!m || props.showCatchments === false) return;
    setCatchment(m, "catchment-hover", props.hovered && props.hovered !== props.selected ? props.hovered : null);
  }, [m, props.hovered, props.selected, props.showCatchments]);
  useEffect(() => {
    if (!m || props.showCatchments === false) return;
    setCatchment(m, "catchment-selected", props.selected ?? null);
  }, [m, props.selected, props.showCatchments]);

  // Exposure layer.
  useEffect(() => {
    if (!m) return;
    const vis = props.showExposure ? "visible" : "none";
    for (const id of ["exposure-fill", "exposure-line", "exposure-point", "exposure-label"]) m.setLayoutProperty(id, "visibility", vis);
    if (!props.showExposure) return;
    const city = props.reaches.city;
    cached(`exposure:${city}`, () => api.exposure(city)).then((layer: ExposureLayer) => {
      (m.getSource("exposure") as GeoJSONSource | undefined)?.setData({
        type: "FeatureCollection",
        features: layer.features.map((f) => ({
          type: "Feature",
          geometry: f.geometry as GeoJSON.Geometry,
          properties: { ...f.properties, label: EXPOSURE_NAMES[f.properties.feature_type] ?? f.properties.feature_type },
        })),
      });
    });
  }, [m, props.showExposure, props.reaches.city]);

  // Lasso.
  useEffect(() => {
    if (!m) return;
    if (props.lasso) {
      m.dragPan.disable();
      m.getCanvas().style.cursor = "crosshair";
    } else {
      m.dragPan.enable();
      m.getCanvas().style.cursor = "";
      (m.getSource("lasso") as GeoJSONSource).setData(EMPTY);
    }
  }, [m, props.lasso]);

  function onPointerDown(e: React.PointerEvent) {
    if (!props.lasso || !m) return;
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    const r = el.current!.getBoundingClientRect();
    setLassoPath([[e.clientX - r.left, e.clientY - r.top]]);
  }
  function onPointerMove(e: React.PointerEvent) {
    if (!lassoPath || !m) return;
    const r = el.current!.getBoundingClientRect();
    const next = [...lassoPath, [e.clientX - r.left, e.clientY - r.top] as [number, number]];
    setLassoPath(next);
    const ring = next.map((p) => m.unproject(p).toArray());
    (m.getSource("lasso") as GeoJSONSource).setData({ type: "Feature", properties: {}, geometry: { type: "Polygon", coordinates: [[...ring, ring[0]]] } });
  }
  function onPointerUp() {
    if (!lassoPath || !m) return;
    const path = lassoPath;
    setLassoPath(null);
    (m.getSource("lasso") as GeoJSONSource).setData(EMPTY);
    if (path.length < 3) return;
    const ids = props.reaches.features
      .filter((f) => {
        const coords: [number, number][] =
          f.geometry.type === "LineString" ? (f.geometry.coordinates as [number, number][]) : (f.geometry.coordinates as [number, number][][]).flat();
        return coords.some((c) => {
          const p = m.project(c);
          return inside([p.x, p.y], path);
        });
      })
      .map((f) => f.id);
    props.onLasso?.(ids);
  }

  return (
    <div className="relative h-full w-full" role="region" aria-label={props.label}>
      {/* maplibre-gl.css sets .maplibregl-map { position: relative } - size it explicitly */}
      <div ref={el} style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }} />
      {props.lasso && (
        <div
          className="absolute inset-0 z-10"
          style={{ cursor: "crosshair", touchAction: "none" }}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          aria-hidden
        />
      )}
      {!ready && <div className="absolute inset-0 grid place-items-center t-ui text-muted">Loading map</div>}
    </div>
  );
}

const EXPOSURE_NAMES: Record<string, string> = {
  school: "School",
  kindergarten: "Kindergarten",
  playground: "Playground",
  park: "Park",
  healthcare: "Healthcare",
  footway: "Footpath",
  cycleway: "Cycleway",
  water_access: "Water access point",
};

const latestCatchment = new WeakMap<MLMap, Record<string, string | null>>();

function setCatchment(m: MLMap, source: string, id: string | null) {
  const src = m.getSource(source) as GeoJSONSource | undefined;
  if (!src) return;
  const latest = latestCatchment.get(m) ?? {};
  latest[source] = id;
  latestCatchment.set(m, latest);
  if (!id) {
    src.setData(EMPTY);
    return;
  }
  cached(`catchment:${id}`, () => api.catchment(id)).then(
    (c: CatchmentFeature) => {
      if (latestCatchment.get(m)?.[source] !== id) return; // the pointer has moved on
      if (c.geometry) src.setData({ type: "Feature", properties: {}, geometry: c.geometry as GeoJSON.Geometry });
      else src.setData(EMPTY);
    },
    () => src.setData(EMPTY),
  );
}

