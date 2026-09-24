import * as maplibregl from "maplibre-gl";
import type { GeoJSONSource, Map as MLMap, MapLayerMouseEvent } from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
// MapLibre 6 finds its worker from a runtime URL the bundler cannot see, so the worker (and
// the shared chunk it imports) is bundled explicitly and handed over.
import workerUrl from "maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url";
import { useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import type { CatchmentFeature, ExposureLayer, ReachCollection, ReachFeature } from "../api/types";
import { Loading } from "../components/bits";
import { cached } from "../lib/useApi";
import { NEGLIGIBLE_DAYS, PALETTE, rampExpression, widthExpression, type Palette, type Scheme } from "../lib/ramp";
import { useTheme } from "../lib/theme";
import { basemapStyle, hatchImage } from "./basemap";

export interface ReachValue {
  value: number | null;
  spread?: number | null;
}

export interface ReachMapProps {
  reaches: ReachCollection;
  /** Per-reach value to colour by. Missing / null -> hatched "no value" state. */
  values: Map<string, ReachValue>;
  breaks: number[];
  /** "exceed": the sequential exceedance ramp. "change": the diverging scenario-change
   * scale; |value| below NEGLIGIBLE_DAYS is drawn as an explicit "negligible" state. */
  scheme?: Scheme;
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

/** Bounds of one reach, for fly-to. */
export function featureBounds(f: ReachFeature): maplibregl.LngLatBounds | null {
  return bounds({ features: [f] });
}

function bounds(fc: Pick<ReachCollection, "features">): maplibregl.LngLatBounds | null {
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
      const has = value !== null && Number.isFinite(value);
      const negligible = p.scheme === "change" && has && Math.abs(value) < NEGLIGIBLE_DAYS;
      return {
        type: "Feature",
        id: f.id,
        geometry: f.geometry as GeoJSON.Geometry,
        properties: {
          reach_id: f.id,
          name: f.properties.name ?? "",
          observable: f.properties.observable === true,
          value,
          has_value: has,
          negligible,
          sounding: has ? p.sounding(value) : "",
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

/** Line widths grow with zoom: 1x at city scale (never under 3 px for a valued reach),
 * about 2x at street scale. Zoom must be the top-level interpolate input. */
function zoomed(expr: unknown): unknown {
  return ["interpolate", ["linear"], ["zoom"], 10, ["*", expr, 1], 13, ["*", expr, 1.3], 16, ["*", expr, 2]];
}

const RESET_ICON =
  '<svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M2.5 6V2.5H6M13.5 6V2.5H10M2.5 10v3.5H6M13.5 10v3.5H10"/></svg>';

/** "Reset view" button under the zoom controls: back to the whole network. */
class ResetControl implements maplibregl.IControl {
  private el: HTMLDivElement | null = null;
  constructor(private readonly onReset: () => void) {}
  onAdd() {
    const el = document.createElement("div");
    el.className = "maplibregl-ctrl maplibregl-ctrl-group";
    const b = document.createElement("button");
    b.type = "button";
    b.title = "Reset to the whole network";
    b.setAttribute("aria-label", "Reset view to the whole stream network");
    b.className = "kf-reset";
    b.innerHTML = RESET_ICON;
    b.onclick = () => this.onReset();
    el.appendChild(b);
    this.el = el;
    return el;
  }
  onRemove() {
    this.el?.remove();
  }
}

/** The middle vertex of a reach: where a "Δ≈0" tag sits (a point, so short reaches still
 * get one - a line-placed label is dropped when the text is longer than the line). */
function midpoint(f: ReachFeature): [number, number] | null {
  const g = f.geometry;
  const pts = g.type === "LineString" ? (g.coordinates as [number, number][]) : g.type === "MultiLineString" ? (g.coordinates as [number, number][][]).flat() : [];
  return pts.length ? pts[Math.floor(pts.length / 2)] : null;
}

function tagData(p: ReachMapProps): GeoJSON.FeatureCollection {
  if (p.scheme !== "change") return EMPTY;
  return {
    type: "FeatureCollection",
    features: p.reaches.features.flatMap((f) => {
      const v = p.values.get(f.id)?.value;
      if (v === null || v === undefined || !Number.isFinite(v) || Math.abs(v) >= NEGLIGIBLE_DAYS) return [];
      const c = midpoint(f);
      return c ? [{ type: "Feature" as const, geometry: { type: "Point" as const, coordinates: c }, properties: { reach_id: f.id } }] : [];
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
  const cameraRef = useRef<{ center: maplibregl.LngLat; zoom: number } | null>(null);
  const markers = useRef<maplibregl.Marker[]>([]);
  const fittedCity = useRef<string | null>(null);
  const [ready, setReady] = useState(false);
  const propsRef = useRef(props);
  propsRef.current = props;
  const [lassoPath, setLassoPath] = useState<[number, number][] | null>(null);
  const theme = useTheme();
  const pal = PALETTE[theme];
  const palRef = useRef<Palette>(pal);
  palRef.current = pal;

  // ---- create the map (again on a theme change: MapLibre cannot restyle in place) --
  useEffect(() => {
    let disposed = false;
    let map: MLMap | null = null;
    const prev = cameraRef.current;
    basemapStyle(theme).then((style) => {
      if (disposed || !el.current) return;
      const b = bounds(propsRef.current.reaches);
      const mm: MLMap = new maplibregl.Map({
        container: el.current,
        style,
        bounds: prev || propsRef.current.fit === false ? undefined : (b ?? undefined),
        center: prev?.center,
        zoom: prev?.zoom,
        fitBoundsOptions: { padding: 48 },
        attributionControl: { compact: true },
        dragRotate: false,
        pitchWithRotate: false,
        interactive: propsRef.current.interactive !== false,
      });
      map = mm;
      mm.touchZoomRotate.disableRotation();
      if (propsRef.current.interactive !== false) {
        mm.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
        mm.addControl(
          new ResetControl(() => {
            const bb = bounds(propsRef.current.reaches);
            if (bb) mm.fitBounds(bb, { padding: 48, duration: 600 });
          }),
          "top-right",
        );
        mm.addControl(new maplibregl.ScaleControl({ maxWidth: 90 }), "bottom-right");
      }
      mapRef.current = mm;
      mm.on("error", (e) => console.warn("map:", e.error?.message ?? e));
      // Draw the network as soon as the style is parsed - "load" would also wait for every
      // basemap and terrain tile.
      let done = false;
      const onStyle = () => {
        if (done || disposed) return;
        done = true;
        mm.addImage("hatch", hatchImage(palRef.current.hatch));
        setup(mm);
        setReady(true);
        propsRef.current.onMap?.(mm);
      };
      if (mm.isStyleLoaded()) onStyle();
      else mm.once("style.load", onStyle);
    });
    return () => {
      disposed = true;
      if (map) cameraRef.current = { center: map.getCenter(), zoom: map.getZoom() };
      markers.current.forEach((mk) => mk.remove());
      markers.current = [];
      map?.remove();
      mapRef.current = null;
      setReady(false);
    };
  }, [theme]);

  function setup(map: MLMap) {
    const c = palRef.current;
    const ramp = rampExpression("value", propsRef.current.breaks, c, propsRef.current.scheme) as never;
    const brks = propsRef.current.breaks;
    const W = (scale = 1) => zoomed(widthExpression("value", brks, scale)) as never;
    map.addSource("catchment-hover", { type: "geojson", data: EMPTY });
    map.addSource("catchment-selected", { type: "geojson", data: EMPTY });
    map.addSource("reaches", { type: "geojson", data: reachData(propsRef.current), promoteId: "reach_id" });
    map.addSource("pins", { type: "geojson", data: pinData(propsRef.current.reaches) });
    map.addSource("exposure", { type: "geojson", data: EMPTY });
    map.addSource("lasso", { type: "geojson", data: EMPTY });
    map.addSource("tags", { type: "geojson", data: tagData(propsRef.current) });

    map.addLayer({ id: "catchment-selected-fill", type: "fill", source: "catchment-selected", paint: { "fill-color": c.brand, "fill-opacity": 0.08 } });
    map.addLayer({ id: "catchment-selected-line", type: "line", source: "catchment-selected", paint: { "line-color": c.brand, "line-width": 1 } });
    map.addLayer({ id: "catchment-hover-fill", type: "fill", source: "catchment-hover", paint: { "fill-color": c.ink, "fill-opacity": 0.05 } });
    map.addLayer({ id: "catchment-hover-line", type: "line", source: "catchment-hover", paint: { "line-color": c.inkMuted, "line-width": 1, "line-dasharray": [3, 2] } });

    // Exposure: neutral ink, never data-coloured.
    map.addLayer({ id: "exposure-fill", type: "fill", source: "exposure", filter: ["==", ["geometry-type"], "Polygon"], layout: { visibility: "none" }, paint: { "fill-color": c.ink, "fill-opacity": 0.12, "fill-outline-color": c.inkMuted } });
    map.addLayer({ id: "exposure-line", type: "line", source: "exposure", filter: ["==", ["geometry-type"], "LineString"], layout: { visibility: "none" }, paint: { "line-color": c.inkMuted, "line-width": 1, "line-dasharray": [1, 1.5] } });
    map.addLayer({ id: "exposure-point", type: "circle", source: "exposure", filter: ["==", ["geometry-type"], "Point"], layout: { visibility: "none" }, paint: { "circle-radius": 3, "circle-color": c.surface, "circle-stroke-color": c.ink, "circle-stroke-width": 1.2 } });
    map.addLayer({
      id: "exposure-label",
      type: "symbol",
      source: "exposure",
      minzoom: 14,
      layout: { visibility: "none", "text-field": ["get", "label"], "text-size": 10, "text-font": ["Noto Sans Regular"], "text-offset": [0, 0.9], "text-anchor": "top" },
      paint: { "text-color": c.inkMuted, "text-halo-color": c.halo, "text-halo-width": 1 },
    });

    // Glow: a soft halo in the line's own colour, so water reads as the focal element even
    // at thumbnail size. Wider where the forecast spread (P10-P90) is wider.
    map.addLayer({
      id: "reach-spread",
      type: "line",
      source: "reaches",
      filter: ["all", ["get", "has_value"], ["!", ["get", "faded"]]],
      layout: { "line-cap": "round", "line-join": "round" },
      paint: {
        "line-color": ramp,
        "line-width": ["interpolate", ["linear"], ["get", "spread"], 0, 9, 2, 18],
        "line-blur": ["interpolate", ["linear"], ["get", "spread"], 0, 6, 2, 12],
        "line-opacity": 0.3,
      },
    });
    // Hover: whatever the pointer is over - on the map, a list, a table - lights up here,
    // so every panel points at the same stream.
    map.addLayer({
      id: "reach-hover",
      type: "line",
      source: "reaches",
      filter: ["==", ["get", "reach_id"], "__none__"],
      layout: { "line-cap": "round", "line-join": "round" },
      paint: { "line-color": c.brandLight, "line-width": zoomed(14) as never, "line-opacity": 0.45, "line-blur": 3 },
    });
    // Selection / scenario casing: kingfisher around the line.
    map.addLayer({
      id: "reach-casing",
      type: "line",
      source: "reaches",
      filter: ["any", ["==", ["get", "reach_id"], ""], ["get", "highlighted"]],
      layout: { "line-cap": "round", "line-join": "round" },
      paint: { "line-color": c.brandLight, "line-width": zoomed(9) as never, "line-opacity": 0.85 },
    });
    // No value: insufficient evidence. Drawn as dim water (it IS a stream) with the
    // unsurveyed hatch over it once zoomed in - never mistaken for a low value.
    map.addLayer({
      id: "reach-unknown-base",
      type: "line",
      source: "reaches",
      filter: ["!", ["get", "has_value"]],
      layout: { "line-cap": "round", "line-join": "round" },
      paint: { "line-color": c.unknown, "line-width": zoomed(2.5) as never, "line-opacity": ["case", ["get", "faded"], 0.25, 0.9] },
    });
    map.addLayer({
      id: "reach-unknown",
      type: "line",
      source: "reaches",
      minzoom: 12,
      filter: ["!", ["get", "has_value"]],
      layout: { "line-cap": "butt", "line-join": "round" },
      paint: { "line-pattern": "hatch", "line-width": zoomed(2.5) as never, "line-opacity": ["case", ["get", "faded"], 0.15, 0.55] },
    });
    map.addLayer({
      id: "reach-observable",
      type: "line",
      source: "reaches",
      filter: ["all", ["get", "has_value"], ["get", "observable"]],
      layout: { "line-cap": "round", "line-join": "round" },
      paint: { "line-color": ramp, "line-width": W(), "line-opacity": ["case", ["get", "faded"], 0.2, 1] },
    });
    map.addLayer({
      id: "reach-driver",
      type: "line",
      source: "reaches",
      filter: ["all", ["get", "has_value"], ["!", ["get", "observable"]]],
      layout: { "line-cap": "butt", "line-join": "round" },
      paint: { "line-color": ramp, "line-width": W(0.8), "line-dasharray": [3, 1.5], "line-opacity": ["case", ["get", "faded"], 0.2, 1] },
    });
    // Change map: a negligible change is a finding, not silence. A light dash over the
    // neutral line plus a "Δ≈0" tag, so the map itself says "this ran, and found ~0".
    map.addLayer({
      id: "reach-negligible",
      type: "line",
      source: "reaches",
      filter: ["get", "negligible"],
      layout: { "line-cap": "butt", "line-join": "round" },
      paint: { "line-color": c.ink, "line-width": zoomed(1.2) as never, "line-dasharray": [1.5, 2.5], "line-opacity": 0.75 },
    });
    map.addLayer({
      id: "reach-negligible-tag",
      type: "symbol",
      source: "tags",
      layout: {
        "text-field": "\u0394\u22480",
        "text-size": 11,
        "text-font": ["Noto Sans Bold"],
        "text-offset": [0, -1.1],
        "text-padding": 6,
      },
      paint: { "text-color": c.ink, "text-halo-color": c.halo, "text-halo-width": 1.6 },
    });
    // Soundings: the current reading set along the feature, as on a survey chart. In the
    // change map they carry the signed delta, so direction never rests on colour alone.
    map.addLayer({
      id: "reach-sounding",
      type: "symbol",
      source: "reaches",
      minzoom: 13,
      filter: ["all", ["get", "has_value"], ["!", ["get", "faded"]], ["!", ["get", "negligible"]]],
      layout: { "symbol-placement": "line", "text-field": ["get", "sounding"], "text-size": 11, "text-font": ["Noto Sans Regular"], "symbol-spacing": 220, "text-offset": [0, -1] },
      paint: { "text-color": c.ink, "text-halo-color": c.halo, "text-halo-width": 1.4 },
    });
    // Wide transparent hit area so a 2 px line is clickable.
    map.addLayer({ id: "reach-hit", type: "line", source: "reaches", paint: { "line-color": "#000", "line-opacity": 0, "line-width": 14 } });

    map.addLayer({
      id: "pins",
      type: "circle",
      source: "pins",
      paint: {
        "circle-radius": ["match", ["get", "severity"], "ALERT", 6, 5],
        "circle-color": ["match", ["get", "severity"], "ALERT", c.alert, c.watch],
        "circle-stroke-color": c.surface,
        "circle-stroke-width": 2,
      },
    });
    map.addLayer({ id: "lasso-fill", type: "fill", source: "lasso", paint: { "fill-color": c.brand, "fill-opacity": 0.06 } });
    map.addLayer({ id: "lasso-line", type: "line", source: "lasso", paint: { "line-color": c.brand, "line-width": 1, "line-dasharray": [2, 1] } });

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
    (m.getSource("tags") as GeoJSONSource).setData(tagData(props));
  }, [m, props.reaches, props.values, props.highlighted, props.visible, props.sounding, props.scheme]);

  useEffect(() => {
    if (!m) return;
    for (const id of ["reach-observable", "reach-driver", "reach-spread"])
      m.setPaintProperty(id, "line-color", rampExpression("value", props.breaks, pal, props.scheme) as never);
    m.setPaintProperty("reach-observable", "line-width", zoomed(widthExpression("value", props.breaks)) as never);
    m.setPaintProperty("reach-driver", "line-width", zoomed(widthExpression("value", props.breaks, 0.8)) as never);
  }, [m, props.breaks, pal, props.scheme]);

  useEffect(() => {
    if (!m) return;
    m.setFilter("reach-casing", ["any", ["==", ["get", "reach_id"], props.selected ?? "__none__"], ["get", "highlighted"]]);
  }, [m, props.selected]);

  // A reach chosen from a list or search may be off-screen: bring it into view. A reach
  // clicked on the map is already visible, so the camera stays put.
  useEffect(() => {
    if (!m || !props.selected) return;
    const f = props.reaches.features.find((x) => x.id === props.selected);
    const b = f ? featureBounds(f) : null;
    if (!b) return;
    const view = m.getBounds();
    if (view.contains(b.getSouthWest()) && view.contains(b.getNorthEast())) return;
    m.fitBounds(b, { padding: 120, maxZoom: 15, duration: 700 });
  }, [m, props.selected]);

  useEffect(() => {
    if (!m) return;
    m.setLayoutProperty("pins", "visibility", props.showPins === false ? "none" : "visible");
  }, [m, props.showPins]);

  // Alert pins get a halo (DOM marker, so CSS can pulse it). Decorative: clicks fall
  // through to the pin layer underneath.
  useEffect(() => {
    if (!m) return;
    markers.current.forEach((mk) => mk.remove());
    markers.current = [];
    if (props.showPins === false) return;
    for (const f of pinData(props.reaches).features) {
      if (f.properties?.severity !== "ALERT") continue;
      const node = document.createElement("div");
      node.className = "kf-pulse";
      node.setAttribute("aria-hidden", "true");
      markers.current.push(new maplibregl.Marker({ element: node }).setLngLat((f.geometry as GeoJSON.Point).coordinates as [number, number]).addTo(m));
    }
  }, [m, props.reaches, props.showPins]);

  // New city: refit.
  useEffect(() => {
    if (!m || props.fit === false) return;
    // A theme rebuild keeps the camera; only a different city refits.
    if (fittedCity.current === null && cameraRef.current) fittedCity.current = props.reaches.city;
    if (fittedCity.current === props.reaches.city) return;
    fittedCity.current = props.reaches.city;
    const b = bounds(props.reaches);
    if (b) m.fitBounds(b, { padding: 40, duration: 0 });
  }, [m, props.reaches.city]);

  useEffect(() => {
    if (!m) return;
    m.setFilter("reach-hover", ["==", ["get", "reach_id"], props.hovered ?? "__none__"]);
  }, [m, props.hovered]);

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
      {!ready && (
        <div className="absolute inset-0 grid place-items-center">
          <Loading what="map" />
        </div>
      )}
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

