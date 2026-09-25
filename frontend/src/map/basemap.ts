import type { StyleSpecification } from "maplibre-gl";
import { PALETTE, type Palette, type Theme } from "../lib/ramp";

// OpenFreeMap Positron (no key, OSM data), restyled water-first. Most basemaps are drawn for
// navigation; this one is drawn for hydrology:
//  - land is one flat, warm neutral fill (no hillshade), a different hue from the chrome
//    so the map reads as its own surface in both themes
//  - roads recede to a muted line, and minor roads only fade in once zoomed in
//  - labels are hidden except places and water names until zoomed in
//  - water polygons and the basemap's own waterways read as water; our reaches draw on top
//    and, with the severity colours, are the only saturated marks.
const STYLE_URL = "https://tiles.openfreemap.org/styles/positron";

const raw: { p: Promise<StyleSpecification | null> | null } = { p: null };
const styles = new Map<Theme, Promise<StyleSpecification>>();

function fetchRaw(): Promise<StyleSpecification | null> {
  raw.p ??= fetch(STYLE_URL)
    .then((r) => {
      if (!r.ok) throw new Error(`basemap style: HTTP ${r.status}`);
      return r.json() as Promise<StyleSpecification>;
    })
    .catch((e) => {
      raw.p = null;
      console.warn("basemap unavailable, drawing on a blank ground", e);
      return null;
    });
  return raw.p;
}

export function basemapStyle(theme: Theme): Promise<StyleSpecification> {
  let s = styles.get(theme);
  if (!s) {
    const pal = PALETTE[theme];
    s = fetchRaw().then((st) => {
      if (!st) {
        styles.delete(theme);
        return blank(pal);
      }
      return restyle(st, pal);
    });
    styles.set(theme, s);
  }
  return s;
}

function blank(p: Palette): StyleSpecification {
  return {
    version: 8,
    glyphs: "https://tiles.openfreemap.org/fonts/{fontstack}/{range}.pbf",
    sources: {},
    layers: [{ id: "background", type: "background", paint: { "background-color": p.bg } }],
  };
}

type AnyLayer = StyleSpecification["layers"][number] & {
  paint?: Record<string, unknown>;
  layout?: Record<string, unknown>;
  minzoom?: number;
};

/** Fade a layer in between two zooms instead of popping it in. */
const fadeIn = (z0: number, z1: number, to: number) => ["interpolate", ["linear"], ["zoom"], z0, 0, z1, to];

// Dropped outright: relief, shields, airports, minor place classes, casings (a flat map has
// no need for road outlines), sub-national boundaries.
const DROP = /^(highway-shield|road_shield|airport|label_other|label_state|label_country|boundary_3|boundary_disputed|aeroway)|casing/;

function restyle(style: StyleSpecification, c: Palette): StyleSpecification {
  const layers: AnyLayer[] = [];
  for (const src of style.layers as AnyLayer[]) {
    const id = src.id;
    if (src.type === "raster" || src.type === "hillshade" || DROP.test(id)) continue;
    const l = { ...src, paint: { ...(src.paint ?? {}) }, layout: { ...(src.layout ?? {}) } } as AnyLayer;
    const p = l.paint!;
    const lay = l.layout!;

    if (l.type === "background") p["background-color"] = c.bg;
    else if (id === "water") {
      p["fill-color"] = c.water;
      p["fill-opacity"] = 1;
      delete p["fill-outline-color"];
    } else if (id === "waterway") {
      // The basemap's own streams: a guide under our network, still clearly water.
      p["line-color"] = c.waterway;
      p["line-opacity"] = 0.8;
    } else if (id === "park" || id.startsWith("landcover")) {
      p["fill-color"] = c.park;
      p["fill-opacity"] = 0.85;
    } else if (id.startsWith("landuse")) {
      p["fill-color"] = c.landuse;
      p["fill-opacity"] = 0.8;
    } else if (id === "building") {
      p["fill-color"] = c.building;
      p["fill-opacity"] = fadeIn(13, 15, 0.9);
      delete p["fill-outline-color"];
    } else if (id === "road_area_pier") {
      p["fill-color"] = c.bg;
    } else if (l.type === "line" && id.startsWith("boundary")) {
      p["line-color"] = c.boundary;
      p["line-opacity"] = 0.6;
    } else if (l.type === "line" && id.startsWith("railway")) {
      p["line-color"] = c.rail;
      p["line-opacity"] = 0.35;
    } else if (l.type === "line" && (id === "highway_minor" || id === "highway_path" || id === "road_pier")) {
      // Minor roads fade in on zoom-in; at city scale they would compete with the water.
      l.minzoom = 12.5;
      p["line-color"] = c.road;
      p["line-opacity"] = fadeIn(12.5, 14.5, c.roadOpacity * 0.7);
    } else if (l.type === "line" && (id.startsWith("highway") || id.startsWith("tunnel") || id.startsWith("road"))) {
      p["line-color"] = c.road;
      p["line-opacity"] = c.roadOpacity;
    } else if (l.type === "fill") {
      p["fill-color"] = c.bg;
    }

    if (l.type === "symbol") {
      const water = id.startsWith("water_name") || id.startsWith("waterway");
      const major = id === "label_city" || id === "label_city_capital" || id === "label_town";
      p["text-halo-color"] = c.halo;
      p["text-halo-width"] = 1.4;
      if (water) {
        p["text-color"] = c.waterLabel;
        p["text-opacity"] = 0.95;
      } else if (major) {
        p["text-color"] = c.label;
        p["text-opacity"] = 0.9;
      } else {
        // Villages, street names: hidden until zoomed in, and quieter than the water.
        l.minzoom = Math.max(l.minzoom ?? 0, id.startsWith("highway-name") ? 15 : 13);
        p["text-color"] = c.labelMinor;
        p["text-opacity"] = 0.8;
      }
      delete lay["icon-image"];
    }
    layers.push(l);
  }
  return { ...style, layers: layers as StyleSpecification["layers"] };
}

/** 45 degree hatch as a line-pattern image, transparent between the strokes: drawn over a
 * dim-water base line it gives the INSUFFICIENT_EVIDENCE state (the unsurveyed-area hatch). */
export function hatchImage(color: string, size = 6): { width: number; height: number; data: Uint8Array } {
  const data = new Uint8Array(size * size * 4);
  const [r, g, b] = [1, 3, 5].map((i) => parseInt(color.slice(i, i + 2), 16));
  for (let y = 0; y < size; y++)
    for (let x = 0; x < size; x++) {
      const on = (x + y) % size < 2;
      const o = (y * size + x) * 4;
      data[o] = r;
      data[o + 1] = g;
      data[o + 2] = b;
      data[o + 3] = on ? 255 : 0;
    }
  return { width: size, height: size, data };
}
