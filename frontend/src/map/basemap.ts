import type { StyleSpecification } from "maplibre-gl";
import { PALETTE, type Palette, type Theme } from "../lib/ramp";

// OpenFreeMap Positron (no key, OSM data), recoloured: desaturated land, water that reads
// as water, and a faint hillshade for relief. The basemap stays quiet so the exceedance
// ramp owns the saturated pixels on the map.
const STYLE_URL = "https://tiles.openfreemap.org/styles/positron";
// Open terrain tiles (Mapzen Terrarium on AWS Open Data), no key.
const DEM_TILES = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png";

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
      return recolour(st, pal);
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

type AnyLayer = StyleSpecification["layers"][number] & { paint?: Record<string, unknown>; layout?: Record<string, unknown> };

function recolour(style: StyleSpecification, c: Palette): StyleSpecification {
  const layers: AnyLayer[] = [];
  let hillshadeAt = -1;
  for (const src of style.layers as AnyLayer[]) {
    const l = { ...src, paint: { ...(src.paint ?? {}) }, layout: { ...(src.layout ?? {}) } } as AnyLayer;
    const id = l.id;
    if (l.type === "raster" || id.startsWith("highway-shield") || id.startsWith("road_shield") || id === "airport" || id === "label_other") continue;
    const p = l.paint!;
    if (l.type === "background") p["background-color"] = c.bg;
    else if (id === "water") {
      p["fill-color"] = c.water;
      delete p["fill-outline-color"];
      // Relief goes under the water and everything built.
      if (hillshadeAt < 0) hillshadeAt = layers.length;
    } else if (id === "waterway") {
      // The network we draw IS the waterway layer; the basemap's is a guide only.
      p["line-color"] = c.waterway;
    } else if (id === "park" || id.startsWith("landcover")) {
      p["fill-color"] = c.park;
      p["fill-opacity"] = 0.7;
    } else if (id.startsWith("landuse")) {
      p["fill-color"] = c.landuse;
      p["fill-opacity"] = 0.6;
    } else if (id === "building") {
      p["fill-color"] = c.building;
      delete p["fill-outline-color"];
    } else if (l.type === "line" && (id.includes("casing") || id.startsWith("boundary"))) {
      p["line-color"] = c.roadCasing;
    } else if (l.type === "line" && (id.startsWith("highway") || id.startsWith("tunnel") || id.startsWith("road") || id.startsWith("aeroway"))) {
      p["line-color"] = c.road;
    } else if (l.type === "line" && id.startsWith("railway")) {
      p["line-color"] = c.rail;
    } else if (l.type === "fill") {
      p["fill-color"] = c.land;
    }
    if (l.type === "symbol") {
      p["text-color"] = c.label;
      p["text-halo-color"] = c.bg;
      p["text-halo-width"] = 1.2;
      p["text-opacity"] = 0.8;
      if (id.startsWith("water_name") || id.startsWith("waterway")) p["text-color"] = c.waterLabel;
    }
    layers.push(l);
  }
  const hillshade = {
    id: "hillshade",
    type: "hillshade",
    source: "dem",
    maxzoom: 16,
    paint: {
      "hillshade-exaggeration": 0.25,
      "hillshade-shadow-color": c.shadow,
      "hillshade-highlight-color": c.highlight,
      "hillshade-accent-color": c.shadow,
      "hillshade-illumination-anchor": "map",
    },
  } as unknown as AnyLayer;
  layers.splice(hillshadeAt < 0 ? 1 : hillshadeAt, 0, hillshade);
  return {
    ...style,
    sources: {
      ...style.sources,
      dem: {
        type: "raster-dem",
        tiles: [DEM_TILES],
        encoding: "terrarium",
        tileSize: 256,
        maxzoom: 14,
        attribution: "Terrain: Mapzen / AWS Open Data",
      },
    },
    layers: layers as StyleSpecification["layers"],
  };
}

/** 45 degree hatch as a line-pattern image, transparent between the strokes: drawn over a
 * grey base line it gives the INSUFFICIENT_EVIDENCE state (the unsurveyed-area hatch). */
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
