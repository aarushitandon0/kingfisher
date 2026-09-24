import type { StyleSpecification } from "maplibre-gl";
import { HAIRLINE, INK_MUTED, PAPER } from "../lib/ramp";

// OpenFreeMap Positron (no key, OSM data), recoloured to chart paper: the basemap is a
// quiet ground so the sediment ramp owns every coloured pixel on the map.
const STYLE_URL = "https://tiles.openfreemap.org/styles/positron";

const PAPER_DEEP = "#E9E6DD";
const LAND_GREEN = "#E3E4D6";
const WATER_BODY = "#D9DFDC";
const ROAD = "#FBFAF6";
const ROAD_CASING = "#DAD6CB";
const BUILDING = "#E4E1D7";

let styleP: Promise<StyleSpecification> | null = null;

export function chartPaperStyle(): Promise<StyleSpecification> {
  styleP ??= fetch(STYLE_URL)
    .then((r) => {
      if (!r.ok) throw new Error(`basemap style: HTTP ${r.status}`);
      return r.json() as Promise<StyleSpecification>;
    })
    .then(recolour)
    .catch((e) => {
      styleP = null;
      console.warn("basemap unavailable, drawing on blank paper", e);
      return blankPaper();
    });
  return styleP;
}

function blankPaper(): StyleSpecification {
  return {
    version: 8,
    glyphs: "https://tiles.openfreemap.org/fonts/{fontstack}/{range}.pbf",
    sources: {},
    layers: [{ id: "background", type: "background", paint: { "background-color": PAPER } }],
  };
}

type AnyLayer = StyleSpecification["layers"][number] & { paint?: Record<string, unknown>; layout?: Record<string, unknown> };

function recolour(style: StyleSpecification): StyleSpecification {
  const layers: AnyLayer[] = [];
  for (const raw of style.layers as AnyLayer[]) {
    const l = { ...raw, paint: { ...(raw.paint ?? {}) }, layout: { ...(raw.layout ?? {}) } } as AnyLayer;
    const id = l.id;
    if (l.type === "raster" || id.startsWith("highway-shield") || id.startsWith("road_shield") || id === "airport" || id === "label_other") continue;
    const p = l.paint!;
    if (l.type === "background") p["background-color"] = PAPER;
    else if (id === "water") {
      p["fill-color"] = WATER_BODY;
      delete p["fill-outline-color"];
    } else if (id === "waterway") {
      // The network we draw IS the waterway layer; the basemap's is a faint guide only.
      p["line-color"] = WATER_BODY;
    } else if (id === "park" || id.startsWith("landcover")) {
      p["fill-color"] = LAND_GREEN;
      p["fill-opacity"] = 0.6;
    } else if (id.startsWith("landuse")) {
      p["fill-color"] = PAPER_DEEP;
      p["fill-opacity"] = 0.6;
    } else if (id === "building") {
      p["fill-color"] = BUILDING;
      delete p["fill-outline-color"];
    } else if (l.type === "line" && (id.includes("casing") || id.startsWith("boundary"))) {
      p["line-color"] = ROAD_CASING;
    } else if (l.type === "line" && (id.startsWith("highway") || id.startsWith("tunnel") || id.startsWith("road") || id.startsWith("aeroway"))) {
      p["line-color"] = ROAD;
    } else if (l.type === "line" && id.startsWith("railway")) {
      p["line-color"] = HAIRLINE;
    } else if (l.type === "fill") {
      p["fill-color"] = PAPER_DEEP;
    }
    if (l.type === "symbol") {
      p["text-color"] = INK_MUTED;
      p["text-halo-color"] = PAPER;
      p["text-halo-width"] = 1.2;
      p["text-opacity"] = 0.7;
      if (id.startsWith("water_name") || id.startsWith("waterway")) p["text-color"] = "#6F7C7C";
    }
    layers.push(l);
  }
  return { ...style, layers: layers as StyleSpecification["layers"] };
}

/** 45 degree hatch as a line-pattern image, transparent between the strokes: drawn over a
 * --unknown base line it gives the INSUFFICIENT_EVIDENCE state (the unsurveyed-area hatch). */
export function hatchImage(color = "#5F5E59", size = 6): { width: number; height: number; data: Uint8Array } {
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
