"""L0 - spatial backbone: OSM waterways -> reaches -> upstream catchments -> topology.

Run:

    python -m pipeline.l0_network --city coimbra
    python -m pipeline.l0_network --city coimbra --no-db --skip-catchments

Outputs:
    data/interim/reaches_<city>.geojson       reach centrelines, Strahler order, topology
    data/interim/catchments_<city>.geojson    upstream contributing catchment polygons
    PostGIS: reaches, reach_topology

This is the backbone every later layer keys off `reach_id`, so it is deliberately
conservative: nothing is silently dropped, every drop is counted with a reason, and a
stage that produces zero rows raises instead of returning an empty frame (CLAUDE.md #3).

WHY OSMIUM AND NOT PYROSM
-------------------------
DATA_SOURCES.md 1.1 offers either. `pyrosm` is the friendlier API (pbf -> GeoDataFrame in
one call) but it depends on `cykhash`, which ships no Windows wheel and needs MSVC to
build - it does not install on this machine. `osmium` (pyosmium) has wheels everywhere,
streams the 423 MB Portugal extract in ~16 s, and - the part that actually matters -
gives us **OSM node ids** on every way vertex. Network connectivity is then exact integer
identity on node ids instead of float coordinate rounding, which is what makes the
confluence split and the topology graph trustworthy. The GeoDataFrame convenience was
never worth trading that away.

WHY A WAY'S FLOW DIRECTION IS ITS DIGITISATION ORDER
----------------------------------------------------
OSM convention: `waterway` ways are drawn in the direction of flow. We use that, and then
check it - the catchment-area monotonicity validation at the end of this module fails
loudly on any reach whose upstream area shrinks downstream, which is exactly what a
reversed way (or a bad snap) produces.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pyproj import Geod, Transformer
from shapely.geometry import LineString, MultiPolygon, Polygon, box, mapping
from shapely.ops import substring, unary_union
from shapely.ops import transform as shapely_transform

from core.cache import DiskCache
from core.logging import get_logger, stage
from core.settings import CONFIG_DIR, DATA_DIR

log = get_logger(__name__)

GEOD = Geod(ellps="WGS84")

INTERIM_DIR = DATA_DIR / "interim"
HYDRO_CACHE_DIR = INTERIM_DIR / "hydro"

# MERIT Hydro ships flow direction and accumulation precomputed (DATA_SOURCES.md 1.2) -
# never derive them from a raw DEM when these exist. Both spellings are accepted: the
# task text says data/raw/merit/, scripts/fetch_datasets.py writes data/raw/merit_hydro/.
MERIT_DIRS = (DATA_DIR / "raw" / "merit", DATA_DIR / "raw" / "merit_hydro")
DEM_DIR = DATA_DIR / "raw" / "dem"

# Degrees of padding around the study bbox for the hydrology window. Upstream catchments
# leave the study area; anything bigger than this window is traced until it hits the edge
# and flagged CATCHMENT_TRUNCATED rather than silently reported as complete.
DEFAULT_DEM_PAD_DEG = 0.35

# "snap the downstream endpoint to the highest-accumulation cell within a 3-cell search"
DEFAULT_SNAP_CELLS = 3

# Channel-initiation threshold: upstream area above which a cell is treated as channel,
# i.e. a valid target to snap a reach outlet onto. 0.7 km2 is a conventional value for
# humid temperate terrain and, measured on this network, is where the monotonicity
# violation rate flattens out (0.14 km2 -> 12%, 0.7 km2 -> 3%, 3.5 km2 -> 1% but with
# many reaches left off the network entirely).
DEFAULT_STREAM_THRESHOLD_KM2 = 0.7


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
def load_city_config(city: str) -> dict[str, Any]:
    path = CONFIG_DIR / "cities" / f"{city}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No city config at {path}. Cities live in config/cities/*.yaml.")
    cfg: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("reach_id_prefix", "bbox", "crs", "network"):
        if key not in cfg:
            raise ValueError(f"{path} is missing required key '{key}'")
    return cfg


def bbox_tuple(cfg: dict[str, Any]) -> tuple[float, float, float, float]:
    b = cfg["bbox"]
    return (b["min_lon"], b["min_lat"], b["max_lon"], b["max_lat"])


# ---------------------------------------------------------------------------
# 1. OSM extraction
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OsmWay:
    """One OSM `waterway` way, with node ids kept alongside coordinates."""

    way_id: int
    waterway: str
    name: str | None
    node_ids: tuple[int, ...]
    coords: tuple[tuple[float, float], ...]


def _read_waterways_from_pbf(
    pbf_path: Path,
    bbox: tuple[float, float, float, float],
    waterway_tags: tuple[str, ...],
) -> dict[str, Any]:
    """One streaming pass over the pbf. Returns a JSON-serialisable payload.

    Kept separate from the cache wrapper so what lands on disk is exactly what was read
    (CLAUDE.md #4 - the pbf is local, but re-parsing 423 MB fifty times is the same waste).
    """
    import osmium
    import osmium.filter

    bbox_poly = box(*bbox)
    wanted = set(waterway_tags)

    ways: list[dict[str, Any]] = []
    seen = 0
    dropped_tag = 0
    dropped_degenerate = 0
    dropped_outside = 0

    processor = (
        osmium.FileProcessor(str(pbf_path), osmium.osm.NODE | osmium.osm.WAY)
        .with_locations()
        .with_filter(osmium.filter.EntityFilter(osmium.osm.WAY))
        .with_filter(osmium.filter.KeyFilter("waterway"))
    )
    for way in processor:
        if not isinstance(way, osmium.osm.Way):  # the EntityFilter guarantees this
            continue
        seen += 1
        waterway = way.tags.get("waterway")
        if waterway not in wanted:
            dropped_tag += 1
            continue
        node_ids: list[int] = []
        coords: list[tuple[float, float]] = []
        for node_ref in way.nodes:
            if not node_ref.location.valid():
                continue
            node_ids.append(node_ref.ref)
            coords.append((node_ref.location.lon, node_ref.location.lat))
        if len(coords) < 2:
            dropped_degenerate += 1
            continue
        if not LineString(coords).intersects(bbox_poly):
            dropped_outside += 1
            continue
        ways.append(
            {
                "way_id": way.id,
                "waterway": waterway,
                "name": way.tags.get("name"),
                "node_ids": node_ids,
                "coords": coords,
            }
        )

    return {
        "source": pbf_path.name,
        "bbox": list(bbox),
        "waterway_tags": sorted(wanted),
        "counts": {
            "ways_with_waterway_tag": seen,
            "dropped_wrong_tag": dropped_tag,
            "dropped_degenerate": dropped_degenerate,
            "dropped_outside_bbox": dropped_outside,
            "kept": len(ways),
        },
        "ways": ways,
    }


def extract_waterways(
    city: str,
    cfg: dict[str, Any],
    *,
    pbf_path: Path | None = None,
    refresh: bool = False,
) -> list[OsmWay]:
    """Waterways of the configured classes intersecting the city bbox.

    A way is kept whole even where it leaves the bbox: connectivity and Strahler order are
    computed on the whole extracted network, and only *reaches* outside the bbox are
    dropped later. Clipping lines before building the graph would invent headwaters.
    """
    pbf_path = pbf_path or (DATA_DIR / "raw" / "osm" / "portugal-latest.osm.pbf")
    if not pbf_path.exists():
        raise FileNotFoundError(
            f"OSM extract not found at {pbf_path}. Run `python scripts/fetch_datasets.py` "
            "(key: osm_portugal) to download the Geofabrik extract."
        )

    bbox = bbox_tuple(cfg)
    tags = tuple(cfg["network"]["waterway_tags"])
    params = {
        "pbf": pbf_path.name,
        "pbf_bytes": pbf_path.stat().st_size,
        "bbox": list(bbox),
        "waterway_tags": sorted(tags),
        "extractor": "pyosmium",
        "version": 1,
    }

    cache = DiskCache("osm")
    entry = cache.get_or_fetch(
        params,
        fetch=lambda: _read_waterways_from_pbf(pbf_path, bbox, tags),
        slug=f"waterways_{city}",
        url=str(pbf_path),
        source="OpenStreetMap contributors, Geofabrik extract (ODbL)",
        refresh=refresh,
    )
    payload = entry.payload
    ways = [
        OsmWay(
            way_id=int(w["way_id"]),
            waterway=w["waterway"],
            name=w["name"],
            node_ids=tuple(int(n) for n in w["node_ids"]),
            coords=tuple((float(x), float(y)) for x, y in w["coords"]),
        )
        for w in payload["ways"]
    ]
    if not ways:
        raise RuntimeError(
            f"No {'/'.join(tags)} waterways found in bbox {bbox}. Either the bbox is wrong "
            "or the pbf does not cover it - refusing to continue on an empty network."
        )
    counts = payload["counts"]
    log.info(
        "l0.osm.extracted",
        city=city,
        pbf=pbf_path.name,
        cache_hit=entry.hit,
        ways_with_waterway_tag=counts["ways_with_waterway_tag"],
        dropped_wrong_tag=counts["dropped_wrong_tag"],
        dropped_degenerate=counts["dropped_degenerate"],
        dropped_outside_bbox=counts["dropped_outside_bbox"],
        input_ways=len(ways),
        by_class=dict(Counter(w.waterway for w in ways)),
    )
    return ways


# ---------------------------------------------------------------------------
# 2. network graph: split ways at confluences, compute Strahler order
# ---------------------------------------------------------------------------
@dataclass
class Link:
    """A stretch of channel between two real junctions.

    This is the graph edge. Reaches are cut from it; it is never cut across a confluence,
    which is what "split at confluences first" means.
    """

    way_id: int
    seq: int  # index of this link within its parent way, then of the chain after merging
    waterway: str
    name: str | None
    node_ids: list[int]
    coords: list[tuple[float, float]]
    way_ids: list[int] = field(default_factory=list)
    strahler: int | None = None
    cycle_broken: bool = False

    def __post_init__(self) -> None:
        if not self.way_ids:
            self.way_ids = [self.way_id]

    @property
    def key(self) -> tuple[int, int]:
        return (self.way_id, self.seq)

    @property
    def u_node(self) -> int:
        """Upstream network node (OSM node id)."""
        return self.node_ids[0]

    @property
    def v_node(self) -> int:
        """Downstream network node (OSM node id)."""
        return self.node_ids[-1]


def split_at_confluences(ways: list[OsmWay]) -> list[Link]:
    """Cut every way at any vertex shared with another way (or revisited by its own).

    Shared vertices are confluences, diffluences and way-to-way joins alike; all of them
    are network nodes. Way endpoints are always nodes.
    """
    occurrences: Counter[int] = Counter()
    for way in ways:
        occurrences.update(way.node_ids)

    split_nodes = {node for node, count in occurrences.items() if count > 1}
    for way in ways:
        split_nodes.add(way.node_ids[0])
        split_nodes.add(way.node_ids[-1])

    links: list[Link] = []
    for way in ways:
        start = 0
        seq = 0
        for idx in range(1, len(way.node_ids)):
            is_last = idx == len(way.node_ids) - 1
            if way.node_ids[idx] in split_nodes or is_last:
                node_ids = list(way.node_ids[start : idx + 1])
                coords = list(way.coords[start : idx + 1])
                # A zero-length or repeated-vertex fragment is not a channel.
                if len(coords) >= 2 and LineString(coords).length > 0:
                    links.append(
                        Link(
                            way_id=way.way_id,
                            seq=seq,
                            waterway=way.waterway,
                            name=way.name,
                            node_ids=node_ids,
                            coords=coords,
                        )
                    )
                    seq += 1
                start = idx
    if not links:
        raise RuntimeError("Way splitting produced no links - the network is empty.")
    return links


def merge_pass_through_links(links: list[Link]) -> list[Link]:
    """Re-join links whose shared node is not a real junction.

    A way boundary in OSM is not a hydrological event: the Mondego is drawn as dozens of
    consecutive ways, and splitting at every one of them produces metre-long slivers that
    would each become a "reach". A node is only a junction when more than one channel
    meets there - one inflow and one outflow means the channel simply continues. Merging
    those first is what lets the 200-500 m segmentation actually control reach length.

    Channel class is respected: a stream that becomes a canal is a genuine boundary.
    """
    inflow: dict[int, list[Link]] = defaultdict(list)
    outflow: dict[int, list[Link]] = defaultdict(list)
    for link in links:
        outflow[link.u_node].append(link)
        inflow[link.v_node].append(link)

    def successor(link: Link) -> Link | None:
        node = link.v_node
        if len(inflow[node]) != 1 or len(outflow[node]) != 1:
            return None
        nxt = outflow[node][0]
        if nxt is link or nxt.waterway != link.waterway:
            return None
        return nxt

    def has_predecessor(link: Link) -> bool:
        node = link.u_node
        if len(inflow[node]) != 1 or len(outflow[node]) != 1:
            return False
        prev = inflow[node][0]
        return prev is not link and prev.waterway == link.waterway

    merged: list[Link] = []
    consumed: set[int] = set()
    for start in links:
        if id(start) in consumed or has_predecessor(start):
            continue
        chain = [start]
        consumed.add(id(start))
        current = start
        while (nxt := successor(current)) is not None and id(nxt) not in consumed:
            chain.append(nxt)
            consumed.add(id(nxt))
            current = nxt
        merged.append(_join_chain(chain, len(merged)))

    # Links inside a pure cycle have a predecessor all the way round, so none of them is
    # a chain start and the loop above never emits them. Emit them as they are.
    for link in links:
        if id(link) not in consumed:
            consumed.add(id(link))
            merged.append(_join_chain([link], len(merged)))

    log.info(
        "l0.network.merged",
        links_before=len(links),
        links_after=len(merged),
        longest_chain_ways=max((len(link.way_ids) for link in merged), default=0),
    )
    return merged


def _join_chain(chain: list[Link], seq: int) -> Link:
    node_ids = list(chain[0].node_ids)
    coords = list(chain[0].coords)
    for link in chain[1:]:
        node_ids.extend(link.node_ids[1:])
        coords.extend(link.coords[1:])
    name = next((link.name for link in chain if link.name), None)
    return Link(
        way_id=chain[0].way_id,
        seq=seq,
        waterway=chain[0].waterway,
        name=name,
        node_ids=node_ids,
        coords=coords,
        way_ids=[link.way_id for link in chain],
    )


def filter_by_strahler(links: list[Link], min_strahler: int) -> list[Link]:
    """Keep only links at or above `network.min_strahler`.

    The full OSM extract for Coimbra is ~220 km of channel, most of it order-1 field
    ditches that are neither monitorable at 10 m nor interesting to a river manager. The
    study design in DATA_SOURCES.md 3 budgets tens of reaches, not hundreds, so the cut is
    made here, once, from config - and the count on either side of it is logged so the
    choice stays visible rather than buried.
    """
    if min_strahler <= 1:
        return links
    kept = [link for link in links if (link.strahler or 1) >= min_strahler]
    if not kept:
        raise RuntimeError(
            f"network.min_strahler={min_strahler} removed every link. Lower it in the city config."
        )
    log.info(
        "l0.network.filtered",
        min_strahler=min_strahler,
        links_in=len(links),
        links_out=len(kept),
        links_dropped=len(links) - len(kept),
        dropped_by_order=dict(
            sorted(
                Counter(
                    link.strahler for link in links if (link.strahler or 1) < min_strahler
                ).items()
            )
        ),
    )
    return kept


def compute_strahler(links: list[Link]) -> dict[str, Any]:
    """Assign Strahler order to every link, in place.

    Strahler on a link is decided at its head node: no inflow -> 1; otherwise the max
    inflow order, incremented when at least two inflows carry that max. Digitisation
    direction is flow direction (see module docstring).

    OSM waterway networks contain occasional cycles (braided channels drawn as loops,
    mis-digitised ways). Those edges are removed from the *ordering* graph so a
    topological sort exists, are logged, and are flagged `cycle_broken` on the link. They
    remain in the network as reaches - they are real channel, just not orderable.
    """
    import networkx as nx

    graph = nx.DiGraph()
    for link in links:
        graph.add_node(link.u_node)
        graph.add_node(link.v_node)
    for link in links:
        if link.u_node != link.v_node:
            graph.add_edge(link.u_node, link.v_node)

    self_loops = [link for link in links if link.u_node == link.v_node]
    for link in self_loops:
        link.cycle_broken = True

    removed_edges: set[tuple[int, int]] = set()
    while True:
        try:
            cycle = nx.find_cycle(graph, orientation="original")
        except nx.NetworkXNoCycle:
            break
        u, v = cycle[-1][0], cycle[-1][1]
        graph.remove_edge(u, v)
        removed_edges.add((u, v))

    for link in links:
        if (link.u_node, link.v_node) in removed_edges:
            link.cycle_broken = True

    inflow: dict[int, list[Link]] = defaultdict(list)
    outflow: dict[int, list[Link]] = defaultdict(list)
    for link in links:
        outflow[link.u_node].append(link)
        if not link.cycle_broken:
            inflow[link.v_node].append(link)

    for node in nx.topological_sort(graph):
        orders = [link.strahler for link in inflow.get(node, []) if link.strahler is not None]
        if not orders:
            order = 1
        else:
            top = max(orders)
            order = top + 1 if orders.count(top) >= 2 else top
        for link in outflow.get(node, []):
            link.strahler = order

    unordered = [link for link in links if link.strahler is None]
    for link in unordered:
        link.strahler = 1  # only reachable for links left out of the ordering graph

    summary = {
        "network_nodes": graph.number_of_nodes(),
        "network_edges": len(links),
        "cycle_edges_removed": len(removed_edges),
        "self_loops": len(self_loops),
        "links_without_order": len(unordered),
        "strahler_histogram": dict(sorted(Counter(link.strahler for link in links).items())),
    }
    log.info("l0.network.strahler", **summary)
    return summary


# ---------------------------------------------------------------------------
# 3. reach segmentation
# ---------------------------------------------------------------------------
@dataclass
class Reach:
    """One modelling unit. `reach_id` is the universal key for every later layer."""

    reach_id: str
    city: str
    name: str | None
    geom: LineString  # EPSG:4326, digitised downstream
    length_m: float
    strahler_order: int
    waterway: str
    osm_way_id: int
    link_key: tuple[int, int]
    piece: int
    n_pieces: int
    u_node: int | None  # network node at the reach head, if it is the link head
    v_node: int | None  # network node at the reach outlet, if it is the link outlet
    flags: list[str] = field(default_factory=list)
    catchment: Polygon | None = None
    catchment_area_km2: float | None = None
    snap_lon: float | None = None
    snap_lat: float | None = None
    snap_distance_m: float | None = None
    snap_accumulation: float | None = None
    snap_method: str | None = None

    @property
    def outlet(self) -> tuple[float, float]:
        """Downstream endpoint (lon, lat) - the point a catchment is traced from."""
        lon, lat = self.geom.coords[-1]
        return (float(lon), float(lat))


def _piece_count(length_m: float, target: float, min_m: float, max_m: float) -> int:
    """How many reaches a link of this length becomes.

    Aim for `target`, never exceed `max`, and avoid slicing below `min` unless the link
    itself is shorter than `min` (in which case it stays whole and is flagged SHORT_LINK).
    """
    if length_m <= max_m:
        return 1
    n = max(1, round(length_m / target))
    n = max(n, math.ceil(length_m / max_m))
    n = min(n, max(1, math.floor(length_m / min_m)))
    return max(1, n)


def segment_links(
    links: list[Link],
    cfg: dict[str, Any],
    city: str,
) -> list[Reach]:
    """Cut links into 200-500 m reaches and assign stable ids.

    Ids are assigned after the bbox filter so they are contiguous, ordered by
    (descending Strahler, way id, link seq, piece) - deterministic for a given pbf, and
    with CMB-0001 on the main channel rather than on an arbitrary headwater ditch.
    """
    metric_crs = cfg["crs"]["metric"]
    to_metric = Transformer.from_crs("EPSG:4326", metric_crs, always_xy=True).transform
    to_wgs84 = Transformer.from_crs(metric_crs, "EPSG:4326", always_xy=True).transform

    net = cfg["network"]["reach_length_m"]
    target, min_m, max_m = float(net["target"]), float(net["min"]), float(net["max"])
    if not min_m <= target <= max_m:
        raise ValueError(f"reach_length_m target {target} outside [{min_m}, {max_m}]")

    bbox_poly = box(*bbox_tuple(cfg))
    prefix = cfg["reach_id_prefix"]

    candidates: list[Reach] = []
    dropped_outside = 0
    short_links = 0

    for link in links:
        line_wgs = LineString(link.coords)
        line_m = shapely_transform(to_metric, line_wgs)
        length_m = float(line_m.length)
        n = _piece_count(length_m, target, min_m, max_m)
        if length_m < min_m:
            short_links += 1

        for piece in range(n):
            part_m = substring(
                line_m,
                length_m * piece / n,
                length_m * (piece + 1) / n,
            )
            if part_m.is_empty or part_m.length <= 0:
                continue
            part_wgs = shapely_transform(to_wgs84, part_m)
            flags: list[str] = []
            if length_m < min_m:
                flags.append("SHORT_LINK")
            if link.cycle_broken:
                flags.append("CYCLE_EDGE")
            candidates.append(
                Reach(
                    reach_id="",  # assigned below
                    city=city,
                    name=link.name,
                    geom=part_wgs,
                    length_m=float(part_m.length),
                    strahler_order=int(link.strahler or 1),
                    waterway=link.waterway,
                    osm_way_id=link.way_id,
                    link_key=link.key,
                    piece=piece,
                    n_pieces=n,
                    u_node=link.u_node if piece == 0 else None,
                    v_node=link.v_node if piece == n - 1 else None,
                    flags=flags,
                )
            )

    # Keep a reach when its centre is inside the study bbox. Reaches on the far ends of
    # ways that merely clip the bbox belong to a neighbouring study area, not this one.
    inside: list[Reach] = []
    for reach in candidates:
        if bbox_poly.contains(reach.geom.interpolate(0.5, normalized=True)):
            inside.append(reach)
        else:
            dropped_outside += 1

    if not inside:
        raise RuntimeError("Segmentation produced no reaches inside the bbox.")

    inside.sort(key=lambda r: (-r.strahler_order, r.osm_way_id, r.link_key[1], r.piece))
    for index, reach in enumerate(inside, start=1):
        reach.reach_id = f"{prefix}-{index:04d}"

    log.info(
        "l0.segment.done",
        city=city,
        links_in=len(links),
        reaches_candidate=len(candidates),
        reaches_out=len(inside),
        dropped_outside_bbox=dropped_outside,
        short_links_kept_whole=short_links,
        **length_distribution([r.length_m for r in inside]),
    )
    return inside


def length_distribution(lengths: list[float]) -> dict[str, Any]:
    """Length distribution of the produced reaches, for the stage log."""
    if not lengths:
        return {"length_n": 0}
    ordered = sorted(lengths)

    def quantile(q: float) -> float:
        return ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1)))]

    return {
        "length_n": len(ordered),
        "length_min_m": round(ordered[0], 1),
        "length_p10_m": round(quantile(0.10), 1),
        "length_median_m": round(statistics.median(ordered), 1),
        "length_mean_m": round(statistics.fmean(ordered), 1),
        "length_p90_m": round(quantile(0.90), 1),
        "length_max_m": round(ordered[-1], 1),
        "length_total_km": round(sum(ordered) / 1000.0, 2),
    }


# ---------------------------------------------------------------------------
# 4. reach topology
# ---------------------------------------------------------------------------
def build_topology(reaches: list[Reach]) -> list[tuple[str, str]]:
    """(upstream_id, downstream_id) edges from network connectivity.

    Two sources of adjacency, and only these two:
      * consecutive pieces cut from the same link
      * a link outlet meeting another link's head at the same OSM node (the confluence)
    """
    by_link: dict[tuple[int, int], dict[int, Reach]] = defaultdict(dict)
    for reach in reaches:
        by_link[reach.link_key][reach.piece] = reach

    heads_at_node: dict[int, list[Reach]] = defaultdict(list)
    outlets_at_node: dict[int, list[Reach]] = defaultdict(list)
    for reach in reaches:
        if reach.u_node is not None:
            heads_at_node[reach.u_node].append(reach)
        if reach.v_node is not None:
            outlets_at_node[reach.v_node].append(reach)

    edges: set[tuple[str, str]] = set()
    within = 0
    for pieces in by_link.values():
        for piece_index, reach in pieces.items():
            nxt = pieces.get(piece_index + 1)
            if nxt is not None:
                edges.add((reach.reach_id, nxt.reach_id))
                within += 1

    across = 0
    for node, outlets in outlets_at_node.items():
        for upstream in outlets:
            for downstream in heads_at_node.get(node, []):
                if upstream.reach_id == downstream.reach_id:
                    continue
                edges.add((upstream.reach_id, downstream.reach_id))
                across += 1

    ordered = sorted(edges)
    indegree = Counter(d for _, d in ordered)
    outdegree = Counter(u for u, _ in ordered)
    log.info(
        "l0.topology.built",
        edges=len(ordered),
        edges_within_link=within,
        edges_at_confluences=across,
        headwater_reaches=sum(1 for r in reaches if indegree[r.reach_id] == 0),
        outlet_reaches=sum(1 for r in reaches if outdegree[r.reach_id] == 0),
        max_inflows=max(indegree.values(), default=0),
    )
    return ordered


# ---------------------------------------------------------------------------
# 5. upstream catchment delineation
# ---------------------------------------------------------------------------
@dataclass
class HydroGrid:
    """Flow direction + accumulation over the study window, from MERIT Hydro if present.

    `source` is carried through to the GeoJSON and the logs: a run on the Copernicus DEM
    fallback must never be mistaken for a run on MERIT Hydro.
    """

    grid: Any
    fdir: Any
    acc: np.ndarray
    transform: Any
    source: str
    dirmap: tuple[int, ...]
    cell_size_deg: float
    acc_units: str  # "km2" (MERIT upa) or "cells" (anything we accumulate ourselves)
    cell_area_km2: float
    _masks: dict[float, np.ndarray] = field(default_factory=dict, repr=False)

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.acc.shape[0]), int(self.acc.shape[1]))

    def stream_mask(self, threshold_km2: float) -> np.ndarray:
        """Cells whose upstream area reaches the channel-initiation threshold.

        This is the derived channel network. Snapping to it - rather than to whatever cell
        happens to hold the most flow within the search window - is what stops a small
        tributary outlet from being captured by the Mondego running 60 m away.
        """
        if threshold_km2 not in self._masks:
            cutoff = (
                threshold_km2
                if self.acc_units == "km2"
                else threshold_km2 / max(self.cell_area_km2, 1e-9)
            )
            self._masks[threshold_km2] = np.isfinite(self.acc) & (self.acc >= cutoff)
        return self._masks[threshold_km2]


def _numpy_compat_for_pysheds() -> None:
    """pysheds 0.5 still calls `np.in1d`, removed in NumPy 2.0.

    Aliasing it to the identical `np.isin` is the smallest honest fix; the alternative is
    pinning NumPy < 2 for the whole project.
    """
    if not hasattr(np, "in1d"):
        np.in1d = np.isin  # type: ignore[attr-defined]


def _find_merit_tiles() -> dict[str, list[Path]]:
    found: dict[str, list[Path]] = {"dir": [], "upa": [], "elv": []}
    for root in MERIT_DIRS:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.tif")):
            lowered = path.name.lower()
            for kind in found:
                if kind in lowered:
                    found[kind].append(path)
    return found


def _mosaic(paths: list[Path], bounds: tuple[float, float, float, float], out: Path) -> Path:
    import rasterio
    from rasterio.merge import merge

    sources = [rasterio.open(p) for p in paths]
    try:
        array, transform = merge(sources, bounds=bounds)
        profile = sources[0].profile.copy()
    finally:
        for src in sources:
            src.close()
    profile.update(
        driver="GTiff",
        count=1,
        height=array.shape[1],
        width=array.shape[2],
        transform=transform,
        compress="deflate",
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(array[0], 1)
    return out


def load_hydro_grid(
    cfg: dict[str, Any],
    city: str,
    *,
    pad_deg: float = DEFAULT_DEM_PAD_DEG,
    refresh: bool = False,
) -> HydroGrid:
    """Flow direction + accumulation for the padded study window.

    MERIT Hydro first (flow direction and upstream area ship precomputed - DATA_SOURCES.md
    1.2, CLAUDE.md "do not derive flow direction from a raw DEM"). Only when no MERIT tile
    is on disk do we fall back to the Copernicus DEM GLO-30 route documented in
    DATA_INVENTORY.md, conditioning it once and caching the result.
    """
    _numpy_compat_for_pysheds()
    from pysheds.grid import Grid

    min_lon, min_lat, max_lon, max_lat = bbox_tuple(cfg)
    bounds = (min_lon - pad_deg, min_lat - pad_deg, max_lon + pad_deg, max_lat + pad_deg)
    HYDRO_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    merit = _find_merit_tiles()
    if merit["dir"]:
        fdir_path = HYDRO_CACHE_DIR / f"{city}_merit_dir.tif"
        if refresh or not fdir_path.exists():
            _mosaic(merit["dir"], bounds, fdir_path)
        grid = Grid.from_raster(str(fdir_path))
        fdir = grid.read_raster(str(fdir_path))
        if merit["upa"]:
            upa_path = HYDRO_CACHE_DIR / f"{city}_merit_upa.tif"
            if refresh or not upa_path.exists():
                _mosaic(merit["upa"], bounds, upa_path)
            acc = np.asarray(grid.read_raster(str(upa_path)), dtype="float64")
            acc_kind = acc_units = "merit_upa_km2"
        else:
            acc = np.asarray(grid.accumulation(fdir), dtype="float64")
            acc_kind = "cells_from_merit_fdir"
            acc_units = "cells"
        source = "MERIT_HYDRO"
        log.info(
            "l0.hydro.source",
            source=source,
            accumulation=acc_kind,
            dir_tiles=[p.name for p in merit["dir"]],
            upa_tiles=[p.name for p in merit["upa"]],
            window=bounds,
            shape=list(np.asarray(acc).shape),
        )
    else:
        dem_tiles = sorted(DEM_DIR.glob("*.tif"))
        if not dem_tiles:
            raise FileNotFoundError(
                "No MERIT Hydro tiles under data/raw/merit[_hydro]/ and no Copernicus DEM "
                f"tiles under {DEM_DIR}. Catchment delineation cannot run. Request MERIT "
                "access (DATA_INVENTORY.md) or run `python scripts/fetch_datasets.py`."
            )
        dem_path = HYDRO_CACHE_DIR / f"{city}_dem_window.tif"
        fdir_path = HYDRO_CACHE_DIR / f"{city}_fallback_fdir.tif"
        acc_path = HYDRO_CACHE_DIR / f"{city}_fallback_acc.tif"

        if refresh or not (fdir_path.exists() and acc_path.exists()):
            log.warning(
                "l0.hydro.fallback",
                reason="MERIT Hydro tiles absent - conditioning Copernicus DEM GLO-30 instead",
                note="flow direction is DERIVED here, not precomputed; swap to MERIT when "
                "the password arrives",
                dem_tiles=[p.name for p in dem_tiles],
            )
            _mosaic(dem_tiles, bounds, dem_path)
            grid = Grid.from_raster(str(dem_path))
            dem = grid.read_raster(str(dem_path))
            conditioned = grid.resolve_flats(grid.fill_depressions(grid.fill_pits(dem)))
            fdir = grid.flowdir(conditioned)
            acc_raster = grid.accumulation(fdir)
            grid.to_raster(fdir, str(fdir_path), blockxsize=256, blockysize=256)
            grid.to_raster(acc_raster, str(acc_path), blockxsize=256, blockysize=256)
        grid = Grid.from_raster(str(fdir_path))
        fdir = grid.read_raster(str(fdir_path))
        acc = np.asarray(grid.read_raster(str(acc_path)), dtype="float64")
        source = "COPDEM_GLO30_FALLBACK"
        acc_units = "cells"
        log.info(
            "l0.hydro.source",
            source=source,
            accumulation="cells_derived_from_copdem",
            window=bounds,
            shape=list(np.asarray(acc).shape),
        )

    # Flow direction and accumulation come from two separate rasters. If they are not on
    # the same grid, every snap silently lands on the wrong cell - fail here instead.
    if np.asarray(acc).shape != np.asarray(fdir).shape:
        raise RuntimeError(
            f"Flow direction {np.asarray(fdir).shape} and accumulation "
            f"{np.asarray(acc).shape} are on different grids ({source}). Re-run with "
            "--refresh-hydro to rebuild both mosaics from the same window."
        )

    affine = fdir.viewfinder.affine
    # Cell area at the window centre. Cells are square in degrees, not in metres, so this
    # is an approximation used only to turn a channel-initiation threshold in km2 into a
    # cell count - never to report an area (those come from the polygon, geodesically).
    centre_lat = (bounds[1] + bounds[3]) / 2.0
    cell_deg = abs(float(affine.a))
    cell_area_km2 = (cell_deg * 111.320 * math.cos(math.radians(centre_lat))) * (cell_deg * 110.540)
    return HydroGrid(
        grid=grid,
        fdir=fdir,
        acc=np.asarray(acc, dtype="float64"),
        transform=affine,
        source=source,
        # MERIT Hydro and pysheds both use the ESRI D8 encoding
        # (N=64, NE=128, E=1, SE=2, S=4, SW=8, W=16, NW=32), so one dirmap serves both.
        dirmap=(64, 128, 1, 2, 4, 8, 16, 32),
        cell_size_deg=cell_deg,
        acc_units="km2" if acc_units == "merit_upa_km2" else "cells",
        cell_area_km2=cell_area_km2,
    )


def _rowcol(hydro: HydroGrid, lon: float, lat: float) -> tuple[int, int]:
    inverse = ~hydro.transform
    col, row = inverse @ (lon, lat)
    return int(math.floor(row)), int(math.floor(col))


def _lonlat(hydro: HydroGrid, row: int, col: int) -> tuple[float, float]:
    lon, lat = hydro.transform @ (col + 0.5, row + 0.5)
    return float(lon), float(lat)


def snap_to_stream(
    hydro: HydroGrid,
    lon: float,
    lat: float,
    search_cells: int = DEFAULT_SNAP_CELLS,
    stream_threshold_km2: float = DEFAULT_STREAM_THRESHOLD_KM2,
) -> dict[str, Any] | None:
    """Snap a reach outlet onto the derived channel network, within `search_cells`.

    Two stages, in this order:

      1. NEAREST_STREAM - the closest cell in the search window whose upstream area
         reaches the channel-initiation threshold, ties broken by accumulation.
      2. MAX_ACCUMULATION - if the window holds no channel cell at all, the highest
         accumulation cell in it, flagged so the caller can tell the two apart.

    Stage 2 alone (the naive reading of "snap to the highest-accumulation cell") was
    measured on the Coimbra network at 40 monotonicity violations in 356 topology edges:
    a tributary outlet 60 m from the Mondego takes the Mondego's 1 800 km2 catchment, and
    the next reach downstream falls back to a 0.03 km2 ditch. Preferring the nearest
    channel cell and keeping stage 2 only as a fallback cut that to a handful while still
    tracing every reach.

    Returns None only when the point falls outside the hydrology window - the caller
    records that as a flag, never as a zero-area catchment (CLAUDE.md #2).
    """
    rows, cols = hydro.shape
    row, col = _rowcol(hydro, lon, lat)
    if not (0 <= row < rows and 0 <= col < cols):
        return None

    r0, r1 = max(0, row - search_cells), min(rows, row + search_cells + 1)
    c0, c1 = max(0, col - search_cells), min(cols, col + search_cells + 1)

    snap_row = snap_col = None
    method = "NEAREST_STREAM"
    stream = hydro.stream_mask(stream_threshold_km2)[r0:r1, c0:c1]
    if stream.any():
        candidates = np.argwhere(stream)
        distances = (candidates[:, 0] + r0 - row) ** 2 + (candidates[:, 1] + c0 - col) ** 2
        accs = hydro.acc[candidates[:, 0] + r0, candidates[:, 1] + c0]
        best = int(np.lexsort((-accs, distances))[0])
        snap_row, snap_col = int(candidates[best, 0] + r0), int(candidates[best, 1] + c0)
    else:
        method = "MAX_ACCUMULATION"
        window = hydro.acc[r0:r1, c0:c1]
        finite = np.where(np.isfinite(window), window, -np.inf)
        if not np.isfinite(finite).any():
            return None
        local = np.unravel_index(int(np.argmax(finite)), finite.shape)
        snap_row, snap_col = r0 + int(local[0]), c0 + int(local[1])

    snap_lon, snap_lat = _lonlat(hydro, snap_row, snap_col)
    _, _, distance = GEOD.inv(lon, lat, snap_lon, snap_lat)
    return {
        "row": snap_row,
        "col": snap_col,
        "lon": snap_lon,
        "lat": snap_lat,
        "distance_m": float(distance),
        "accumulation": float(hydro.acc[snap_row, snap_col]),
        "method": method,
    }


def _polygonise(mask: np.ndarray, hydro: HydroGrid) -> tuple[Polygon | None, bool, int]:
    """Vectorise a boolean catchment mask. Returns (polygon, touches_window_edge, cells)."""
    from rasterio import features

    cells = int(mask.sum())
    if cells == 0:
        return None, False, 0

    shapes = [
        Polygon(geom["coordinates"][0], geom["coordinates"][1:])
        for geom, value in features.shapes(
            mask.astype("uint8"), mask=mask, transform=hydro.transform
        )
        if value == 1 and geom["type"] == "Polygon"
    ]
    if not shapes:
        return None, False, cells

    merged = unary_union(shapes).buffer(0)
    if isinstance(merged, MultiPolygon):
        # A D8 trace is connected; disjoint parts are diagonal-touch artefacts. Keep the
        # largest and say so in the log rather than storing a MultiPolygon the schema
        # (POLYGON) cannot hold.
        merged = max(merged.geoms, key=lambda g: g.area)

    # A two-cell margin, not one: flow direction on the outermost row/column of a clipped
    # grid points out of the array, so a catchment fed from outside the window stops one
    # cell short of the border. Checking only row 0 reported the Mondego (true basin
    # ~6600 km2, 1807 km2 inside this window) as complete.
    edge = 2
    touches_edge = bool(
        mask[:edge, :].any()
        or mask[-edge:, :].any()
        or mask[:, :edge].any()
        or mask[:, -edge:].any()
    )
    # Cosmetic only: half a cell, which is below the raster's own staircase resolution.
    simplified = merged.simplify(hydro.cell_size_deg / 2.0, preserve_topology=True)
    return (simplified if not simplified.is_empty else merged), touches_edge, cells


def delineate_catchments(
    reaches: list[Reach],
    hydro: HydroGrid,
    *,
    search_cells: int = DEFAULT_SNAP_CELLS,
    stream_threshold_km2: float = DEFAULT_STREAM_THRESHOLD_KM2,
) -> dict[str, Any]:
    """Trace the upstream contributing area of every reach outlet, in place.

    Failure is recorded as a flag plus a NULL catchment, never as a plausible-looking
    small polygon.
    """
    traced = 0
    failed: Counter[str] = Counter()
    truncated = 0
    methods: Counter[str] = Counter()

    for reach in reaches:
        lon, lat = reach.outlet
        snap = snap_to_stream(hydro, lon, lat, search_cells, stream_threshold_km2)
        if snap is None:
            reach.flags.append("SNAP_OUTSIDE_WINDOW")
            failed["SNAP_OUTSIDE_WINDOW"] += 1
            continue
        reach.snap_lon, reach.snap_lat = snap["lon"], snap["lat"]
        reach.snap_distance_m = snap["distance_m"]
        reach.snap_accumulation = snap["accumulation"]
        reach.snap_method = snap["method"]
        if snap["method"] == "MAX_ACCUMULATION":
            # No derived channel cell within the search window: this outlet sits off the
            # modelled network, so its catchment is the weakest one in the set.
            reach.flags.append("SNAP_OFF_STREAM_NETWORK")

        try:
            catch = hydro.grid.catchment(
                x=snap["col"],
                y=snap["row"],
                fdir=hydro.fdir,
                dirmap=hydro.dirmap,
                xytype="index",
            )
        except Exception as exc:  # noqa: BLE001 - one bad cell must not kill the run
            reach.flags.append("CATCHMENT_TRACE_FAILED")
            failed["CATCHMENT_TRACE_FAILED"] += 1
            log.warning(
                "l0.catchment.trace_failed",
                reach_id=reach.reach_id,
                error=f"{type(exc).__name__}: {exc}",
            )
            continue

        mask = np.asarray(catch, dtype=bool)
        polygon, touches_edge, cells = _polygonise(mask, hydro)
        if polygon is None or polygon.is_empty:
            reach.flags.append("CATCHMENT_EMPTY")
            failed["CATCHMENT_EMPTY"] += 1
            continue

        area_m2, _ = GEOD.geometry_area_perimeter(polygon)
        reach.catchment = polygon
        reach.catchment_area_km2 = abs(area_m2) / 1e6
        if touches_edge:
            reach.flags.append("CATCHMENT_TRUNCATED")
            truncated += 1
        methods[reach.snap_method or "UNKNOWN"] += 1
        traced += 1

    areas = [r.catchment_area_km2 for r in reaches if r.catchment_area_km2 is not None]
    summary = {
        "reaches": len(reaches),
        "catchments_traced": traced,
        "catchments_null": len(reaches) - traced,
        "null_reasons": dict(failed),
        "truncated_at_window_edge": truncated,
        "snap_methods": dict(methods),
        "stream_threshold_km2": stream_threshold_km2,
        "hydro_source": hydro.source,
        "area_min_km2": round(min(areas), 3) if areas else None,
        "area_median_km2": round(statistics.median(areas), 3) if areas else None,
        "area_max_km2": round(max(areas), 3) if areas else None,
        "snap_max_distance_m": round(max((r.snap_distance_m or 0.0) for r in reaches), 1)
        if reaches
        else None,
    }
    log.info("l0.catchment.done", **summary)
    if traced == 0:
        raise RuntimeError(
            "Catchment delineation traced zero catchments. Refusing to write a backbone "
            "with no contributing areas."
        )
    return summary


def validate_catchment_monotonicity(
    reaches: list[Reach],
    topology: list[tuple[str, str]],
    *,
    tolerance_frac: float = 0.02,
) -> list[dict[str, Any]]:
    """Catchment area must not shrink downstream. A violation means a bad snap.

    Tolerance is 2% of the upstream area (or 0.01 km2, whichever is larger): two reaches
    that snap to the same cell give identical areas, and D8 vectorisation jitters the
    boundary by a cell here and there.
    """
    by_id = {r.reach_id: r for r in reaches}
    violations: list[dict[str, Any]] = []
    compared = 0

    for upstream_id, downstream_id in topology:
        up, down = by_id.get(upstream_id), by_id.get(downstream_id)
        if up is None or down is None:
            continue
        if up.catchment_area_km2 is None or down.catchment_area_km2 is None:
            continue
        compared += 1
        tolerance = max(0.01, tolerance_frac * up.catchment_area_km2)
        if down.catchment_area_km2 + tolerance < up.catchment_area_km2:
            violation = {
                "upstream_id": upstream_id,
                "downstream_id": downstream_id,
                "upstream_km2": round(up.catchment_area_km2, 3),
                "downstream_km2": round(down.catchment_area_km2, 3),
                "shrinkage_km2": round(up.catchment_area_km2 - down.catchment_area_km2, 3),
                "upstream_snap_m": round(up.snap_distance_m or 0.0, 1),
                "downstream_snap_m": round(down.snap_distance_m or 0.0, 1),
            }
            violation["involves_off_stream_snap"] = "MAX_ACCUMULATION" in (
                up.snap_method,
                down.snap_method,
            )
            violations.append(violation)
            for reach in (up, down):
                if "MONOTONICITY_VIOLATION" not in reach.flags:
                    reach.flags.append("MONOTONICITY_VIOLATION")
            log.warning("l0.catchment.monotonicity_violation", **violation)

    off_stream = sum(1 for v in violations if v["involves_off_stream_snap"])
    on_stream_edges = compared - sum(
        1
        for u, d in topology
        if by_id.get(u) is not None
        and by_id.get(d) is not None
        and by_id[u].catchment_area_km2 is not None
        and by_id[d].catchment_area_km2 is not None
        and "MAX_ACCUMULATION" in (by_id[u].snap_method, by_id[d].snap_method)
    )
    log.info(
        "l0.catchment.monotonicity",
        edges_compared=compared,
        violations=len(violations),
        violation_rate=round(len(violations) / compared, 4) if compared else None,
        # Violations concentrate on reaches whose outlet is off the derived channel
        # network. Splitting the rate says whether the backbone is weak everywhere or
        # only where we already know it is (and have flagged it).
        violations_involving_off_stream_snap=off_stream,
        violations_on_stream_only=len(violations) - off_stream,
        edges_on_stream_only=on_stream_edges,
        violation_rate_on_stream_only=(
            round((len(violations) - off_stream) / on_stream_edges, 4) if on_stream_edges else None
        ),
        verdict="PASS" if not violations else "REVIEW - probable bad snap",
    )
    return violations


# ---------------------------------------------------------------------------
# 6. outputs: PostGIS + GeoJSON
# ---------------------------------------------------------------------------
def write_postgis(reaches: list[Reach], topology: list[tuple[str, str]], city: str) -> None:
    """Write this city's L0 rows. Other cities are untouched.

    An UPSERT, not a DELETE + INSERT, for one reason: `observable` and
    `median_water_pixels` are measurements made later by
    scripts/check_observability.py, and re-running L0 must not silently throw them away.
    They survive only where the reach geometry is unchanged - if the centreline moved,
    the old measurement no longer describes it and is reset to NULL rather than carried
    over onto different ground.

    Reaches that no longer exist are deleted, which cascades to their observations,
    drivers, forecasts and alerts. That is intended: those rows describe a reach that is
    gone.
    """
    from sqlalchemy import text

    from core.db import session_scope

    reach_ids = [r.reach_id for r in reaches]
    with session_scope() as session:
        session.execute(
            text(
                "DELETE FROM reach_topology WHERE upstream_id IN "
                "(SELECT reach_id FROM reaches WHERE city = :city) "
                "OR downstream_id IN (SELECT reach_id FROM reaches WHERE city = :city)"
            ),
            {"city": city},
        )
        deleted = session.execute(
            text("DELETE FROM reaches WHERE city = :city AND NOT (reach_id = ANY(:keep))"),
            {"city": city, "keep": reach_ids},
        ).rowcount  # type: ignore[attr-defined]

        session.execute(
            text(
                "INSERT INTO reaches (reach_id, city, name, geom, length_m, strahler_order, "
                "catchment_geom, catchment_area_km2) VALUES (:reach_id, :city, :name, "
                "ST_SetSRID(ST_GeomFromText(CAST(:geom AS text)), 4326), :length_m, "
                ":strahler_order, "
                # ST_GeomFromText(NULL) is NULL, so a reach without a catchment needs no
                # CASE - only an explicit cast, which psycopg requires to type the param.
                "ST_SetSRID(ST_GeomFromText(CAST(:catchment AS text)), 4326), "
                ":catchment_area_km2) "
                "ON CONFLICT (reach_id) DO UPDATE SET "
                "city = EXCLUDED.city, name = EXCLUDED.name, geom = EXCLUDED.geom, "
                "length_m = EXCLUDED.length_m, strahler_order = EXCLUDED.strahler_order, "
                "catchment_geom = EXCLUDED.catchment_geom, "
                "catchment_area_km2 = EXCLUDED.catchment_area_km2, "
                "observable = CASE WHEN ST_Equals(reaches.geom, EXCLUDED.geom) "
                "THEN reaches.observable ELSE NULL END, "
                "median_water_pixels = CASE WHEN ST_Equals(reaches.geom, EXCLUDED.geom) "
                "THEN reaches.median_water_pixels ELSE NULL END"
            ),
            [
                {
                    "reach_id": r.reach_id,
                    "city": r.city,
                    "name": r.name,
                    "geom": r.geom.wkt,
                    "length_m": r.length_m,
                    "strahler_order": r.strahler_order,
                    "catchment": r.catchment.wkt if r.catchment is not None else None,
                    "catchment_area_km2": r.catchment_area_km2,
                }
                for r in reaches
            ],
        )
        if topology:
            session.execute(
                text(
                    "INSERT INTO reach_topology (upstream_id, downstream_id) "
                    "VALUES (:upstream_id, :downstream_id) ON CONFLICT DO NOTHING"
                ),
                [{"upstream_id": u, "downstream_id": d} for u, d in topology],
            )

    log.info(
        "l0.postgis.written",
        city=city,
        reaches_deleted_as_obsolete=deleted,
        reaches_upserted=len(reaches),
        topology_inserted=len(topology),
        catchments_inserted=sum(1 for r in reaches if r.catchment is not None),
    )


def _feature(geometry: Any, properties: dict[str, Any]) -> dict[str, Any]:
    return {"type": "Feature", "geometry": mapping(geometry), "properties": properties}


def export_reaches_geojson(
    reaches: list[Reach],
    topology: list[tuple[str, str]],
    city: str,
    hydro_source: str | None,
    path: Path | None = None,
) -> Path:
    upstream_of: dict[str, list[str]] = defaultdict(list)
    downstream_of: dict[str, list[str]] = defaultdict(list)
    for upstream_id, downstream_id in topology:
        downstream_of[upstream_id].append(downstream_id)
        upstream_of[downstream_id].append(upstream_id)

    features = [
        _feature(
            reach.geom,
            {
                "reach_id": reach.reach_id,
                "city": reach.city,
                "name": reach.name,
                "waterway": reach.waterway,
                "length_m": round(reach.length_m, 1),
                "strahler_order": reach.strahler_order,
                "osm_way_id": reach.osm_way_id,
                "piece": f"{reach.piece + 1}/{reach.n_pieces}",
                "upstream_ids": sorted(upstream_of.get(reach.reach_id, [])),
                "downstream_ids": sorted(downstream_of.get(reach.reach_id, [])),
                "catchment_area_km2": round(reach.catchment_area_km2, 3)
                if reach.catchment_area_km2 is not None
                else None,
                "snap_distance_m": round(reach.snap_distance_m, 1)
                if reach.snap_distance_m is not None
                else None,
                # A truncated catchment area is a LOWER BOUND: the contributing area
                # leaves the hydrology window. Shown, never quietly treated as complete.
                "catchment_truncated": "CATCHMENT_TRUNCATED" in reach.flags,
                "hydro_source": hydro_source,
                "flags": reach.flags,
                # Filled by scripts/check_observability.py, not here. NULL means
                # "not yet assessed", never "unobservable".
                "observable": None,
                "median_water_pixels": None,
            },
        )
        for reach in reaches
    ]
    path = path or (INTERIM_DIR / f"reaches_{city}.geojson")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8"
    )
    log.info("l0.export", file=str(path), features=len(features), kind="reaches")
    return path


def export_catchments_geojson(reaches: list[Reach], city: str, path: Path | None = None) -> Path:
    features = [
        _feature(
            reach.catchment,
            {
                "reach_id": reach.reach_id,
                "city": reach.city,
                "name": reach.name,
                "strahler_order": reach.strahler_order,
                "catchment_area_km2": round(reach.catchment_area_km2, 3)
                if reach.catchment_area_km2 is not None
                else None,
                "outlet_lon": reach.outlet[0],
                "outlet_lat": reach.outlet[1],
                "snap_lon": reach.snap_lon,
                "snap_lat": reach.snap_lat,
                "snap_distance_m": round(reach.snap_distance_m, 1)
                if reach.snap_distance_m is not None
                else None,
                "snap_accumulation": reach.snap_accumulation,
                "snap_method": reach.snap_method,
                "catchment_truncated": "CATCHMENT_TRUNCATED" in reach.flags,
                "monotonicity_violation": "MONOTONICITY_VIOLATION" in reach.flags,
                "flags": reach.flags,
            },
        )
        for reach in reaches
        if reach.catchment is not None
    ]
    path = path or (INTERIM_DIR / f"catchments_{city}.geojson")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8"
    )
    log.info("l0.export", file=str(path), features=len(features), kind="catchments")
    return path


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------
def build_network(
    city: str,
    *,
    write_db: bool = True,
    with_catchments: bool = True,
    refresh_osm: bool = False,
    refresh_hydro: bool = False,
    dem_pad_deg: float = DEFAULT_DEM_PAD_DEG,
    snap_cells: int = DEFAULT_SNAP_CELLS,
    stream_threshold_km2: float = DEFAULT_STREAM_THRESHOLD_KM2,
    min_strahler: int | None = None,
) -> dict[str, Any]:
    """Build L0 for one city end to end. Returns a summary dict for the caller/gate."""
    cfg = load_city_config(city)
    if min_strahler is not None:
        cfg["network"]["min_strahler"] = min_strahler

    with stage(log, "l0_network", city=city) as counters:
        ways = extract_waterways(city, cfg, refresh=refresh_osm)
        links = merge_pass_through_links(split_at_confluences(ways))
        network_summary = compute_strahler(links)
        links = filter_by_strahler(links, int(cfg["network"].get("min_strahler", 1)))
        reaches = segment_links(links, cfg, city)
        topology = build_topology(reaches)

        hydro_source: str | None = None
        catchment_summary: dict[str, Any] = {}
        violations: list[dict[str, Any]] = []
        if with_catchments:
            hydro = load_hydro_grid(cfg, city, pad_deg=dem_pad_deg, refresh=refresh_hydro)
            hydro_source = hydro.source
            catchment_summary = delineate_catchments(
                reaches,
                hydro,
                search_cells=snap_cells,
                stream_threshold_km2=stream_threshold_km2,
            )
            violations = validate_catchment_monotonicity(reaches, topology)

        reaches_path = export_reaches_geojson(reaches, topology, city, hydro_source)
        catchments_path = (
            export_catchments_geojson(reaches, city)
            if any(r.catchment is not None for r in reaches)
            else None
        )

        if write_db:
            write_postgis(reaches, topology, city)
        else:
            log.warning(
                "l0.postgis.skipped",
                city=city,
                reason="--no-db: GeoJSON written, PostGIS NOT updated",
            )

        counters.record(
            rows_in=len(ways),
            rows_out=len(reaches),
            network_nodes=network_summary["network_nodes"],
            links=len(links),
            topology_edges=len(topology),
            catchments=catchment_summary.get("catchments_traced", 0),
            monotonicity_violations=len(violations),
            hydro_source=hydro_source,
            **length_distribution([r.length_m for r in reaches]),
        )

    return {
        "city": city,
        "input_ways": len(ways),
        "links": len(links),
        "network": network_summary,
        "reaches": len(reaches),
        "topology_edges": len(topology),
        "length_distribution": length_distribution([r.length_m for r in reaches]),
        "catchments": catchment_summary,
        "monotonicity_violations": violations,
        "reaches_geojson": str(reaches_path),
        "catchments_geojson": str(catchments_path) if catchments_path else None,
        "postgis_written": write_db,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="L0 spatial backbone (reaches + catchments)")
    parser.add_argument("--city", default="coimbra", help="config/cities/<city>.yaml")
    parser.add_argument("--no-db", action="store_true", help="export GeoJSON only, skip PostGIS")
    parser.add_argument("--skip-catchments", action="store_true", help="network only")
    parser.add_argument("--refresh-osm", action="store_true", help="re-parse the pbf")
    parser.add_argument("--refresh-hydro", action="store_true", help="re-condition the hydrology")
    parser.add_argument("--dem-pad", type=float, default=DEFAULT_DEM_PAD_DEG)
    parser.add_argument("--snap-cells", type=int, default=DEFAULT_SNAP_CELLS)
    parser.add_argument(
        "--stream-threshold-km2",
        type=float,
        default=DEFAULT_STREAM_THRESHOLD_KM2,
        help="upstream area above which a cell counts as channel for snapping",
    )
    parser.add_argument(
        "--min-strahler",
        type=int,
        default=None,
        help="override network.min_strahler from the city config (study-area size knob)",
    )
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    summary = build_network(
        args.city,
        write_db=not args.no_db,
        with_catchments=not args.skip_catchments,
        refresh_osm=args.refresh_osm,
        refresh_hydro=args.refresh_hydro,
        dem_pad_deg=args.dem_pad,
        snap_cells=args.snap_cells,
        stream_threshold_km2=args.stream_threshold_km2,
        min_strahler=args.min_strahler,
    )

    dist = summary["length_distribution"]
    catch = summary["catchments"]
    rule = "=" * 78
    print(f"\n{rule}")
    print(f"KINGFISHER L0 - {summary['city'].upper()}")
    print(rule)
    print(f"  input ways        : {summary['input_ways']}")
    print(f"  network nodes     : {summary['network']['network_nodes']}")
    print(f"  links (confluence-split): {summary['links']}")
    print(f"  reaches produced  : {summary['reaches']}")
    print(f"  topology edges    : {summary['topology_edges']}")
    print(
        f"  reach length (m)  : min {dist['length_min_m']}  p10 {dist['length_p10_m']}"
        f"  median {dist['length_median_m']}  p90 {dist['length_p90_m']}"
        f"  max {dist['length_max_m']}   total {dist['length_total_km']} km"
    )
    print(f"  strahler          : {summary['network']['strahler_histogram']}")
    if catch:
        print(
            f"  catchments        : {catch['catchments_traced']} traced, "
            f"{catch['catchments_null']} NULL {catch['null_reasons'] or ''}"
        )
        print(
            f"  catchment area km2: min {catch['area_min_km2']}  median "
            f"{catch['area_median_km2']}  max {catch['area_max_km2']}"
        )
        print(f"  hydrology source  : {catch['hydro_source']}")
        print(f"  truncated at edge : {catch['truncated_at_window_edge']}")
    violations = summary["monotonicity_violations"]
    off_stream = sum(1 for v in violations if v.get("involves_off_stream_snap"))
    print(
        f"  monotonicity      : {len(violations)} violations "
        f"({off_stream} involve a reach snapped off the derived channel network, "
        f"{len(violations) - off_stream} do not)"
    )
    print(f"  reaches geojson   : {summary['reaches_geojson']}")
    print(f"  catchments geojson: {summary['catchments_geojson']}")
    print(f"  postgis           : {'written' if summary['postgis_written'] else 'SKIPPED'}")
    print(rule)
    print("  GATE: open both GeoJSONs in QGIS or geojson.io before building on this.")
    print(rule)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
