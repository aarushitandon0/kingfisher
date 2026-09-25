# Kingfisher — Design System

Visual direction for the frontend. Read alongside `MASTERSPEC.md` §13.

---

## Round 3 (25 Sep 2026) — polish pass: type, feedback, phones

Supersedes Round 2 where they conflict. No new colours or tokens.

- **Type**: page titles 30px (26px on phones) at weight 650, balanced wrapping; stat numbers
  28px/650; eyebrows tracked 0.05em. Heavier than this reads as a poster, not an instrument.
- **Top-bar tabs** are `.nav-tab`: hover tint, active gets a `--brand-50` pill plus a brand
  underline that grows in. Icons drop below 400px so all four tabs fit.
- **Segmented controls** have a sliding thumb (`.seg-thumb`, measured per option).
- **Press feedback**: `.btn` / `.ctl` settle 1px on `:active`. Still no hover lifts.
- **Loading** is shaped like what is coming (`.skeleton`, `<Skeleton>`): stat tiles and
  table lines on Alerts/Validation, the city summary on the map sidebar, a pill on the map.
- **Phones**: Alerts renders the same rows as a stacked list under 768px (no sideways
  table); map overlays shorten labels; the scenario legend collapses by default; every
  wide table sits in its own horizontal scroller inside its card.
- **Very wide screens**: Alerts and Validation cap content at 1760px (`.page-wide`); the map
  views stay full-bleed.

## Round 2 (25 Sep 2026) — map contrast, water-first basemap, dashboard grids, Change map

Supersedes everything below where they conflict.

- **Dark by default** (`lib/theme.ts`); light stays one toggle away. Three darkness steps
  that never collapse: `--chrome-bg #0B0D11` (page, nav) < `--surface #12151B` (cards) <
  `--map-bg #142235` (a navy, never black). The map sits in `.map-frame`: 16px radius,
  hairline ring, deep shadow, page padding around it — it never bleeds into the chrome.
- **Water-first basemap** (`map/basemap.ts`): no hillshade; flat land; roads recede to a
  muted line and minor roads only fade in from z12.5; labels hidden except places and water
  names until zoomed in. Our reaches are 3 px minimum, grow with zoom, and carry a soft glow
  in their own colour. The lowest exceedance band IS water blue (`--exceed-0 #2FA8E0`).
  Reaches with no value are dim water (`--map-water-dim`), hatched from z12.
- **Search and locate**: map search box (reach IDs, stream names, basemap places), reset-view
  control under the zoom buttons, ranked list rows fly to the reach.
- **Every page is a grid of cards**, fluid to the viewport. Alerts: 4 stat tiles, table +
  a side column (mini-map, run breakdown, by variable, reason codes), details in a
  right-hand drawer. Validation: KPI strip, then a 2-column grid (reliability spans two
  rows), skill values green/red, response-check verdicts as badges.
- **Scenario map has three modes**: Baseline | With interventions | **Change** (default
  after a run). Change uses the diverging `--change-*` scale on scenario − baseline days;
  |Δ| < 0.05 d is an explicit state (neutral line, dashed overlay, a `Δ≈0` tag), and an
  on-map notice counts better / worse / Δ≈0 / no estimate — the map itself confirms the
  run, even when the honest answer is ~0.
- **Resizable panes, VS Code style** (`components/Splitter.tsx`): the gap between two panels
  is the drag handle (brand line on hover). Double-click resets, arrows nudge, side panels
  collapse when dragged small; sizes persist per browser. Ctrl+B side panel, Ctrl+J map
  timeline, 1-4 pages, / search, ? shortcut sheet. Hovering a reach in any list or table
  lights it on every map. Stat numbers count up (off under reduced motion).
- **Segmented controls** are pill tracks with the active option filled brand. Buttons are
  36px; the primary button is visibly disabled until the run is possible.

## Visual refresh (25 Sep 2026) — supersedes the sections below where they conflict

The chart-paper look read as a rendered spec, not a monitoring product. The refresh keeps
the concepts (hydrograph rail, hatching = insufficient evidence, mono for measured values,
citations as text) and changes the surface:

- **Tokens live in `frontend/src/theme.css`**; MapLibre's hex twins are `PALETTE` in
  `frontend/src/lib/ramp.ts`. Change both together.
- **Palette is the bird.** Brand = kingfisher cobalt `--brand-500 #1C6FA8` (actions, active
  nav, links, focus, forecast marks). Severity = breast orange/red, used for severity
  only: `--severity-critical #C6432B`, `--severity-watch #D98C2B`, insufficient
  `#9A9CA3`. Text-safe `*-text` twins exist for each.
- **Exceedance ramp** `--exceed-0..4`: `#CBD9E6 #8FB7D6 #E8B24A #D9772E #B8331F`, replacing
  the sediment ramp. Map line width also scales with band (2 → 5 px).
- **Dark mode ships**: follows the system, with a toggle in the top bar (`lib/theme.ts`,
  `data-theme` on `<html>`). The map is rebuilt with the dark palette on a switch.
- **Type**: Inter Tight 600 for display/titles (page titles 34px), Inter for UI, IBM Plex
  Mono for values. Section labels are 11px uppercase eyebrows (`.t-eyebrow`).
- **Surfaces**: panels are `.card` (surface, 10px radius, soft shadow, no border); map
  overlays are `.map-panel` (8px, float shadow). Borders only for same-plane dividers.
- **Severity is a pill** (`.pill-alert/-watch/-insufficient`); table rows keep the 3px
  left edge; stat tiles get a 3px top edge and severity-coloured numbers.
- **Motion**: 150ms fade + 4px rise on route change; 100ms row hover; the alert pin's halo
  pulses twice then settles; the scenario bar descent. All off under reduced motion.
- **Basemap**: Positron recoloured per theme, blue-grey water, faint hillshade from open
  Terrarium DEM tiles.

---

## The brief

**Subject:** a live surveillance instrument for urban streams — it watches water
continuously, warns before it turns, and shows what to change.

**Audience:** researchers, environmental programme managers and a freshwater ecologist
first (the judging panel); city water officers and Local Alliance coordinators second.

**Primary job:** let someone look at a city's streams and understand, in under ten
seconds, which ones are about to be in trouble, how confident we are, and what would help.

This is an **instrument**, not a marketing page and not a SaaS dashboard. It should feel
closer to a tide table or a gauge-station readout than to a startup product tour.

---

## What we are deliberately not doing

The environmental-tech dashboard has a default look, and every climate hackathon entry
arrives wearing it. Avoid all of the following:

- Dark slate background with cyan/teal accents and glassmorphic panels
- Green → amber → red risk ramps
- Content chopped into identical rounded cards with the same soft grey shadow
- Gradient washes used as decoration
- Tracked-out ALL-CAPS eyebrow labels above headings
- Meta strings joined with middle dots (`Coimbra · 40 reaches · Updated 2h ago`)
- A `→` glued to the end of button text
- Warm cream background + high-contrast serif + terracotta accent
- Fade-and-slide-up entrance animation on every section
- Hover lift/scale transitions on every card

If a choice below could have been made for any other project, it's wrong and should be
revised.

---

## Grounding: hydrographic survey charts

The visual vernacular comes from nautical and hydrographic charts — the real cartographic
tradition for representing water measurement. What we borrow, and why it earns its place:

| Chart device | What it does here |
|---|---|
| Pale buff/grey chart paper | low-glare base that lets water data carry all the colour |
| Fine hairline linework | dense information without heavy containers |
| Small dense numerals set along features | soundings; here, current readings along reaches |
| **Diagonal hatching for unsurveyed areas** | our `INSUFFICIENT_EVIDENCE` state |
| Depth-graded blues | our turbidity ramp, inverted (see below) |
| Sparse, functional labelling | no decorative text anywhere |

---

## Colour

### The core insight: the colour IS the measurement

Turbidity is suspended sediment. So the risk ramp runs from **clear water to silt-laden
water** — the actual colour of the thing being measured as it degrades. Not an arbitrary
green-to-red mapping.

This is honest, it's legible without a legend, and it runs along the blue–yellow axis so
it survives red-green colour blindness.

```
--water-clear     #A8C4CC   clear, low turbidity
--water-slight    #C4C2AE   first suspended load
--water-turbid    #C79A5B   ochre — meaningful degradation
--water-heavy     #9A6636   umber — exceedance
--water-severe    #6B4423   deep sediment — sustained exceedance
```

### Base

```
--chart-paper     #F2F0EA   primary background (chart stock, not "cream aesthetic")
--chart-paper-alt #E8E5DC   recessed panels, table stripes
--ink             #1C1F1E   primary text — a true dark green-black, not #111
--ink-muted       #5A605D   secondary text
--hairline        #C8C5BB   all borders and rules, 1px, never heavier
```

### Semantic

```
--alert           #B03A2E   ALERT severity only. Used nowhere else. Ever.
--watch           #C79A5B   WATCH severity (shares the turbidity ramp — intentional)
--unknown         #9B9A94   INSUFFICIENT_EVIDENCE base, always shown with hatching
--kingfisher      #1B6B8C   the single accent: selection, active state, focus rings
--scenario        #4A7C59   scenario "after" state only — the one green in the system
```

`--kingfisher` is the bird's back — a deep cobalt-teal. It appears **only** for
interaction state. It never encodes data. Keeping the accent free of meaning is what lets
the sediment ramp own the data space.

### Dark mode

Not a priority. If time allows: invert to `--ink` ground with the sediment ramp lightened
10%. Do not build it before Day 9.

---

## Typography

**Instrument Sans** (Google Fonts) for interface and prose.
**IBM Plex Mono** for measured values only.

Instrument Sans is a neutral grotesque with slightly open apertures and good density —
it holds up at 12px in dense tables, which is where most of this interface lives. It is
not Inter, which has become the default tell of generated interfaces.

### The mono rule

Monospace for small labels is a known template tell. So it is restricted:

✅ Mono is for **actual measured or computed values**: readings, probabilities,
coordinates, dates, exceedance day counts, distances.
❌ Mono is **never** for labels, headings, buttons, nav, or body prose.

The rule is legible to the user without being told: if it's in mono, a machine measured it.

### Scale

```
display   32 / 36   Instrument Sans   500   -0.02em   reach name, city name
title     20 / 28   Instrument Sans   500   -0.01em   panel headers
body      15 / 24   Instrument Sans   400    0        prose, descriptions
ui        13 / 20   Instrument Sans   450    0        controls, nav, labels
dense     12 / 16   Instrument Sans   450    0.01em   table cells, map labels
reading   28 / 32   IBM Plex Mono     400   -0.01em   the hero number on a reach
value     14 / 20   IBM Plex Mono     400    0        inline measured values
```

Tabular figures on everything numeric: `font-variant-numeric: tabular-nums`.

Sentence case throughout. No all-caps labels anywhere.

---

## Layout

### The concept: map plus hydrograph rail

A conventional monitoring app is sidebar + map. The distinctive move here is making
**time a persistent spatial dimension of the interface** — correct for a forecasting
product, and the thing nobody else will have.

```
┌──────────────────────────────────────────────────────────────────────┐
│ Kingfisher   Coimbra ▾            Map  Alerts  Scenarios  Validation │  56px
├────────────────────────────────────────┬─────────────────────────────┤
│                                        │  Ribeira de Coselhas        │
│                                        │  CMB-0041                   │
│              MAP                       │                             │
│   stream network, sediment-ramped      │  0.71                       │
│   catchment on hover                   │  exceedance probability     │
│   hatched = insufficient evidence      │  23–25 Sep                  │
│   alert pins                           │                             │
│                                        │  ── driver attribution ──   │
│                                        │  dry days      ▇▇▇▇▇▇ +0.31 │
│                                        │  forecast rain ▇▇▇▇▇  +0.28 │
│                                        │  imperviousness▇▇▇    +0.19 │
│                                        │                             │
│                                        │  ── exposure ──             │
│                                        │  school       180 m         │
│                                        │  access points      3       │
│                                        │                             │
│                                        │  ◆ optically observable     │
├────────────────────────────────────────┴─────────────────────────────┤
│  HYDROGRAPH RAIL                                              ◀ ▶    │
│  ╭──────────────────────────────── now ─────╮                        │  140px
│  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~╱▒▒▒▒▒▒▒▒▒▒▒▒▒▒          │
│  observed history                      forecast fan P10–P90          │
│  Jun        Jul        Aug        Sep  ┃  Oct                        │
└──────────────────────────────────────────────────────────────────────┘
```

**Alignment:** left throughout. Numbers right-aligned in tables. Nothing centred except
the empty state.

**Density:** high. This is an instrument for people who read data. Generous whitespace
would be a misread of the audience — tighten to 8px base spacing, not 16.

---

## The signature interaction

**The hydrograph rail is where the design spends its boldness. Everything else stays quiet.**

- Persistent strip along the bottom edge, ~140px
- Shows the selected reach's observed history and forecast on one continuous axis
- A draggable head marks the rendered date
- **Drag the head and the map re-renders at that date** — you can scrub backwards through
  the last decade of a city's streams
- Push past `now` and the rail widens into the P10–P90 fan; the map's reach colours become
  the median prediction and gain a soft edge indicating spread
- Gaps in observation are rendered as gaps — never interpolated lines across missing data

This makes the core scientific claim of the project visible as an interaction: we know the
past precisely, we know the future with widening uncertainty, and we never pretend
otherwise.

### Motion

One orchestrated moment only: when a scenario runs, the exceedance-day bars descend from
baseline to scenario over ~600ms with a single ease. That's it.

Everything else is user-triggered state change: drawer opens, hatching appears, the rail
head follows the cursor. No entrance animations. No hover lifts. Respect
`prefers-reduced-motion` on the one animation that exists.

---

## Component specifications

### Reach rendering on the map

| State | Treatment |
|---|---|
| Normal | 3px line, colour from the sediment ramp at current/predicted value |
| Unobservable (driver-predicted) | same colour, 2px, dashed 6/4 |
| Insufficient evidence | `--unknown` with 45° diagonal hatch overlay |
| Selected | `--kingfisher` 1px casing around the line, catchment polygon at 8% fill |
| Under alert | small pin at the downstream end, `--alert` |

The hatching is an SVG pattern fill, not an opacity reduction. It must read as a
deliberate cartographic state, not as a disabled element.

### The reading

The hero of the reach panel is one number in Plex Mono at 28px with a plain sentence-case
label beneath. No card, no border, no gradient. The number sits directly on the paper.

### Alerts list

A table, not cards. Hairline row separators. Columns: reach, severity, probability, window,
exposure summary. Severity is a 3px left border in the semantic colour, not a pill badge.

`INSUFFICIENT_EVIDENCE` rows appear inline with a hatched left border — **not filtered out,
not greyed to invisibility.** They are part of the system's output.

### Scenario workbench

Before/after as a **vertical swipe divider on a single map**, not side-by-side panes. Drag
the divider left to right to wipe between baseline and scenario. Shared geography, so the
eye compares colour directly rather than trying to align two maps.

Beneath: the hydrograph rail shows both curves overlaid, baseline in the sediment ramp,
scenario in `--scenario`.

Results strip: `23.4 → 14.1 exceedance days/year`, with intervals in mono, and every
coefficient's citation shown as visible text beneath — not in a tooltip, not behind an
info icon. The citations are a feature, so display them.

Caveat banner is persistent, in `--ink-muted` on `--chart-paper-alt`, one sentence:
*Planning estimate from cited literature applied to a statistical model. Not a causal
experiment.*

### Validation page

Plain charts on paper. No chart junk, no gridline decoration. The reliability diagram gets
the largest allocation — it's the most credible thing in the submission.

Where the model loses to a baseline, that row is shown in the same weight as the others.
No red, no apology, no visual softening. A table that doesn't flinch is more persuasive
than one that hides.

---

## Copy

Plain, active, specific. The interface never sells.

| Not this | This |
|---|---|
| "AI-Powered Stream Intelligence" | "Coimbra — 40 reaches" |
| "No data available" | "No usable observation since 2 Sep. Cloud cover on 6 of 8 passes." |
| "Submit" | "Run scenario" |
| "Risk Level: HIGH" | "Exceedance probability 0.71 for 23–25 Sep" |
| "Oops! Something went wrong." | "Forecast unavailable — model version 0.3.1 has no output for this reach." |

Empty and refusal states explain *what's missing and why*, and offer the next action.
`INSUFFICIENT_EVIDENCE` is the most important copy in the product — write it as a finding,
not as a failure:

> Not enough recent observation to issue an alert. Last usable reading 24 days ago;
> this reach is 4m wide and optically unobservable at 10m resolution.

---

## Quality floor

Build these without announcing them:

- Responsive to 768px (the map collapses the panel to a bottom sheet). Below that, list view.
- Visible keyboard focus using `--kingfisher`, 2px offset ring. Never `outline: none`.
- `prefers-reduced-motion` respected on the scenario animation.
- All colour pairings ≥ 4.5:1 for text. Check the ochre range against paper — it's the
  one that will fail.
- The sediment ramp never carries meaning alone: severity is also in the label and in the
  line treatment.
- Map is keyboard-navigable: reaches reachable by tab, arrow keys move the rail head.

---

## Before you ship: remove one accessory

On Day 8, look at the whole interface and cut the least necessary visual element. The
likely candidates: a redundant legend, a second accent colour that crept in, a border
around something that didn't need one. Restraint is what makes the hydrograph rail land.