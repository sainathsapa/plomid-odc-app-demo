# PLOMID Cabin Design System

## Principles

PLOMID Cabin is an engineering workbench. The interface prioritizes current
state, evidence, recovery and direct manipulation over decorative metrics.
Copper is reserved for active emphasis and primary actions. It is not a safety
or success colour.

## Semantic tokens

The canonical tokens live in `odc/console/static/style.css` and are consumed by
all templates:

- `--ink`, `--ink-2`: application and navigation backgrounds.
- `--graphite`, `--graphite-2`: work surfaces and elevated surfaces.
- `--t-d`, `--t-d2`, `--t-d3`: primary, secondary and metadata text.
- `--line-d`, `--line-d2`: separators and control borders.
- `--accent`, `--accent-2`, `--accent-08`, `--accent-14`: copper interaction
  emphasis.
- `--state-*`: semantic empty, occupied, uncertain, child, restraint, belt and
  unknown states. State colours are always paired with text or shape.
- `--space-*`, `--control-*`, `--shadow-*`, `--motion-*`: shared geometry and
  interaction rhythm.

## Themes

The application supports Light, Dark and System preference. The explicit
preference is stored as `odc-theme-mode`; changing it does not reload the page,
reset the session or clear selection. The bootstrap script applies the resolved
theme before the first paint where the browser permits.

## Components

Use existing shared classes before adding page-specific styling:

- `.card`, `.card-head`: work surfaces and compact section headers.
- `.page-head`, `.eyebrow`, `.sub`: route hierarchy.
- `.status`, `.pill`, `.dot`: state indicators.
- `.empty`, `.notice`, `.error`: non-success states.
- `.btn-primary`, `.ghost`, `.icon-btn`: action hierarchy.
- `.data`, `.mini-table`: structured evidence.
- `.seat`, `.seat-overlay`, `.chip`: direct seat interaction.

Interactive elements must have a visible focus state, a semantic name and an
explicit disabled reason when unavailable. Dynamic processing status uses a
polite live region. `prefers-reduced-motion` disables nonessential transitions.

## Layout conventions

The shared seat inspector appears on Live. Seat map uses the full workspace
for cabin geometry and per-seat attributes, with a contextual selected-seat
detail panel beside the map. Timeline keeps its own event evidence pane;
diagnostics, validation, configuration and fleet use the full content width.
The unused Pilot dossier screen is removed from navigation; its machine-readable
API remains available. The Live route keeps setup available before a run and
tucks it away while processing. Layouts collapse at explicit breakpoints;
narrow navigation scrolls horizontally instead of wrapping into a tall menu.

Light mode uses distinct page, panel and raised-control surfaces, and its map
labels and state colors are adjusted for contrast. Theme changes update chart
text and surfaces in place without navigating or resetting session context.
