# PLOMID Cabin UX Audit

Date: 2026-10-23  
Scope: FastAPI/Jinja operator console, live/replay workflow, seat configuration,
timeline, diagnostics, validation, configuration and assistant surfaces.

## Existing product structure

The console is a server-rendered FastAPI/Jinja application with one shared
shell in `odc/console/templates/base.html`, a semantic CSS layer in
`odc/console/static/style.css`, HTMX partial updates, Alpine controls in the
live setup panel, SSE/polling for live state, and ECharts for timeline and
diagnostic charts. The main routes are Live/Replay, Seat map, Timeline,
Diagnostics, Validation, Config, Pilot dossier and Fleet. The machine benchmark
card screen and its inference-triggering endpoint have been removed from the
operator navigation.

The live path is connected to the actual service hub and API endpoints. Seat
selection is direct and drives the shared inspector. PLOMID remains the
durable boundary; live frame previews are transient and structured evidence is
persisted through the repository.

## Findings

| Priority | Area | Finding | Risk | Resolution / verification |
|---|---|---|---|---|
| P0 | Status | Idle and active status were visually too similar; DB readiness and perception readiness were conflated in the shell. | Operators could interpret an idle session as a database failure. | Show separate service/database semantics; verify with idle, active and failed-DB states. |
| P0 | Live workspace | Primary source controls, video, cabin map, current state, events and evidence compete for attention on dense screens. | Operators spend time scanning instead of investigating. | Establish a three-zone live hierarchy and retain direct seat/event selection. Verify at desktop and laptop widths. |
| P0 | State truth | Status styling relied heavily on colour and compact labels. | Unknown, degraded and unsupported evidence can be misread. | Add semantic status text, icons/labels, focus states and explicit unavailable copy. Verify keyboard and colour-independent reading. |
| P1 | Theme | Theme switching existed but only persisted light/dark and did not expose system mode. | Enterprise installations with OS themes feel inconsistent. | Add a three-state Light/Dark/System preference with first-render initialization. Verify across routes and overlays. |
| P1 | Navigation | The shell exposes many destinations with equal visual weight. | Live operations and configuration are not clearly separated. | Group navigation into Operate, Inspect and Configure while preserving one-click routes. Verify no route becomes hidden. |
| P1 | Layout | The live setup area contains many low-level controls before the operator can focus on the session. | Excessive decisions before Start. | Make source/layout/start the primary controls; keep advanced toggles in disclosure. Verify standard five-seat replay in one setup pass. |
| P1 | Accessibility | Some dynamic live updates and SVG seat controls lacked a consistent status announcement and visible text hierarchy. | Keyboard and assistive-technology users lose context. | Add live-region status, skip link, semantic labels, focus styles and reduced-motion rules. Verify tab workflow. |
| P1 | Responsive behavior | Three-column shell and dense evidence cards collapse late on smaller desktop windows. | Clipping and horizontal scanning during pilot demonstrations. | Use explicit breakpoints and resilient grid sizing. Verify 1280, 1024 and 768 CSS widths. |
| P2 | Visual hierarchy | Repeated uppercase micro-headings and inline styles create a diagnostic rather than product-like rhythm. | Increased cognitive load. | Add shared surface, heading, badge and spacing tokens; remove decorative repetition where safe. |
| P2 | Feedback | Loading and recovery copy is present in parts of the live screen but inconsistent across secondary routes. | Users do not know whether a request is pending or failed. | Standardize empty/loading/error classes and status language. Verify route-by-route. |
| P2 | Performance | Live state uses bounded history and version-gated polling, but all dynamic seat/table DOM is rebuilt on state updates. | Potential avoidable layout work on long sessions. | Keep bounded lists, use contained panels and avoid broad page replacement. Verify no duplicate polling/subscriptions. |

## High-friction workflows before redesign

### Start a standard replay

The operator must understand source type, choose a file or bundled clip, choose
or accept a layout, review optional analysis toggles and press Start. The
standard layout is available, but the setup panel is visually dense and its
advanced controls are adjacent to the primary action.

### Inspect one occupant

The map, seat chips/table, event list and inspector are connected, but the
operator must infer which panel is authoritative. The redesign keeps direct
selection and makes the selected-seat context persistent and explicit.

### Diagnose a failure

Diagnostics and the shell expose health, but a failed database/model/source
state is easier to notice than to recover from. The redesign keeps the failure
visible and adds the next actionable operation without claiming automatic
recovery has succeeded.

## Deliberate non-changes

The UI redesign does not change seat identity, layout inference, inference
thresholds, state reasoning, PLOMID schema, vector semantics, or assistant
truth boundaries. Those are frozen backend contracts documented in
`docs/odc/UI_PHASE_FREEZE.md`.

## Verification plan

- Render every route through the existing TestClient suite.
- Exercise light, dark and system theme persistence without replacing the
  running session.
- Verify direct Seat 1–Seat 7 selection from map, chips, table and events.
- Verify five-seat 2+3 and seven-seat 2+3+2 row geometry at desktop and narrow
  widths.
- Verify idle, running, degraded, empty, loading and replay copy.
- Run full pytest, compilation and diff checks.
- Perform manual browser verification of keyboard focus, reduced motion and
  actual video-to-state behavior on the supported machine.
