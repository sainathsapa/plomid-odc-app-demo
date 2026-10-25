# PLOMID Cabin UI verification

Updated: 2026-10-24. This verifies template, responsive layout and theme
presentation. It does not certify inference accuracy or replace a target-device
operator trial.

## Checks run

- `PYTHONPATH=. ./.venv/bin/pytest -q -o addopts='' tests/unit/test_console.py tests/unit/test_seat_editor_api.py tests/unit/test_layouts.py tests/integration/test_console_app.py::test_all_screens_200` —
  **35 passed, 1 skipped**. The skipped database integration requires a running
  PLOMID instance via `ODC_TEST_DSN`. Tests cover the full-width Seat map,
  contextual calibration warnings, active-layout controls, editor/API behavior,
  and the dossier redirect while preserving its data API.
- Rendered inline JavaScript on Live, Seat map and Configuration passed
  `node --check` (**10 scripts**).
- Installed headless Chromium, without taking or saving screenshots: Live
  computed to `196px 840px 320px` at 1440px; Seat map and Configuration used
  `196px 1178px`. At 390px, all three checked routes had document width equal
  to the viewport and a 370px content area. The Seat map uses an in-page seat
  detail panel and readable per-seat attribute blocks; its previous narrow
  shared inspector and 10-column attributes table are gone.
- The uncalibrated and proposed-layout notices no longer consume global space.
  Calibration status appears only on Seat map, beside the controls that can
  resolve it. Pilot dossier was removed from navigation and its old page URL
  redirects to Live; `/api/dossier` remains unchanged.
- In Chromium, switching the theme changed the active theme in place. The
  generated cabin SVG was checked in both themes: the child seat color changed
  from `rgb(59, 130, 246)` to the higher-contrast `rgb(36, 95, 158)`; light
  mode also used a white seat outline and dark label text.
- Ruff check and format check passed on changed Python files; `compileall`
  passed for `odc/console`; `git diff --check` passed.

## Scope limits

The browser layout check used the production FastAPI/Jinja templates and CSS
with the repository's test repository fixture. It did not start a live camera
or replay inference session, and did not validate real PLOMID data updates,
operator keyboard use, or physical target-device behavior. The UI change leaves
API, inference, session and persistence behavior unchanged. No screenshots or
new visual-testing dependencies were created.
