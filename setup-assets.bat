@echo off
rem ===========================================================================
rem  PLOMID ODC - one-time runtime asset setup for Windows.
rem
rem    setup-assets.bat
rem
rem  Downloads and verifies everything the console needs outside PLOMID:
rem    - RT-DETRv2-S detector checkpoint + ONNX export
rem    - MiVOLO V2 age/gender checkpoint
rem    - MediaPipe pose + face landmarker task bundles
rem    - the eight staged footage clips in footage/README.md
rem
rem  Re-running is safe: verified files are kept, only missing ones are fetched.
rem  Requires the Python environment used by start.bat (local .venv, ..\.venv,
rem  py -3.12 or python).
rem ===========================================================================
setlocal EnableExtensions
cd /d "%~dp0"
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

rem --- resolve Python 3.12+ ------------------------------------------------
set "PY="
if exist "%ROOT%\.venv\Scripts\python.exe" set "PY=%ROOT%\.venv\Scripts\python.exe"
if not defined PY if exist "%ROOT%\..\.venv\Scripts\python.exe" set "PY=%ROOT%\..\.venv\Scripts\python.exe"
if not defined PY (
  for /f "usebackq delims=" %%P in (`py -3.12 -c "import sys;print(sys.executable)" 2^>nul`) do set "PY=%%P"
)
if not defined PY (
  for /f "usebackq delims=" %%P in (`python -c "import sys;print(sys.executable)" 2^>nul`) do set "PY=%%P"
)
if not defined PY (
  echo [odc] error: Python 3.12 was not found.
  echo [odc]        create a venv with: py -3.12 -m venv .venv
  exit /b 1
)
"%PY%" -c "import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 12) else 1)" >nul 2>&1
if errorlevel 1 (
  echo [odc] error: Python 3.12 or newer is required: %PY%
  exit /b 1
)

rem --- Python runtime dependencies -----------------------------------------
"%PY%" -c "import importlib.util as u; m=['psycopg','fastapi','uvicorn','jinja2','yaml','pydantic_settings','numpy','cv2','onnxruntime','mediapipe','torch','transformers','huggingface_hub','onnx']; miss=[x for x in m if u.find_spec(x) is None]; print('missing: '+', '.join(miss) if miss else ''); raise SystemExit(1 if miss else 0)"
if errorlevel 1 (
  echo [odc] installing requirements.lock, skipping uvloop on Windows...
  powershell -NoProfile -Command "Get-Content -LiteralPath '%ROOT%\requirements.lock' | Where-Object { $_ -notmatch '^uvloop' } | Set-Content -Encoding ASCII -LiteralPath $env:TEMP\plomid-odc-requirements-windows.txt"
  if errorlevel 1 (
    echo [odc] error: could not prepare Windows requirements.
    exit /b 1
  )
  "%PY%" -m pip install --disable-pip-version-check -r "%TEMP%\plomid-odc-requirements-windows.txt"
  if errorlevel 1 (
    echo [odc] error: dependency installation failed.
    exit /b 1
  )
)

rem --- model checkpoints + ONNX export -------------------------------------
echo [odc] fetching model checkpoints...
"%PY%" "%ROOT%\scripts\fetch_models.py"
if errorlevel 1 goto :fail

if not exist "%ROOT%\models\rtdetrv2-s-coco\onnx\model.onnx" (
  echo [odc] exporting RT-DETRv2-S to ONNX, one time, a few minutes...
  "%PY%" "%ROOT%\scripts\export_onnx.py"
  if errorlevel 1 goto :fail
) else (
  echo [odc] ONNX detector present: models\rtdetrv2-s-coco\onnx\model.onnx
)

rem --- footage + mediapipe bundles -----------------------------------------
echo [odc] fetching footage and MediaPipe task bundles...
"%PY%" "%ROOT%\scripts\prepare_assets.py"
if errorlevel 1 goto :fail

echo [odc] assets ready.
exit /b 0

:fail
echo [odc] error: asset setup failed. Fix the message above and re-run.
exit /b 1
