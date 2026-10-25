@echo off
rem ===========================================================================
rem  PLOMID ODC - one-command startup for Windows.
rem
rem    start.bat [--auto-start] [--port 8080]
rem
rem  Starts the host PLOMID server from
rem    ..\plomid\target\release\plomid-server.exe
rem  only when nothing answers on the configured endpoint, provisions the
rem  ODC schema INSIDE the single PLOMID database, then serves the console.
rem
rem  PLOMID is the only database used. ODC never opens SQLite or any other
rem  store: all durable state lives in the `plomid` database (the ODC tables
rem  are in the `plomid_odc` schema). A non-PLOMID DSN is rejected before
rem  anything starts.
rem
rem  Env knobs (all optional):
rem    ODC_DSN       default postgresql://plomid:plomid@127.0.0.1:5432/plomid
rem    ODC_PYTHON    interpreter to use (default: .venv, ..\.venv, py -3.12)
rem    PLOMID_BIN    default ..\plomid\target\release\plomid-server.exe
rem    PLOMID_DATA   default ..\plomid\data
rem    ODC_NO_SETUP  set to 1 to skip automatic pip install
rem ===========================================================================
setlocal EnableExtensions
set "EXITCODE=0"
cd /d "%~dp0"
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

rem --- resolve Python 3.12+ ------------------------------------------------
set "PY="
if defined ODC_PYTHON (
  if exist "%ODC_PYTHON%" (
    set "PY=%ODC_PYTHON%"
  ) else (
    echo [odc] error: ODC_PYTHON not found: %ODC_PYTHON%
    goto :fail
  )
)
if not defined PY if exist "%ROOT%\.venv\Scripts\python.exe" set "PY=%ROOT%\.venv\Scripts\python.exe"
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
  goto :fail
)
"%PY%" -c "import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 12) else 1)" >nul 2>&1
if errorlevel 1 (
  echo [odc] error: Python 3.12 or newer is required: %PY%
  goto :fail
)

rem --- PLOMID is the only database -----------------------------------------
if not defined ODC_DSN set "ODC_DSN=postgresql://plomid:plomid@127.0.0.1:5432/plomid"
set "ODC_DSN"
"%PY%" -c "import os,sys,urllib.parse as u; d=os.environ.get('ODC_DSN',''); p=u.urlparse(d); db=p.path.strip('/'); ok=p.scheme in ('postgresql','postgres') and db=='plomid'; print(('PLOMID database: ' + (p.hostname or '127.0.0.1') + ':' + str(p.port or 5432) + '/plomid') if ok else ('error: ODC_DSN must target the PLOMID database, e.g. postgresql://plomid:plomid@127.0.0.1:5432/plomid (got: ' + d + ')'))"
if errorlevel 1 goto :fail
if not defined TEMP set "TEMP=%ROOT%"
set "ENDPOINT_FILE=%TEMP%\plomid-odc-endpoint.txt"
"%PY%" -c "import os,urllib.parse as u; p=u.urlparse(os.environ['ODC_DSN']); print(p.hostname or '127.0.0.1', p.port or 5432)" > "%ENDPOINT_FILE%"
if errorlevel 1 (
  del "%ENDPOINT_FILE%" >nul 2>&1
  echo [odc] error: could not parse ODC_DSN.
  goto :fail
)
for /f "usebackq tokens=1,2" %%A in ("%ENDPOINT_FILE%") do (
  set "DB_HOST=%%A"
  set "DB_PORT=%%B"
)
del "%ENDPOINT_FILE%" >nul 2>&1
if not defined DB_HOST (
  echo [odc] error: could not parse ODC_DSN.
  goto :fail
)

rem --- Python runtime dependencies -----------------------------------------
"%PY%" -c "import importlib.util as u; m=['psycopg','fastapi','uvicorn','jinja2','yaml','pydantic_settings','numpy','cv2','onnxruntime','mediapipe','torch','transformers','huggingface_hub']; miss=[x for x in m if u.find_spec(x) is None]; print('missing: '+', '.join(miss) if miss else ''); raise SystemExit(1 if miss else 0)"
if errorlevel 1 (
  if defined ODC_NO_SETUP (
    echo [odc] error: Python packages are missing.
    echo [odc]        install them with:
    echo [odc]        "%PY%" -m pip install -r "%ROOT%\requirements.lock"
    goto :fail
  )
  echo [odc] Python packages missing; installing requirements.lock on first run...
  rem uvloop is POSIX-only and the lock is generated on macOS; drop it on Windows.
  set "REQ_WIN=%TEMP%\plomid-odc-requirements-windows.txt"
  powershell -NoProfile -Command "Get-Content -LiteralPath '%ROOT%\requirements.lock' | Where-Object { $_ -notmatch '^uvloop' } | Set-Content -Encoding ASCII -LiteralPath $env:TEMP\plomid-odc-requirements-windows.txt"
  if errorlevel 1 (
    echo [odc] error: could not prepare Windows requirements.
    goto :fail
  )
  "%PY%" -m pip install --disable-pip-version-check -r "%REQ_WIN%"
  if errorlevel 1 (
    echo [odc] error: dependency installation failed.
    echo [odc]        install manually:
    echo [odc]        "%PY%" -m pip install -r "%REQ_WIN%"
    goto :fail
  )
)

rem --- model / footage assets ----------------------------------------------
"%PY%" "%ROOT%\scripts\prepare_assets.py" --check >nul 2>&1
if errorlevel 1 (
  echo [odc] warning: model or footage assets are missing.
  echo [odc]          run setup-assets.bat once to download and verify them.
)

rem --- host PLOMID server ---------------------------------------------------
set "LOCAL_DB="
if /i "%DB_HOST%"=="127.0.0.1" set "LOCAL_DB=1"
if /i "%DB_HOST%"=="localhost" set "LOCAL_DB=1"
if /i "%DB_HOST%"=="::1" set "LOCAL_DB=1"

if defined LOCAL_DB (
  call :is_up
  if errorlevel 1 (
    call :start_server
    if errorlevel 1 goto :fail
  ) else (
    echo [odc] PLOMID server already up at %DB_HOST%:%DB_PORT%
  )
) else (
  echo [odc] using remote PLOMID at %DB_HOST%:%DB_PORT%
)

rem --- wait for the database ------------------------------------------------
set /a DB_TRIES=0
:wait_db
call :is_up
if not errorlevel 1 goto :db_ready
set /a DB_TRIES+=1
if %DB_TRIES% GEQ 30 (
  echo [odc] error: PLOMID at %DB_HOST%:%DB_PORT% never became ready.
  if defined STARTED_DB echo [odc]        see "%SERVER_ERR%"
  if defined STARTED_DB call :show_server_error
  goto :fail
)
"%PY%" -c "import time; time.sleep(1)" >nul 2>&1
goto :wait_db

:db_ready
rem --- provision ODC inside the PLOMID database -----------------------------
echo [odc] provisioning ODC schema inside PLOMID...
"%PY%" -m odc.cli bootstrap
if errorlevel 1 (
  echo [odc] error: PLOMID bootstrap failed.
  goto :fail
)

rem --- serve the console ----------------------------------------------------
echo [odc] starting console...
"%PY%" -m odc.cli serve %*
set "EXITCODE=%ERRORLEVEL%"
goto :cleanup

rem --- helpers --------------------------------------------------------------
:is_up
"%PY%" -c "import socket; socket.create_connection(('%DB_HOST%', %DB_PORT%), 1).close()" >nul 2>&1
exit /b %ERRORLEVEL%

:start_server
if not defined PLOMID_BIN (
  set "PLOMID_BIN=%ROOT%\..\plomid\target\release\plomid-server.exe"
  for %%I in ("%PLOMID_BIN%") do set "PLOMID_BIN=%%~fI"
)
if not exist "%PLOMID_BIN%" (
  for /f "delims=" %%P in ('where plomid-server.exe 2^>nul') do set "PLOMID_BIN=%%P"
)
if not exist "%PLOMID_BIN%" (
  echo [odc] error: plomid-server binary not found.
  echo [odc]        expected: %ROOT%\..\plomid\target\release\plomid-server.exe
  echo [odc]        build it:   cd /d "%ROOT%\..\plomid" ^&^& cargo build --release --bin plomid-server
  exit /b 1
)
if not defined PLOMID_DATA (
  set "PLOMID_DATA=%ROOT%\..\plomid\data"
  for %%I in ("%PLOMID_DATA%") do set "PLOMID_DATA=%%~fI"
)
if not exist "%PLOMID_DATA%" mkdir "%PLOMID_DATA%" >nul 2>&1
if not defined TEMP set "TEMP=%ROOT%"
set "SERVER_LOG=%TEMP%\plomid-odc-server.log"
set "SERVER_ERR=%TEMP%\plomid-odc-server.err.log"
echo [odc] starting PLOMID server ...
echo [odc]   binary: %PLOMID_BIN%
echo [odc]   data:   %PLOMID_DATA%
echo [odc]   log:    %SERVER_LOG%
set "SERVER_PID="
for /f "usebackq delims=" %%P in (`powershell -NoProfile -Command "$ErrorActionPreference='Stop'; $p = Start-Process -FilePath '%PLOMID_BIN%' -ArgumentList @('--data','%PLOMID_DATA%','--host','127.0.0.1','--port','%DB_PORT%','--username','plomid','--password','plomid') -RedirectStandardOutput '%SERVER_LOG%' -RedirectStandardError '%SERVER_ERR%' -WindowStyle Hidden -PassThru; $p.Id"`) do set "SERVER_PID=%%P"
if not defined SERVER_PID (
  echo [odc] error: could not start PLOMID server.
  exit /b 1
)
set "STARTED_DB=1"
exit /b 0

:show_server_error
powershell -NoProfile -Command "Get-Content -Tail 20 -ErrorAction SilentlyContinue '%SERVER_ERR%'"
exit /b 0

rem --- cleanup --------------------------------------------------------------
:fail
set "EXITCODE=1"

:cleanup
if defined STARTED_DB if defined SERVER_PID (
  echo [odc] stopping PLOMID server we started...
  taskkill /PID %SERVER_PID% /T /F >nul 2>&1
)
exit /b %EXITCODE%
