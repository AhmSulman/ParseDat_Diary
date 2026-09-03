@echo off
REM ===========================================================
REM  ParseDat_Diary - KivyMD desktop app, full GPU offload
REM ===========================================================
REM  Starts llama-server with -ngl 99 and opens the GUI.
REM
REM  The server is started THROUGH main.py, not by calling
REM  llama-server.exe here. That is deliberate: the commit guard
REM  runs, the KV cache is priced against the real context size,
REM  and a load that would thrash the machine is refused with a
REM  reason. Calling the exe directly from a .bat bypasses all
REM  of it, which is how this machine used to lock up.
REM ===========================================================
cd /d "%~dp0"

if not exist "_venv\Scripts\python.exe" (
    echo.
    echo   _venv not found in %CD%
    echo   Create it:  python -m venv _venv
    echo               _venv\Scripts\python.exe -m pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

"_venv\Scripts\python.exe" main.py start --gpu-layers 99 %*

if errorlevel 1 (
    echo.
    echo   Startup failed. Try:  _venv\Scripts\python.exe main.py memory
    echo.
    pause
)
