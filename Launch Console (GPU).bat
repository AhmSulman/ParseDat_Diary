@echo off
REM ===========================================================
REM  ParseDat_Diary - terminal console, full GPU offload
REM ===========================================================
REM  Same guarded startup as the GUI launcher, but drops you in
REM  the terminal console instead: ask questions, inspect the
REM  library, tune retrieval. Type 'help' once it opens.
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

"_venv\Scripts\python.exe" main.py start --console --gpu-layers 99 %*

echo.
pause
