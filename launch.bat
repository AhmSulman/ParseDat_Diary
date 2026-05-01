@echo off
echo.
echo  MAAN — Chat with Books
echo  RTX 4050 · Local AI · Your Data
echo.

python --version >nul 2>&1 || (echo Python not found: https://python.org/downloads && pause && exit /b 1)

if not exist ".deps_ok" (
    echo Installing dependencies...
    pip install -r requirements.txt
    echo. > .deps_ok
)

echo Commands:
echo   python main.py ingest          -- Extract PDFs
echo   python main.py chat            -- Chat with books
echo   python main.py server          -- Web API on :8000
echo   python main.py search "query"  -- Quick search
echo   python main.py service install -- Install as Windows service
echo.

python main.py %*
pause
