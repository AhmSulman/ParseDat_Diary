@echo off
:: ============================================================
::  MAAN Windows Service Installer
::  Run as Administrator!
:: ============================================================
NET SESSION >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo ERROR: Please right-click and "Run as Administrator"
    pause
    exit /b 1
)

echo.
echo  ███╗   ███╗ █████╗  █████╗ ███╗   ██╗
echo  ████╗ ████║██╔══██╗██╔══██╗████╗  ██║
echo  ██╔████╔██║███████║███████║██╔██╗ ██║
echo  ██║╚██╔╝██║██╔══██║██╔══██║██║╚██╗██║
echo  ██║ ╚═╝ ██║██║  ██║██║  ██║██║ ╚████║
echo  ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝
echo.
echo  MAAN — Chat with Books  ^|  Windows Service Installer
echo.

:: Install Python dependencies
echo [1/4] Installing Python dependencies...
pip install -r requirements.txt
pip install pywin32 fastapi uvicorn

:: Post-install pywin32 setup
python -c "import pywin32_postinstall; pywin32_postinstall.install()" 2>nul

:: Install the service
echo.
echo [2/4] Registering MAAN Windows service...
python main.py service install

:: Start the service
echo.
echo [3/4] Starting MAAN service...
python main.py service start

:: Show status
echo.
echo [4/4] Service status:
sc query MAANChatBooks

echo.
echo ============================================================
echo  MAAN is now running as a Windows service!
echo  API available at: http://localhost:8000
echo.
echo  Commands:
echo    python main.py service stop    - Stop the service
echo    python main.py service start   - Start again
echo    python main.py service remove  - Uninstall
echo ============================================================
echo.
pause
