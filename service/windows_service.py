"""
MAAN Windows Service
=====================
Runs MAAN as a background Windows service that:
  - Starts automatically when Windows boots
  - Restarts itself after crashes (auto-recovery)
  - Logs everything to logs/service.log

Requirements:
  pip install pywin32
  Run as Administrator!

Usage (via main.py):
  python main.py service install   ← Install + register
  python main.py service start     ← Start now
  python main.py service stop      ← Stop
  python main.py service remove    ← Uninstall
  python main.py service status    ← Check if running

Or directly:
  python service/windows_service.py install
  python service/windows_service.py start
"""

import sys
import os
import time

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from logs.logger import log


# ── Service Manager (cross-platform CLI) ─────────────────────────────────────

class ServiceManager:
    SERVICE_NAME = "MAANChatBooks"
    DISPLAY_NAME = "MAAN - Chat with Books"

    def handle(self, action: str):
        if sys.platform != "win32":
            log.error("❌ Windows service only runs on Windows.")
            log.info("   On Linux/macOS, use: systemd or launchd instead.")
            return

        if action == "install":
            self._install()
        elif action == "remove":
            self._remove()
        elif action == "start":
            self._start()
        elif action == "stop":
            self._stop()
        elif action == "status":
            self._status()

    def _install(self):
        log.info(f"📦 Installing MAAN as Windows service: {self.SERVICE_NAME}")
        log.info("   Make sure you're running as Administrator!")
        try:
            import win32serviceutil
            # Register the service
            os.system(
                f'sc create "{self.SERVICE_NAME}" '
                f'binPath= "python {PROJECT_ROOT}\\service\\windows_service.py" '
                f'DisplayName= "{self.DISPLAY_NAME}" '
                f'start= auto'
            )
            # Set crash recovery: restart after 1st, 2nd, 3rd failure
            os.system(
                f'sc failure "{self.SERVICE_NAME}" '
                f'reset= 86400 '          # Reset failure count after 1 day
                f'actions= restart/5000/restart/10000/restart/30000'  # Retry delays (ms)
            )
            log.info("✅ Service installed with auto-restart on crash")
            log.info(f"   Run: python main.py service start")
        except ImportError:
            log.error("❌ pywin32 not installed → pip install pywin32")
        except Exception as e:
            log.error(f"❌ Install failed: {e}")

    def _remove(self):
        self._stop()
        os.system(f'sc delete "{self.SERVICE_NAME}"')
        log.info("✅ Service removed")

    def _start(self):
        os.system(f'sc start "{self.SERVICE_NAME}"')
        log.info(f"▶️  Service starting: {self.SERVICE_NAME}")

    def _stop(self):
        os.system(f'sc stop "{self.SERVICE_NAME}"')
        log.info(f"⏹️  Service stopped: {self.SERVICE_NAME}")

    def _status(self):
        os.system(f'sc query "{self.SERVICE_NAME}"')


# ── Actual Windows Service Class ──────────────────────────────────────────────

def run_as_windows_service():
    """Called by Windows SCM when service starts."""
    try:
        import win32serviceutil
        import win32service
        import win32event
        import servicemanager
    except ImportError:
        print("pywin32 not installed. Run: pip install pywin32")
        sys.exit(1)

    class MAANService(win32serviceutil.ServiceFramework):
        _svc_name_ = "MAANChatBooks"
        _svc_display_name_ = "MAAN - Chat with Books"
        _svc_description_ = "Local AI book chat daemon — RTX accelerated RAG pipeline"

        def __init__(self, args):
            win32serviceutil.ServiceFramework.__init__(self, args)
            self.hWaitStop = win32event.CreateEvent(None, 0, 0, None)
            self.running = True

        def SvcStop(self):
            self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
            win32event.SetEvent(self.hWaitStop)
            self.running = False

        def SvcDoRun(self):
            servicemanager.LogInfoMsg("MAAN service starting...")
            self._run_maan()

        def _run_maan(self):
            """Main service loop — runs the MAAN web API server."""
            from chat.server import run_server
            from config.config import Config
            cfg = Config()

            while self.running:
                try:
                    servicemanager.LogInfoMsg(f"MAAN starting API on port {cfg.SERVER_PORT}")
                    run_server(
                        host=cfg.SERVER_HOST,
                        port=cfg.SERVER_PORT,
                        model_path=cfg.LLM_MODEL_PATH,
                    )
                except Exception as e:
                    servicemanager.LogErrorMsg(f"MAAN crashed: {e}. Restarting in 5s...")
                    time.sleep(5)  # Wait before restart (Windows SCM will also retry)

    if len(sys.argv) == 1:
        servicemanager.Initialize()
        servicemanager.PrepareToHostSingle(MAANService)
        servicemanager.StartServiceCtrlDispatcher()
    else:
        win32serviceutil.HandleCommandLine(MAANService)


if __name__ == "__main__":
    run_as_windows_service()
