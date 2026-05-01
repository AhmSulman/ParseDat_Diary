import os, sys
os.makedirs("logs", exist_ok=True)
try:
    from loguru import logger as log
    log.remove()
    log.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}", level="INFO", colorize=True)
    log.add("logs/app.log", format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}", level="DEBUG", rotation="10 MB", retention="7 days", encoding="utf-8")
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S", handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("logs/app.log", encoding="utf-8")])
    log = logging.getLogger("maan")
