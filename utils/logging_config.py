import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path


def configure_logging(level: str | None = None) -> logging.Logger:
    """Configure console and rotating file logs once per process."""
    root = logging.getLogger("earningsiq")
    if root.handlers:
        return root

    log_level = getattr(logging, (level or os.getenv("LOG_LEVEL", "INFO")).upper(), logging.INFO)
    root.setLevel(log_level)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    log_dir = Path(os.getenv("LOG_DIR", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_dir / "earningsiq.log",
        maxBytes=2_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)
    root.propagate = False
    return root


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(f"earningsiq.{name}")
