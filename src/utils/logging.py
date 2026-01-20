from __future__ import annotations

import logging
import os
from typing import Optional


def setup_logging(output_dir: str, name: str = "run", level: int = logging.INFO) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # avoid duplicate handlers in notebooks
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    log_path = os.path.join(output_dir, f"{name}.log")
    if not any(getattr(h, "baseFilename", "") == log_path for h in logger.handlers):
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    return logger
