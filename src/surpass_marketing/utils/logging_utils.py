# -*- coding: utf-8 -*-
"""日志工具：统一输出格式，同时写控制台与 UTF-8 日志文件。"""
from __future__ import annotations

import logging
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def get_logger(
    name: str,
    log_dir: Optional[Path] = None,
    filename: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """创建（或复用）带文件与终端双通道的 logger。"""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    logger.propagate = False
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream = logging.StreamHandler(stream=sys.stdout)
    stream.setFormatter(fmt)
    stream.setLevel(level)
    logger.addHandler(stream)

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        if filename is None:
            filename = f"run_{datetime.now():%Y%m%d_%H%M%S}.log"
        file_handler = logging.FileHandler(log_dir / filename, encoding="utf-8")
        file_handler.setFormatter(fmt)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)
    return logger


def log_environment(logger: logging.Logger) -> None:
    """将运行环境写入日志，便于复现。"""
    logger.info("=" * 72)
    logger.info("Python : %s", sys.version.split()[0])
    logger.info("平台   : %s", platform.platform())
    logger.info("时间   : %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    try:
        import numpy
        import pandas
        import sklearn
        import scipy
        import matplotlib
        import statsmodels

        logger.info(
            "依赖   : pandas=%s numpy=%s sklearn=%s scipy=%s matplotlib=%s statsmodels=%s",
            pandas.__version__,
            numpy.__version__,
            sklearn.__version__,
            scipy.__version__,
            matplotlib.__version__,
            statsmodels.__version__,
        )
    except Exception:  # pragma: no cover - 环境缺包时不应中断主流程
        logger.warning("部分依赖版本读取失败", exc_info=True)
    logger.info("=" * 72)


__all__ = ["get_logger", "log_environment"]
