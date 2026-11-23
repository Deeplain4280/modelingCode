# -*- coding: utf-8 -*-
"""产物仓库：统一管理表格、图片、模型、文本与日志输出。"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd

from ..config import AppConfig
from ..utils.plotting import save_figure


class ArtifactRepository:
    """所有输出统一落在 ``outputs/{tables,figures,models,logs}``。"""

    def __init__(self, config: AppConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

    # ------------------------------ 表格 ------------------------------ #
    def save_table(self, frame: pd.DataFrame, filename: str, index: bool = False) -> Path:
        path = self.config.paths.table_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=index, encoding="utf-8-sig")
        self.logger.info("已保存表格：%s（%d 行）", path.name, len(frame))
        return path

    def save_frame_parquet(self, frame: pd.DataFrame, filename: str) -> Path:
        path = self.config.paths.interim_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        return path

    # ------------------------------ 图片 ------------------------------ #
    def save_figure(self, fig, filename: str) -> Path:
        path = self.config.paths.figure_dir / filename
        save_figure(fig, path)
        self.logger.info("已保存图表：%s", path.name)
        return path

    # ------------------------------ 模型 ------------------------------ #
    def save_model(self, bundle: Any, filename: str) -> Path:
        path = self.config.paths.model_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, path)
        self.logger.info("已保存模型：%s", path.name)
        if hasattr(bundle, "save_meta"):
            bundle.save_meta(path.with_suffix(".json"))
        return path

    def load_model(self, filename: str) -> Any:
        path = self.config.paths.model_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"模型文件不存在：{path}")
        return joblib.load(path)

    # ------------------------------ 文本/JSON ------------------------------ #
    def save_text(self, content: str, filename: str) -> Path:
        path = self.config.paths.table_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def save_json(self, payload: Dict[str, Any], filename: str) -> Path:
        path = self.config.paths.table_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        return path

    # ------------------------------ 汇总 ------------------------------ #
    def list_outputs(self) -> List[Path]:
        files: List[Path] = []
        for directory in (
            self.config.paths.table_dir,
            self.config.paths.figure_dir,
            self.config.paths.model_dir,
            self.config.paths.log_dir,
        ):
            if directory.exists():
                files.extend(sorted(directory.glob("*")))
        return files

    def latest_logs(self, limit: int = 5) -> List[Path]:
        log_dir = self.config.paths.log_dir
        if not log_dir.exists():
            return []
        return sorted(log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]


__all__ = ["ArtifactRepository"]
