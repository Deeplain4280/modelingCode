# -*- coding: utf-8 -*-
"""数据文件定位：根据配置找到附件文件，并在缺失时给出清晰提示。"""
from __future__ import annotations

from pathlib import Path
from typing import List

from ..config import AppConfig


class DatasetLocator:
    """负责定位原始数据文件（只读），不负责解析。"""

    def __init__(self, config: AppConfig):
        self.config = config

    def find_attachment1(self) -> Path:
        path = self.config.paths.attachment1
        if not path.exists():
            raise FileNotFoundError(
                f"未找到附件1：{path}。请确认项目目录结构为 工作区/B题/B题附件/。"
            )
        return path

    def find_attachment2(self) -> Path:
        path = self.config.paths.attachment2
        if not path.exists():
            raise FileNotFoundError(
                f"未找到附件2：{path}。请确认项目目录结构为 工作区/B题/B题附件/。"
            )
        return path

    def validate(self) -> List[str]:
        """返回缺失文件清单（用于 TUI 启动自检）。"""
        missing = []
        for label, path in (
            ("附件1", self.config.paths.attachment1),
            ("附件2", self.config.paths.attachment2),
        ):
            if not path.exists():
                missing.append(f"{label} 不存在：{path}")
        return missing


__all__ = ["DatasetLocator"]
