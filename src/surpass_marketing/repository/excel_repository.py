# -*- coding: utf-8 -*-
"""Excel 数据访问层：只读原始文件，解析结果缓存为 parquet 以加速重复运行。

原始 xlsx 永远只读；所有派生数据写入 ``outputs/interim``。
使用 pandas + openpyxl 完成 Excel 读取（符合“用 Python 文件处理读取 Excel”的要求）。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from ..config import AppConfig
from ..domain import constants as C
from .paths import DatasetLocator


class ExcelRepository:
    """附件1/附件2 的只读访问与 parquet 缓存。"""

    def __init__(self, config: AppConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.locator = DatasetLocator(config)
        self.cache_dir = config.paths.interim_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    def locate(self) -> Dict[str, Path]:
        return {
            "attachment1": self.locator.find_attachment1(),
            "attachment2": self.locator.find_attachment2(),
        }

    def _cache_path(self, key: str) -> Path:
        # 缓存文件直接以工作表名命名，便于人工核对；工作表名不含路径分隔符。
        return self.cache_dir / f"{key}.parquet"

    @staticmethod
    def _sanitize(frame: pd.DataFrame) -> pd.DataFrame:
        """把 object 列统一转为字符串，避免 Excel 混合类型导致 parquet 写入失败。"""
        for column in frame.columns:
            if frame[column].dtype == object:
                frame[column] = [
                    None
                    if value is None or (isinstance(value, float) and pd.isna(value))
                    else str(value)
                    for value in frame[column]
                ]
        return frame

    def sheet_names(self) -> List[str]:
        with pd.ExcelFile(self.locator.find_attachment1(), engine="openpyxl") as workbook:
            return list(workbook.sheet_names)

    def load_sheet(self, sheet_name: str, force: bool = False) -> pd.DataFrame:
        """读取附件1的某个工作表；优先使用 parquet 缓存。"""
        cache = self._cache_path(sheet_name)
        if cache.exists() and not force:
            return pd.read_parquet(cache)

        path = self.locator.find_attachment1()
        self.logger.info("读取 Excel 工作表：%s（首次读取，将写入缓存）", sheet_name)
        if sheet_name == C.SHEET_ID_MAPPING:
            frame = pd.read_excel(
                path, sheet_name=sheet_name, header=None, names=[C.COL_SID, C.COL_USER_ID]
            )
        else:
            frame = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")
        frame.columns = [str(column).strip() for column in frame.columns]
        frame = self._sanitize(frame)
        frame.to_parquet(cache, index=False)
        self.logger.info("工作表 %s 读取完成：%d 行 × %d 列", sheet_name, *frame.shape)
        return frame

    def load_all_sheets(self, force: bool = False) -> Dict[str, pd.DataFrame]:
        return {name: self.load_sheet(name, force=force) for name in C.ALL_SHEETS}

    def load_attachment2(self, force: bool = False) -> pd.DataFrame:
        """读取附件2“待判数据”（50 位用户）。"""
        cache = self._cache_path("附件2")
        if cache.exists() and not force:
            return pd.read_parquet(cache)
        path = self.locator.find_attachment2()
        self.logger.info("读取附件2：%s", path)
        frame = pd.read_excel(path, sheet_name=0, engine="openpyxl")
        frame.columns = [str(column).strip() for column in frame.columns]
        frame = self._sanitize(frame)
        frame = frame.dropna(how="all")
        frame.to_parquet(cache, index=False)
        self.logger.info("附件2 读取完成：%d 行 × %d 列", *frame.shape)
        return frame


__all__ = ["ExcelRepository"]
