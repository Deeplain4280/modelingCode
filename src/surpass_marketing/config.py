# -*- coding: utf-8 -*-
"""全局配置加载。

配置分为三层：
1. ``configs/default.yaml``：可调参数（随机种子、切分比例、分位数阈值等）；
2. ``PathConfig``：由代码目录推导出的绝对路径，保证任何机器上可复现；
3. 运行时环境（Python / 依赖版本）记录到日志，便于复现。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def find_code_root() -> Path:
    """定位 ``代码/`` 目录（本文件位于 ``代码/src/surpass_marketing/config.py``）。"""
    return Path(__file__).resolve().parents[2]


@dataclass
class PathConfig:
    code_root: Path
    workspace_root: Path
    raw_dir: Path
    attachment1: Path
    attachment2: Path
    interim_dir: Path
    table_dir: Path
    figure_dir: Path
    model_dir: Path
    log_dir: Path
    config_file: Path

    def ensure_dirs(self) -> None:
        for directory in (
            self.interim_dir,
            self.table_dir,
            self.figure_dir,
            self.model_dir,
            self.log_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


@dataclass
class DataConfig:
    raw_dir: str
    attachment1: str
    attachment2: str
    purchase_state: str
    user_time_column: str
    user_state_column: str
    user_id_column: str


@dataclass
class RuntimeConfig:
    seed: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    n_jobs: int = 1
    top_n: int = 100


@dataclass
class BusinessConfig:
    tier_high_quantile: float = 0.90
    tier_medium_quantile: float = 0.70
    contact_cost: float = 5.0
    conversion_value: float = 100.0
    attachment2_expected_users: int = 50

    @property
    def break_even_probability(self) -> float:
        """成本收益平衡概率 p* = 触达成本 / 转化毛利。"""
        if self.conversion_value <= 0:
            return 0.0
        return self.contact_cost / self.conversion_value


@dataclass
class AppConfig:
    project: Dict[str, Any]
    runtime: RuntimeConfig
    business: BusinessConfig
    data: DataConfig
    model: Dict[str, Any]
    paths: PathConfig

    # ------------------------------------------------------------------ #
    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "AppConfig":
        code_root = find_code_root()
        config_file = Path(config_path) if config_path else code_root / "configs" / "default.yaml"
        if not config_file.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_file}")
        with open(config_file, "r", encoding="utf-8") as handle:
            raw: Dict[str, Any] = yaml.safe_load(handle)

        workspace_root = code_root.parent
        data_cfg = DataConfig(**raw["data"])
        raw_dir = (workspace_root / data_cfg.raw_dir).resolve()
        out = raw.get("outputs", {})

        paths = PathConfig(
            code_root=code_root,
            workspace_root=workspace_root,
            raw_dir=raw_dir,
            attachment1=raw_dir / data_cfg.attachment1,
            attachment2=raw_dir / data_cfg.attachment2,
            interim_dir=code_root / out.get("interim", "outputs/interim"),
            table_dir=code_root / out.get("tables", "outputs/tables"),
            figure_dir=code_root / out.get("figures", "outputs/figures"),
            model_dir=code_root / out.get("models", "outputs/models"),
            log_dir=code_root / out.get("logs", "outputs/logs"),
            config_file=config_file,
        )
        paths.ensure_dirs()

        return cls(
            project=raw.get("project", {}),
            runtime=RuntimeConfig(**raw.get("runtime", {})),
            business=BusinessConfig(**raw.get("business", {})),
            data=data_cfg,
            model=raw.get("model", {}),
            paths=paths,
        )


__all__ = [
    "AppConfig",
    "BusinessConfig",
    "DataConfig",
    "PathConfig",
    "RuntimeConfig",
    "find_code_root",
]
