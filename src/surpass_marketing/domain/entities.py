# -*- coding: utf-8 -*-
"""领域实体：数据审计结果、模型指标、模型包、预测结果。"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class DataAudit:
    """数据审计结果。"""

    generated_at: str
    sheet_rows: Dict[str, int]
    sheet_unique_users: Dict[str, int]
    duplicate_rows: Dict[str, int]
    coverage: Dict[str, float]
    quality_issues: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for sheet in self.sheet_rows:
            rows.append(
                {
                    "数据表": sheet,
                    "行数": self.sheet_rows[sheet],
                    "唯一用户数": self.sheet_unique_users.get(sheet, 0),
                    "重复行数": self.duplicate_rows.get(sheet, 0),
                    "目标用户覆盖率": round(self.coverage.get(sheet, float("nan")), 4),
                }
            )
        return pd.DataFrame(rows)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelMetrics:
    """二分类模型在测试集上的评估指标。"""

    model_name: str
    roc_auc: float
    pr_auc: float
    precision: float
    recall: float
    f1: float
    brier: float
    log_loss: float
    threshold: float
    accuracy: float = 0.0
    n_test: int = 0
    n_positive: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data.update(data.pop("extra", {}) or {})
        return data

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([self.to_dict()])


@dataclass
class ModelBundle:
    """训练完成的模型包（含预处理管线、阈值与元信息）。"""

    model_name: str
    pipeline: Any
    feature_names: List[str]
    metrics: ModelMetrics
    tier_thresholds: Dict[str, float]
    positive_rate: float
    trained_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    calibration_method: str = "sigmoid"
    notes: str = ""
    # 非线性模型（如梯度提升树）的整体特征重要性，供近似解释使用
    feature_importance: Dict[str, float] = field(default_factory=dict)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        # 只做列对齐：缺失列置为 NaN，交由管线内的 Imputer 处理，避免把“缺失”
        # 误当成数值 0（例如年龄缺失不应等价于 18-24 岁）。
        matrix = frame.reindex(columns=self.feature_names)
        return self.pipeline.predict_proba(matrix)[:, 1]

    def to_meta(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "feature_names": self.feature_names,
            "metrics": self.metrics.to_dict() if self.metrics is not None else None,
            "tier_thresholds": self.tier_thresholds,
            "positive_rate": self.positive_rate,
            "trained_at": self.trained_at,
            "calibration_method": self.calibration_method,
            "notes": self.notes,
        }

    def save_meta(self, path) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_meta(), handle, ensure_ascii=False, indent=2)


@dataclass
class QuestionResult:
    """单个问题的执行结果摘要。"""

    name: str
    tables: List[str] = field(default_factory=list)
    figures: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def summary_lines(self) -> List[str]:
        lines = [f"【{self.name}】完成"]
        if self.tables:
            lines.append("  表格: " + ", ".join(self.tables))
        if self.figures:
            lines.append("  图表: " + ", ".join(self.figures))
        for key, value in self.metrics.items():
            if isinstance(value, float):
                lines.append(f"  {key}: {value:.4f}")
            else:
                lines.append(f"  {key}: {value}")
        if self.notes:
            lines.append("  说明: " + self.notes)
        return lines


__all__ = [
    "DataAudit",
    "ModelMetrics",
    "ModelBundle",
    "QuestionResult",
]
