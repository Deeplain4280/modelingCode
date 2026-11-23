# -*- coding: utf-8 -*-
"""服务抽象层（Service）：只定义契约，由 ``service_impl`` 提供实现。

分层调用关系::

    TUI / main ──> Controller ──> Service(ABC) ──> ServiceImpl ──> Repository / Domain
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from ..domain.entities import DataAudit, ModelBundle, QuestionResult


class DataService(ABC):
    """数据审计、清洗与用户级特征构建。"""

    @abstractmethod
    def audit(self) -> DataAudit:
        """对 8 张原始工作表做完整性/重复/覆盖率审计。"""

    @abstractmethod
    def build_user_features(self, force: bool = False) -> pd.DataFrame:
        """构建建模主表：一行一个用户（含标签与全部特征）。"""

    @abstractmethod
    def build_attachment2_features(self) -> pd.DataFrame:
        """为附件2的 50 位待判用户构建特征表。"""

    @abstractmethod
    def save_data_quality_report(self, audit: DataAudit) -> pd.DataFrame:
        """输出数据质量报告（CSV + 日志）。"""


class AnalysisService(ABC):
    """问题 1~4 的统计分析与模型解释。"""

    @abstractmethod
    def run_question1(self, users: pd.DataFrame) -> QuestionResult:
        """数据预处理与描述性统计（含统计检验）。"""

    @abstractmethod
    def run_question2(self, users: pd.DataFrame) -> QuestionResult:
        """基本行为特征对购买行为的影响。"""

    @abstractmethod
    def run_question3(self, users: pd.DataFrame) -> QuestionResult:
        """个人偏好对购买行为的影响（偏好指标 + PCA/因子 + 聚类画像）。"""

    @abstractmethod
    def run_question4(
        self, users: pd.DataFrame, attachment2_features: pd.DataFrame
    ) -> QuestionResult:
        """目标用户潜力评估、附件2 预测、Top100 与分层营销策略。"""


class ModelingService(ABC):
    """特征矩阵组装、模型训练、评估与持久化。"""

    @abstractmethod
    def build_matrix(
        self, users: pd.DataFrame, feature_columns: Sequence[str]
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """按给定特征列构造 X, y。"""

    @abstractmethod
    def run_model_suite(
        self, users: pd.DataFrame, feature_columns: Sequence[str], task_name: str
    ) -> Any:
        """训练候选模型并返回 ``ModelSuite``（对比表、最优模型、测试集概率）。"""

    @abstractmethod
    def train_q4_final(self, users: pd.DataFrame) -> Any:
        """训练问题4的校准模型（含阈值与分层规则）。"""


class PredictionService(ABC):
    """问题 4：潜力分层、附件2预测、Top100 与营销策略。"""

    @abstractmethod
    def tier_thresholds_from_scores(self, scores: np.ndarray) -> Dict[str, float]:
        """基于训练集分数分位数确定分层阈值。"""

    @abstractmethod
    def assign_tiers(self, probabilities: np.ndarray, thresholds: Dict[str, float]) -> List[str]:
        """将概率映射为高/中/低潜力。"""

    @abstractmethod
    def predict_attachment2(
        self, bundle: ModelBundle, attachment2: pd.DataFrame
    ) -> pd.DataFrame:
        """输出附件2 50 位用户的购买概率、分层与主要驱动因素。"""

    @abstractmethod
    def rank_unpurchased_topn(
        self, bundle: ModelBundle, users: pd.DataFrame, top_n: int = 100
    ) -> pd.DataFrame:
        """对未购买用户排序，输出 Top-N 重点营销名单。"""

    @abstractmethod
    def build_marketing_plan(
        self, scored_users: pd.DataFrame, personas: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """生成分层营销策略矩阵与人群画像摘要。"""


__all__ = [
    "DataService",
    "AnalysisService",
    "ModelingService",
    "PredictionService",
]
