# -*- coding: utf-8 -*-
"""Controller 层：接收 TUI / CLI 请求，调度 Service 并返回领域结果。

Controller 不承载任何统计或建模逻辑，只负责用例编排（use case）与日志记录。
"""
from __future__ import annotations

import logging
from typing import List, Optional

import pandas as pd

from ..domain.entities import DataAudit, QuestionResult
from ..service import AnalysisService, DataService


class DataController:
    """数据审计与特征构建的用例入口。"""

    def __init__(self, data_service: DataService, logger: Optional[logging.Logger] = None):
        self.data = data_service
        self.logger = logger or logging.getLogger(__name__)

    def audit(self) -> DataAudit:
        self.logger.info("Controller: 执行数据审计")
        audit = self.data.audit()
        self.data.save_data_quality_report(audit)
        return audit

    def build_user_features(self, force: bool = False) -> pd.DataFrame:
        self.logger.info("Controller: 构建用户级特征表（force=%s）", force)
        return self.data.build_user_features(force=force)

    def build_attachment2_features(self) -> pd.DataFrame:
        self.logger.info("Controller: 构建附件2 待判用户特征")
        return self.data.build_attachment2_features()


class AnalysisController:
    """问题 1~4 的分析用例入口。"""

    def __init__(
        self,
        data_service: DataService,
        analysis_service: AnalysisService,
        logger: Optional[logging.Logger] = None,
    ):
        self.data = data_service
        self.analysis = analysis_service
        self.logger = logger or logging.getLogger(__name__)

    def run(self, question: str, users: Optional[pd.DataFrame] = None) -> QuestionResult:
        question = question.lower()
        if question == "q4":
            return self.run_question4(users)
        if question not in {"q1", "q2", "q3"}:
            raise ValueError(f"未知问题编号：{question}")
        users = users if users is not None else self.data.build_user_features()
        self.logger.info("Controller: 执行 %s，用户数 %d", question, len(users))
        return getattr(self.analysis, f"run_question{question[1]}")(users)

    def run_question4(self, users: Optional[pd.DataFrame] = None) -> QuestionResult:
        users = users if users is not None else self.data.build_user_features()
        attachment2 = self.data.build_attachment2_features()
        self.logger.info(
            "Controller: 执行 q4，建模用户 %d 人、附件2 %d 人", len(users), len(attachment2)
        )
        return self.analysis.run_question4(users, attachment2)

    def run_all(self, users: Optional[pd.DataFrame] = None) -> List[QuestionResult]:
        users = users if users is not None else self.data.build_user_features()
        results = [self.run(question, users) for question in ("q1", "q2", "q3")]
        results.append(self.run_question4(users))
        return results


__all__ = ["DataController", "AnalysisController"]
