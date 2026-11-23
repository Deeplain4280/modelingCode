# -*- coding: utf-8 -*-
"""应用控制器：完成依赖装配，向 TUI / CLI 暴露任务级用例。"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from ..config import AppConfig
from ..domain.entities import DataAudit, QuestionResult
from ..repository.artifact_repository import ArtifactRepository
from ..repository.excel_repository import ExcelRepository
from ..service_impl.analysis_service_impl import AnalysisServiceImpl
from ..service_impl.data_service_impl import DataServiceImpl
from ..service_impl.modeling_service_impl import ModelingServiceImpl
from ..service_impl.prediction_service_impl import PredictionServiceImpl
from ..utils.logging_utils import get_logger, log_environment
from .data_controller import AnalysisController, DataController

TASK_NAMES = {
    "audit": "数据审计",
    "q1": "问题1：数据预处理与描述统计",
    "q2": "问题2：基本行为特征的影响",
    "q3": "问题3：个人偏好的影响",
    "q4": "问题4：潜力评估与精准营销",
    "all": "一键完整运行",
}


class AppController:
    """组合根（composition root）：装配 Repository → Service → Controller。"""

    def __init__(self, config: AppConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or get_logger("surpass", config.paths.log_dir, "run.log")
        log_environment(self.logger)

        self.excel_repository = ExcelRepository(config, self.logger)
        self.artifact_repository = ArtifactRepository(config, self.logger)

        self.data_service = DataServiceImpl(
            config, self.excel_repository, self.artifact_repository, self.logger
        )
        self.modeling_service = ModelingServiceImpl(config, self.artifact_repository, self.logger)
        self.prediction_service = PredictionServiceImpl(config, self.artifact_repository, self.logger)
        self.analysis_service = AnalysisServiceImpl(
            config,
            self.data_service,
            self.modeling_service,
            self.artifact_repository,
            self.logger,
            self.prediction_service,
        )

        self.data_controller = DataController(self.data_service, self.logger)
        self.analysis_controller = AnalysisController(
            self.data_service, self.analysis_service, self.logger
        )

    # ------------------------------------------------------------------ #
    # 用例
    # ------------------------------------------------------------------ #
    def run(self, task: str) -> Dict[str, QuestionResult]:
        task = task.lower()
        if task not in TASK_NAMES:
            raise ValueError(f"未知任务：{task}（可选 {', '.join(TASK_NAMES)}）")
        self.logger.info("=" * 70)
        self.logger.info("开始执行任务：%s（%s）", task, TASK_NAMES[task])

        if task == "audit":
            self.data_controller.audit()
            return {}
        if task == "all":
            return self.run_all()
        result = self.analysis_controller.run(task)
        self._write_summary({task: result})
        return {task: result}

    def run_all(self) -> Dict[str, QuestionResult]:
        self.logger.info("一键完整运行：数据审计 → 问题1~4")
        self.data_controller.audit()
        users = self.data_controller.build_user_features()
        results: Dict[str, QuestionResult] = {}
        for question in ("q1", "q2", "q3", "q4"):
            results[question] = self.analysis_controller.run(question, users)
        self._write_summary(results)
        self.logger.info("全部任务完成，共 %d 个问题结果", len(results))
        return results

    def ensure_features(self, force: bool = False) -> pd.DataFrame:
        return self.data_controller.build_user_features(force=force)

    # ------------------------------------------------------------------ #
    # 输出与日志
    # ------------------------------------------------------------------ #
    def list_outputs(self, subdir: Optional[str] = None) -> pd.DataFrame:
        paths = self.artifact_repository.list_outputs()
        rows = [
            {
                "目录": path.parent.name,
                "文件": path.name,
                "大小(KB)": path.stat().st_size / 1024.0,
            }
            for path in paths
            if path.is_file() and (subdir is None or path.parent.name == subdir)
        ]
        return pd.DataFrame(rows, columns=["目录", "文件", "大小(KB)"])

    def latest_logs(self, limit: int = 5) -> str:
        lines: List[str] = []
        for path in self.artifact_repository.latest_logs(limit):
            lines.append(f"===== {path.name}（最后 40 行）=====")
            try:
                content = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError as error:  # pragma: no cover - 防御式编程
                lines.append(f"读取失败：{error}")
                continue
            lines.extend(content[-40:])
            lines.append("")
        return "\n".join(lines) if lines else "暂无日志文件。"

    def _write_summary(self, results: Dict[str, QuestionResult]) -> None:
        lines = ["品牌手机目标用户的精准营销 —— 运行摘要", ""]
        for question, result in results.items():
            lines.append(f"[{question}] {result.name}")
            lines.append(f"  数据表：{', '.join(result.tables) if result.tables else '无'}")
            lines.append(f"  图    ：{', '.join(result.figures) if result.figures else '无'}")
            if result.metrics:
                metrics = "；".join(
                    f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}"
                    for key, value in result.metrics.items()
                )
                lines.append(f"  指   标：{metrics}")
            if result.notes:
                lines.append(f"  说   明：{result.notes}")
            lines.append("")
        self.artifact_repository.save_text("\n".join(lines), "运行摘要.txt")


__all__ = ["AppController", "TASK_NAMES"]
