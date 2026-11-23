# -*- coding: utf-8 -*-
"""建模服务实现：候选模型训练、交叉验证选型、校准与可解释性输出。"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import AppConfig
from ..domain import constants as C
from ..domain.entities import ModelBundle, ModelMetrics
from ..repository.artifact_repository import ArtifactRepository
from ..service import ModelingService
from .feature_columns import select_features

MODEL_LOGISTIC = "logistic"
MODEL_RF = "random_forest"
MODEL_HGB = "hist_gradient_boosting"

COMPARISON_COLUMNS = [
    "模型",
    "5折CV_PR_AUC均值",
    "5折CV_PR_AUC标准差",
    "5折CV_ROC_AUC均值",
    "测试集ROC_AUC",
    "测试集PR_AUC",
    "测试集精确率",
    "测试集召回率",
    "测试集F1",
    "测试集Brier",
    "是否入选",
]


@dataclass
class ModelSuite:
    """一次模型训练（含对比、入选模型、测试集概率）的完整结果。"""

    task: str
    comparison: pd.DataFrame
    bundle: ModelBundle
    y_test: pd.Series
    prob_test: np.ndarray
    test_users: pd.Series
    coefficients: pd.DataFrame = field(default_factory=pd.DataFrame)
    feature_importance: pd.Series = field(default_factory=pd.Series)
    extra: Dict[str, Any] = field(default_factory=dict)


class ModelingServiceImpl(ModelingService):
    """模型训练与评估实现。"""

    def __init__(
        self,
        config: AppConfig,
        artifact_repository: ArtifactRepository,
        logger: Optional[logging.Logger] = None,
    ):
        self.config = config
        self.artifacts = artifact_repository
        self.logger = logger or logging.getLogger(__name__)
        self.seed = config.runtime.seed
        self.test_size = config.runtime.test_size
        self.cv_folds = config.runtime.cv_folds
        # 沙箱环境下 joblib 线程池会申请管道句柄被拒绝，统一串行训练
        self.n_jobs = int(config.runtime.n_jobs)
        self.model_cfg = config.model
        self.calibration_cfg = config.model.get("calibration", {"method": "sigmoid", "cv": 5})

    # ------------------------------------------------------------------ #
    # 矩阵构造与数据切分
    # ------------------------------------------------------------------ #
    def build_matrix(
        self, users: pd.DataFrame, feature_columns: Sequence[str]
    ) -> Tuple[pd.DataFrame, pd.Series]:
        missing = [column for column in feature_columns if column not in users.columns]
        if missing:
            raise KeyError(f"特征列不存在：{missing[:10]}")
        matrix = users.loc[:, list(feature_columns)].astype("float64")
        label = users["target"].astype("int8")
        return matrix, label

    def _exclude_reserved_users(self, users: pd.DataFrame) -> pd.DataFrame:
        """训练/评估时剔除附件2 用户，避免待判用户信息进入训练（防泄漏）。"""
        if "is_attachment2" in users.columns:
            reserved = int(users["is_attachment2"].sum())
            if reserved:
                self.logger.info("从训练集中剔除附件2 待判用户 %d 人（防泄漏）", reserved)
            return users.loc[users["is_attachment2"] == 0].reset_index(drop=True)
        return users

    # ------------------------------------------------------------------ #
    # 候选模型
    # ------------------------------------------------------------------ #
    def _candidate_pipelines(self) -> Dict[str, Pipeline]:
        logistic_cfg = self.model_cfg.get("logistic", {})
        rf_cfg = self.model_cfg.get("random_forest", {})
        hgb_cfg = self.model_cfg.get("hist_gradient_boosting", {})

        logistic = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=float(logistic_cfg.get("C", 0.5)),
                        max_iter=int(logistic_cfg.get("max_iter", 3000)),
                        class_weight=logistic_cfg.get("class_weight", "balanced"),
                        solver="lbfgs",
                    ),
                ),
            ]
        )
        forest = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=int(rf_cfg.get("n_estimators", 400)),
                        max_depth=rf_cfg.get("max_depth", 12),
                        min_samples_leaf=int(rf_cfg.get("min_samples_leaf", 5)),
                        class_weight="balanced_subsample",
                        random_state=self.seed,
                        n_jobs=self.n_jobs,
                    ),
                ),
            ]
        )
        hgb = Pipeline(
            [
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        max_iter=int(hgb_cfg.get("max_iter", 300)),
                        learning_rate=float(hgb_cfg.get("learning_rate", 0.05)),
                        max_leaf_nodes=int(hgb_cfg.get("max_leaf_nodes", 31)),
                        l2_regularization=float(hgb_cfg.get("l2_regularization", 1.0)),
                        class_weight="balanced",
                        random_state=self.seed,
                    ),
                )
            ]
        )
        return {MODEL_LOGISTIC: logistic, MODEL_RF: forest, MODEL_HGB: hgb}

    # ------------------------------------------------------------------ #
    # 指标与解释
    # ------------------------------------------------------------------ #
    @staticmethod
    def _metrics(name: str, y_true: pd.Series, prob: np.ndarray, threshold: float) -> ModelMetrics:
        clipped = np.clip(prob, 1e-6, 1 - 1e-6)
        prediction = (prob >= threshold).astype(int)
        unique_classes = np.unique(y_true)
        roc_auc = float(roc_auc_score(y_true, prob)) if len(unique_classes) == 2 else float("nan")
        pr_auc = float(average_precision_score(y_true, prob)) if len(unique_classes) == 2 else float("nan")
        return ModelMetrics(
            model_name=name,
            roc_auc=roc_auc,
            pr_auc=pr_auc,
            precision=float(precision_score(y_true, prediction, zero_division=0)),
            recall=float(recall_score(y_true, prediction, zero_division=0)),
            f1=float(f1_score(y_true, prediction, zero_division=0)),
            brier=float(brier_score_loss(y_true, clipped)),
            log_loss=float(log_loss(y_true, clipped)),
            threshold=float(threshold),
            accuracy=float(accuracy_score(y_true, prediction)),
            n_test=int(len(y_true)),
            n_positive=int(y_true.sum()),
        )

    @staticmethod
    def best_f1_threshold(y_true: pd.Series, prob: np.ndarray) -> float:
        precision, recall, thresholds = precision_recall_curve(y_true, prob)
        f1 = np.divide(
            2 * precision * recall,
            precision + recall,
            out=np.zeros_like(precision),
            where=(precision + recall) > 0,
        )
        if len(thresholds) == 0:
            return 0.5
        return float(thresholds[int(np.argmax(f1[:-1]))])

    def _linear_coefficients(self, pipeline: Pipeline, feature_names: Sequence[str]) -> pd.Series:
        """提取标准化空间中的线性系数（校准模型取各折平均）。"""
        estimator: Any = pipeline
        if isinstance(estimator, CalibratedClassifierCV):
            coefs = []
            for calibrated in estimator.calibrated_classifiers_:
                base = getattr(calibrated, "estimator", None)
                if base is None:  # sklearn < 1.2 兼容
                    base = calibrated.base_estimator
                if hasattr(base, "named_steps") and "clf" in base.named_steps:
                    clf = base.named_steps["clf"]
                    if hasattr(clf, "coef_"):
                        coefs.append(clf.coef_.ravel())
                elif hasattr(base, "coef_"):
                    coefs.append(base.coef_.ravel())
            if not coefs:
                return pd.Series(dtype="float")
            coefficients = np.mean(np.vstack(coefs), axis=0)
        elif hasattr(estimator, "named_steps") and "clf" in estimator.named_steps:
            clf = estimator.named_steps["clf"]
            if not hasattr(clf, "coef_"):
                return pd.Series(dtype="float")
            coefficients = clf.coef_.ravel()
        else:
            return pd.Series(dtype="float")
        return pd.Series(coefficients, index=list(feature_names), name="标准化系数")

    @staticmethod
    def _tree_importance(pipeline: Pipeline, feature_names: Sequence[str]) -> pd.Series:
        clf = pipeline.named_steps.get("clf") if hasattr(pipeline, "named_steps") else None
        if clf is not None and hasattr(clf, "feature_importances_"):
            values = clf.feature_importances_
            return pd.Series(values, index=list(feature_names), name="特征重要性").sort_values(ascending=False)
        return pd.Series(dtype="float")

    # ------------------------------------------------------------------ #
    # 主流程：候选模型对比 + 选型
    # ------------------------------------------------------------------ #
    def run_model_suite(
        self, users: pd.DataFrame, feature_columns: Sequence[str], task_name: str
    ) -> ModelSuite:
        data = self._exclude_reserved_users(users)
        matrix, label = self.build_matrix(data, feature_columns)
        train_index, test_index = train_test_split(
            np.arange(len(matrix)),
            test_size=self.test_size,
            stratify=label,
            random_state=self.seed,
        )
        x_train, x_test = matrix.iloc[train_index], matrix.iloc[test_index]
        y_train, y_test = label.iloc[train_index], label.iloc[test_index]
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.seed)

        rows: List[Dict[str, Any]] = []
        fitted: Dict[str, Pipeline] = {}
        probabilities: Dict[str, np.ndarray] = {}
        for name, pipeline in self._candidate_pipelines().items():
            self.logger.info("[%s] 训练候选模型：%s（%d 折交叉验证）", task_name, name, self.cv_folds)
            cv_pr = cross_val_score(
                pipeline, x_train, y_train, cv=cv, scoring="average_precision", error_score=np.nan
            )
            cv_roc = cross_val_score(pipeline, x_train, y_train, cv=cv, scoring="roc_auc", error_score=np.nan)
            pipeline.fit(x_train, y_train)
            prob = pipeline.predict_proba(x_test)[:, 1]
            metrics = self._metrics(name, y_test, prob, threshold=0.5)
            rows.append(
                {
                    "模型": name,
                    "5折CV_PR_AUC均值": float(np.nanmean(cv_pr)),
                    "5折CV_PR_AUC标准差": float(np.nanstd(cv_pr)),
                    "5折CV_ROC_AUC均值": float(np.nanmean(cv_roc)),
                    "测试集ROC_AUC": metrics.roc_auc,
                    "测试集PR_AUC": metrics.pr_auc,
                    "测试集精确率": metrics.precision,
                    "测试集召回率": metrics.recall,
                    "测试集F1": metrics.f1,
                    "测试集Brier": metrics.brier,
                    "是否入选": False,
                }
            )
            fitted[name] = pipeline
            probabilities[name] = prob

        comparison = pd.DataFrame(rows)
        best_row = comparison.sort_values(
            ["5折CV_PR_AUC均值", "测试集PR_AUC"], ascending=False
        ).iloc[0]
        best_name = str(best_row["模型"])
        logistic_row = comparison.loc[comparison["模型"] == MODEL_LOGISTIC].iloc[0]
        pr_gap = float(best_row["5折CV_PR_AUC均值"] - logistic_row["5折CV_PR_AUC均值"])
        if best_name != MODEL_LOGISTIC and pr_gap <= 0.02:
            self.logger.info(
                "[%s] 树模型 PR-AUC 领先仅 %.4f（≤0.02），按“可解释优先”选择 Logistic", task_name, pr_gap
            )
            best_name = MODEL_LOGISTIC
        comparison.loc[comparison["模型"] == best_name, "是否入选"] = True

        best_pipeline = fitted[best_name]
        best_metrics = self._metrics(best_name, y_test, probabilities[best_name], threshold=0.5)
        best_metrics.extra["最佳F1阈值"] = self.best_f1_threshold(y_test, probabilities[best_name])
        bundle = ModelBundle(
            model_name=best_name,
            pipeline=best_pipeline,
            feature_names=list(feature_columns),
            metrics=best_metrics,
            tier_thresholds={},
            positive_rate=float(y_train.mean()),
            notes=f"{task_name} 入选模型；5折CV PR-AUC={best_row['5折CV_PR_AUC均值']:.4f}",
        )
        coefficients = self._linear_coefficients(best_pipeline, feature_columns)
        importance = self._tree_importance(best_pipeline, feature_columns)
        if importance.empty:
            # 梯度提升树等模型不提供 feature_importances_，改用测试集置换重要性
            try:
                from sklearn.inspection import permutation_importance

                perm = permutation_importance(
                    best_pipeline,
                    x_test,
                    y_test,
                    scoring="average_precision",
                    n_repeats=3,
                    random_state=self.seed,
                    n_jobs=1,
                )
                importance = (
                    pd.Series(perm.importances_mean, index=list(feature_columns), name="特征重要性")
                    .sort_values(ascending=False)
                    .round(6)
                )
                self.logger.info("[%s] 使用置换重要性解释 %s", task_name, best_name)
            except Exception as exc:  # pragma: no cover - 解释失败不影响主流程
                self.logger.warning("[%s] 置换重要性计算失败：%s", task_name, exc)
        bundle.feature_importance = {str(name): float(value) for name, value in importance.items()}
        self.logger.info(
            "[%s] 入选 %s：测试集 PR-AUC=%.4f ROC-AUC=%.4f F1=%.4f",
            task_name,
            best_name,
            best_metrics.pr_auc,
            best_metrics.roc_auc,
            best_metrics.f1,
        )
        return ModelSuite(
            task=task_name,
            comparison=comparison,
            bundle=bundle,
            y_test=y_test.reset_index(drop=True),
            prob_test=probabilities[best_name],
            test_users=data.loc[test_index, C.COL_USER_ID].reset_index(drop=True),
            coefficients=coefficients,
            feature_importance=importance,
            extra={"train_index": train_index, "test_index": test_index, "pr_gap": pr_gap},
        )

    # ------------------------------------------------------------------ #
    # 问题4：校准 + 阈值 + 分层
    # ------------------------------------------------------------------ #
    def train_q4_final(self, users: pd.DataFrame) -> ModelSuite:
        data = self._exclude_reserved_users(users)
        features = select_features(data, "q4")
        suite = self.run_model_suite(data, features, "Q4")
        matrix, label = self.build_matrix(data, features)
        train_index = suite.extra["train_index"]
        test_index = suite.extra["test_index"]
        x_train, x_test = matrix.iloc[train_index], matrix.iloc[test_index]
        y_train, y_test = label.iloc[train_index], label.iloc[test_index]

        base_pipeline = suite.bundle.pipeline
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.seed)
        oof = cross_val_predict(base_pipeline, x_train, y_train, cv=cv, method="predict_proba")[:, 1]
        threshold = self.best_f1_threshold(y_train, oof)
        self.logger.info("Q4 训练集 OOF 最大 F1 阈值：%.4f", threshold)

        method = str(self.calibration_cfg.get("method", "sigmoid"))
        calib_cv = int(self.calibration_cfg.get("cv", 5))
        calibrated = CalibratedClassifierCV(base_pipeline, method=method, cv=calib_cv).fit(x_train, y_train)
        prob_cal = calibrated.predict_proba(x_test)[:, 1]
        # 校准后概率量纲已变化，阈值需在同一量纲上重新确定
        calibrated_train_scores = calibrated.predict_proba(x_train)[:, 1]
        calibrated_threshold = self.best_f1_threshold(y_train, calibrated_train_scores)
        calibrated_metrics = self._metrics(
            "校准后模型", y_test, prob_cal, threshold=calibrated_threshold
        )
        uncalibrated_metrics = suite.bundle.metrics

        # 分层阈值：训练集全体用户分数的 0.90 / 0.70 分位（预测前固定）
        train_scores = calibrated.predict_proba(x_train)[:, 1]
        thresholds = {
            "high": float(np.quantile(train_scores, self.config.business.tier_high_quantile)),
            "medium": float(np.quantile(train_scores, self.config.business.tier_medium_quantile)),
            "break_even": float(self.config.business.break_even_probability),
            "f1": float(calibrated_threshold),
        }

        # 全量重训（供附件2 与 Top100 打分）
        final_pipeline = CalibratedClassifierCV(base_pipeline, method=method, cv=calib_cv).fit(matrix, label)
        all_scores = final_pipeline.predict_proba(matrix)[:, 1]

        bundle = ModelBundle(
            model_name=suite.bundle.model_name,
            pipeline=final_pipeline,
            feature_names=list(features),
            metrics=calibrated_metrics,
            tier_thresholds=thresholds,
            positive_rate=float(label.mean()),
            calibration_method=method,
            notes=(
                f"Q4 校准模型；入选基础模型 {suite.bundle.model_name}；"
                f"未校准 OOF F1 阈值 {threshold:.4f}；校准后 F1 阈值 {calibrated_threshold:.4f}；sigmoid 校准（cv={calib_cv}）"
            ),
        )
        coefficients = self._linear_coefficients(final_pipeline, features)
        importance = suite.feature_importance
        bundle.feature_importance = {str(name): float(value) for name, value in importance.items()}
        self.logger.info(
            "Q4 校准后：Brier=%.4f PR-AUC=%.4f ROC-AUC=%.4f（阈值 %.4f）",
            calibrated_metrics.brier,
            calibrated_metrics.pr_auc,
            calibrated_metrics.roc_auc,
            threshold,
        )
        return ModelSuite(
            task="Q4",
            comparison=suite.comparison,
            bundle=bundle,
            y_test=y_test.reset_index(drop=True),
            prob_test=prob_cal,
            test_users=data.loc[test_index, C.COL_USER_ID].reset_index(drop=True),
            coefficients=coefficients,
            feature_importance=importance,
            extra={
                "threshold": threshold,
                "calibrated_threshold": calibrated_threshold,
                "calibrated_metrics": calibrated_metrics,
                "uncalibrated_metrics": uncalibrated_metrics,
                "all_scores": all_scores,
                "scored_users": data[C.COL_USER_ID].to_numpy(),
                "oof_scores": oof,
            },
        )


__all__ = ["ModelingServiceImpl", "ModelSuite", "COMPARISON_COLUMNS"]
