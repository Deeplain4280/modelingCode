# -*- coding: utf-8 -*-
"""问题 4 的预测服务实现：潜力打分、分层、附件2 预测、Top100 与营销矩阵。"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..config import AppConfig
from ..domain import constants as C
from ..domain.entities import ModelBundle
from ..domain.enums import PotentialTier
from ..repository.artifact_repository import ArtifactRepository
from ..service import PredictionService
from .feature_labels import feature_label


class PredictionServiceImpl(PredictionService):
    """基于校准概率的潜力评分与营销策略生成。"""

    def __init__(
        self,
        config: AppConfig,
        artifact_repository: ArtifactRepository,
        logger: Optional[logging.Logger] = None,
    ):
        self.config = config
        self.artifacts = artifact_repository
        self.logger = logger or logging.getLogger(__name__)
        self.top_n = config.runtime.top_n
        self.business = config.business

    # ------------------------------------------------------------------ #
    # 分层规则
    # ------------------------------------------------------------------ #
    def tier_thresholds_from_scores(self, scores: np.ndarray) -> Dict[str, float]:
        scores = np.asarray(scores, dtype=float)
        scores = scores[~np.isnan(scores)]
        if scores.size == 0:
            raise ValueError("用于确定分层阈值的分数为空")
        high = float(np.quantile(scores, self.business.tier_high_quantile))
        medium = float(np.quantile(scores, self.business.tier_medium_quantile))
        return {
            "high": high,
            "medium": medium,
            "break_even": float(self.business.break_even_probability),
        }

    def assign_tiers(self, probabilities: np.ndarray, thresholds: Dict[str, float]) -> List[str]:
        high = float(thresholds.get("high", 0.9))
        medium = float(thresholds.get("medium", 0.7))
        tiers: List[str] = []
        for probability in np.asarray(probabilities, dtype=float):
            if np.isnan(probability):
                tiers.append(PotentialTier.LOW.value)
            elif probability >= high:
                tiers.append(PotentialTier.HIGH.value)
            elif probability >= medium:
                tiers.append(PotentialTier.MEDIUM.value)
            else:
                tiers.append(PotentialTier.LOW.value)
        return tiers

    # ------------------------------------------------------------------ #
    # 打分与可解释驱动
    # ------------------------------------------------------------------ #
    def score_users(self, bundle: ModelBundle, frame: pd.DataFrame) -> pd.DataFrame:
        probabilities = bundle.predict_proba(frame)
        thresholds = bundle.tier_thresholds or self.tier_thresholds_from_scores(probabilities)
        result = pd.DataFrame(
            {
                C.COL_USER_ID: frame[C.COL_USER_ID].values,
                "预测购买概率": probabilities,
                "潜力评分": probabilities * 100.0,
                "潜力等级": self.assign_tiers(probabilities, thresholds),
            }
        )
        contributions = self._contributions(bundle, frame)
        if contributions is not None:
            result["主要正向驱动"] = [self._top_drivers(row) for _, row in contributions.iterrows()]
        else:
            result["主要正向驱动"] = "非线性模型，按整体特征重要性解释"
        result["建议动作"] = [self._action_for_tier(tier) for tier in result["潜力等级"]]
        return result

    def _first_base_estimator(self, pipeline: Any) -> Any:
        from sklearn.calibration import CalibratedClassifierCV

        if isinstance(pipeline, CalibratedClassifierCV):
            for calibrated in pipeline.calibrated_classifiers_:
                base = getattr(calibrated, "estimator", None)
                if base is None:  # sklearn < 1.2
                    base = getattr(calibrated, "base_estimator", None)
                if base is not None:
                    return base
            return None
        return pipeline

    def _linear_coefficients(self, pipeline: Any, n_features: int) -> Optional[np.ndarray]:
        from sklearn.calibration import CalibratedClassifierCV

        candidates = []
        if isinstance(pipeline, CalibratedClassifierCV):
            for calibrated in pipeline.calibrated_classifiers_:
                base = getattr(calibrated, "estimator", None)
                if base is None:
                    base = getattr(calibrated, "base_estimator", None)
                candidates.append(base)
        else:
            candidates.append(pipeline)
        coefficients = [
            base.named_steps["clf"].coef_.ravel()
            for base in candidates
            if base is not None
            and hasattr(base, "named_steps")
            and "clf" in base.named_steps
            and hasattr(base.named_steps["clf"], "coef_")
        ]
        if not coefficients:
            return None
        mean = np.mean(np.vstack(coefficients), axis=0)
        return mean if mean.shape[0] == n_features else None

    def _contributions(self, bundle: ModelBundle, frame: pd.DataFrame) -> Optional[pd.DataFrame]:
        coefficients = self._linear_coefficients(bundle.pipeline, len(bundle.feature_names))
        if coefficients is None:
            return self._perturbation_contributions(bundle, frame)
        base = self._first_base_estimator(bundle.pipeline)
        matrix = frame.reindex(columns=bundle.feature_names)
        steps = getattr(base, "named_steps", {})
        if "imputer" in steps:
            matrix = steps["imputer"].transform(matrix)
        if "scaler" in steps:
            matrix = steps["scaler"].transform(matrix)
        array = np.asarray(matrix, dtype=float)
        return pd.DataFrame(array * coefficients, columns=bundle.feature_names, index=frame.index)

    def _perturbation_contributions(self, bundle: ModelBundle, frame: pd.DataFrame) -> Optional[pd.DataFrame]:
        """非线性模型的近似个体解释：将全局重要特征逐一置为总体中位数，观察概率变化。"""
        importance = getattr(bundle, "feature_importance", None) or {}
        if not importance:
            return None
        ranked = sorted(importance.items(), key=lambda item: float(item[1]), reverse=True)
        selected = [name for name, _ in ranked if name in frame.columns][:8]
        if not selected:
            return None
        base = bundle.predict_proba(frame)
        contributions = {}
        for name in selected:
            column = frame[name]
            fill = float(column.median()) if hasattr(column, "median") else 0.0
            modified = frame.copy()
            modified[name] = fill
            contributions[name] = base - bundle.predict_proba(modified)
        return pd.DataFrame(contributions, index=frame.index)

    @staticmethod
    def _top_drivers(row: pd.Series, top_k: int = 3) -> str:
        positive = row[row > 0].sort_values(ascending=False).head(top_k)
        if positive.empty:
            return "无显著正向因素"
        return "；".join([f"{feature_label(name)}(+{value:.2f})" for name, value in positive.items()])

    @staticmethod
    def _action_for_tier(tier: str) -> str:
        playbook = C.MARKETING_PLAYBOOK.get(tier, C.MARKETING_PLAYBOOK["低潜力"])
        return f"{playbook['触达渠道']}，{playbook['建议频次']}"

    # ------------------------------------------------------------------ #
    # 附件2 预测
    # ------------------------------------------------------------------ #
    def predict_attachment2(self, bundle: ModelBundle, attachment2: pd.DataFrame) -> pd.DataFrame:
        expected = int(self.business.attachment2_expected_users)
        if len(attachment2) != expected:
            self.logger.warning("附件2 预期 %d 人，实际 %d 人", expected, len(attachment2))
        scored = self.score_users(bundle, attachment2)
        scored["附件2序号"] = np.arange(1, len(scored) + 1)
        if "in_target_table" in attachment2.columns:
            scored["是否在附件1目标用户中"] = np.where(
                attachment2["in_target_table"].to_numpy() == 1, "是", "否"
            )
        if "known_target" in attachment2.columns:
            known = attachment2["known_target"].to_numpy()
            scored["附件1购买标记（如有）"] = np.where(
                known == 1, "已购买", np.where(known == 0, "未购买", "不在附件1")
            )
        scored = scored.sort_values(["潜力评分", C.COL_USER_ID], ascending=[False, True]).reset_index(drop=True)
        scored.insert(0, "排名", np.arange(1, len(scored) + 1))
        ordered = [
            "排名",
            "附件2序号",
            C.COL_USER_ID,
            "预测购买概率",
            "潜力评分",
            "潜力等级",
            "主要正向驱动",
            "建议动作",
        ]
        ordered += [column for column in scored.columns if column not in ordered]
        return scored[ordered]

    # ------------------------------------------------------------------ #
    # 未购用户 Top-N
    # ------------------------------------------------------------------ #
    def rank_unpurchased_topn(
        self, bundle: ModelBundle, users: pd.DataFrame, top_n: Optional[int] = None
    ) -> pd.DataFrame:
        top_n = int(top_n or self.top_n)
        candidates = users
        if "is_attachment2" in candidates.columns:
            candidates = candidates.loc[candidates["is_attachment2"] == 0]
        candidates = candidates.loc[candidates["target"] == 0]
        scored = self.score_users(bundle, candidates)
        scored = scored.sort_values(["潜力评分", C.COL_USER_ID], ascending=[False, True]).head(top_n)
        scored.insert(0, "排名", np.arange(1, len(scored) + 1))
        return scored.reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # 营销矩阵
    # ------------------------------------------------------------------ #
    def build_marketing_plan(
        self, scored_users: pd.DataFrame, personas: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        frame = scored_users.copy()
        if "画像标签" not in frame.columns:
            frame["画像标签"] = "综合型"
        tier_order = {PotentialTier.HIGH.value: 0, PotentialTier.MEDIUM.value: 1, PotentialTier.LOW.value: 2}
        rows: List[Dict[str, Any]] = []
        for tier, tier_frame in frame.groupby("潜力等级"):
            playbook = C.MARKETING_PLAYBOOK.get(tier, C.MARKETING_PLAYBOOK["低潜力"])
            tier_size = max(1, len(tier_frame))
            for persona, group in tier_frame.groupby("画像标签"):
                share = len(group) / tier_size
                known = group.dropna(subset=["target"]) if "target" in group.columns else group.iloc[0:0]
                known_count = int(known["target"].sum()) if len(known) else 0
                rows.append(
                    {
                        "潜力等级": tier,
                        "画像标签": persona,
                        "人群规模": len(group),
                        "占该层比例": share,
                        "平均潜力评分": float(group["潜力评分"].mean()),
                        "平均预测购买概率": float(group["预测购买概率"].mean()),
                        "已知购买率": float(known["target"].mean()) if len(known) else float("nan"),
                        "已知购买人数": known_count,
                        "待触达人数": len(group) - known_count,
                        "层内预算占比": playbook["预算占比"],
                        "该人群预算占比": playbook["预算占比"] * share,
                        "触达渠道": playbook["触达渠道"],
                        "建议频次": playbook["建议频次"],
                        "创意方向": playbook["创意方向"],
                        "核心指标": playbook["核心指标"],
                    }
                )
        plan = pd.DataFrame(rows)
        if plan.empty:
            return plan, personas.copy()
        plan["_tier_order"] = plan["潜力等级"].map(tier_order).fillna(9)
        plan = plan.sort_values(["_tier_order", "人群规模"], ascending=[True, False]).drop(
            columns="_tier_order"
        )
        plan = plan.reset_index(drop=True)

        if personas is None or personas.empty:
            persona_summary = pd.DataFrame(columns=["画像标签", "人群规模"])
        else:
            persona_summary = personas.reset_index() if personas.index.name else personas.copy()
            stats = frame.groupby("画像标签").agg(
                总人数=("潜力评分", "size"),
                平均潜力评分=("潜力评分", "mean"),
                高潜力人数=("潜力等级", lambda series: int((series == PotentialTier.HIGH.value).sum())),
            )
            stats["高潜力占比"] = stats["高潜力人数"] / stats["总人数"]
            persona_summary = persona_summary.merge(
                stats, left_on="画像标签", right_index=True, how="left"
            )
        return plan, persona_summary


__all__ = ["PredictionServiceImpl"]
