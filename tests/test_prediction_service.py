# -*- coding: utf-8 -*-
"""问题4 预测与营销策略测试（使用轻量 Dummy 模型，不依赖原始 Excel）。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from surpass_marketing.domain.enums import PotentialTier
from surpass_marketing.service_impl.prediction_service_impl import PredictionServiceImpl


class DummyBundle:
    """仅返回预设概率的模型包替身（支持按用户编号映射）。"""

    model_name = "dummy"
    feature_names: list = []
    pipeline = object()
    tier_thresholds = {"high": 0.8, "medium": 0.5, "break_even": 0.05}
    positive_rate = 0.1
    metrics = None

    def __init__(self, probabilities):
        if isinstance(probabilities, dict):
            self._mapping = {int(key): float(value) for key, value in probabilities.items()}
        else:
            self._mapping = None
            self._probabilities = np.asarray(probabilities, dtype=float)

    def predict_proba(self, frame):
        if self._mapping is not None:
            return np.asarray([self._mapping[int(uid)] for uid in frame["用户编号"]], dtype=float)
        assert len(frame) == len(self._probabilities)
        return self._probabilities


@pytest.fixture()
def service(config, artifacts, logger):
    return PredictionServiceImpl(config, artifacts, logger)


def test_thresholds_from_scores(service):
    thresholds = service.tier_thresholds_from_scores(np.linspace(0.0, 1.0, 101))
    assert thresholds["high"] == pytest.approx(0.9)
    assert thresholds["medium"] == pytest.approx(0.7)
    assert thresholds["break_even"] == pytest.approx(0.05)


def test_assign_tiers(service):
    tiers = service.assign_tiers(
        np.array([0.95, 0.72, 0.30]), {"high": 0.9, "medium": 0.7}
    )
    assert tiers == [
        PotentialTier.HIGH.value,
        PotentialTier.MEDIUM.value,
        PotentialTier.LOW.value,
    ]


def test_rank_unpurchased_topn_filters_and_tie_break(service):
    users = pd.DataFrame(
        {
            "用户编号": [5, 3, 9, 1, 7],
            "target": [0, 0, 1, 0, 0],
            "is_attachment2": [0, 0, 0, 0, 1],
        }
    )
    bundle = DummyBundle({5: 0.4, 3: 0.4, 9: 0.9, 1: 0.1, 7: 0.99})
    ranked = service.rank_unpurchased_topn(bundle, users, top_n=10)
    # 仅剔除已购买用户(9)；附件1 中未购的附件2 重叠用户(7)允许进入榜单
    assert 9 not in ranked["用户编号"].tolist()
    assert 7 in ranked["用户编号"].tolist()
    assert len(ranked) == 4
    # 同分时用户编号升序
    tied = ranked.loc[ranked["潜力评分"] == 40.0, "用户编号"].tolist()
    assert tied == [3, 5]
    assert ranked["排名"].tolist() == [1, 2, 3, 4]


def test_rank_unpurchased_respects_top_n(service):
    users = pd.DataFrame(
        {"用户编号": np.arange(1, 21), "target": np.zeros(20, dtype=int), "is_attachment2": np.zeros(20)}
    )
    bundle = DummyBundle(np.linspace(0.01, 0.99, 20))
    ranked = service.rank_unpurchased_topn(bundle, users, top_n=5)
    assert len(ranked) == 5
    assert ranked["潜力评分"].is_monotonic_decreasing


def test_predict_attachment2_has_50_rows_and_rank(service):
    frame = pd.DataFrame(
        {
            "用户编号": np.arange(100, 150),
            "in_target_table": np.zeros(50, dtype=int),
            "known_target": np.full(50, -1, dtype=int),
        }
    )
    bundle = DummyBundle(np.linspace(0.001, 0.999, 50))
    result = service.predict_attachment2(bundle, frame)
    assert len(result) == 50
    assert result["排名"].is_monotonic_increasing
    assert result["潜力评分"].is_monotonic_decreasing
    assert result["是否在附件1目标用户中"].eq("否").all()


def test_build_marketing_plan_budget_shares(service):
    scored = pd.DataFrame(
        {
            "用户编号": np.arange(1, 7),
            "预测购买概率": [0.9, 0.85, 0.75, 0.6, 0.2, 0.1],
            "潜力评分": [90, 85, 75, 60, 20, 10],
            "潜力等级": ["高潜力", "高潜力", "中潜力", "中潜力", "低潜力", "低潜力"],
            "画像标签": ["A", "B", "A", "A", "B", "A"],
            "target": [1, 0, 1, 0, 0, 0],
        }
    )
    personas = pd.DataFrame({"画像标签": ["A", "B"], "人数": [4, 2]})
    plan, summary = service.build_marketing_plan(scored, personas)
    assert not plan.empty
    assert plan["该人群预算占比"].sum() == pytest.approx(1.0)
    assert set(summary["画像标签"]) == {"A", "B"}
    assert plan["潜力等级"].iloc[0] == "高潜力"
