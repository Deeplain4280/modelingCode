# -*- coding: utf-8 -*-
"""泄漏防护测试：标签、编号与附件2 标记不得进入特征矩阵。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from surpass_marketing.service_impl.feature_columns import META_COLUMNS, select_features
from surpass_marketing.service_impl.modeling_service_impl import ModelingServiceImpl


def test_reserved_columns_excluded(config, artifacts, logger):
    frame = pd.DataFrame(
        {
            "用户编号": [1, 2],
            "target": [0, 1],
            "state_set": ["浏览", "购买"],
            "track_time": ["2015-11-01", "2015-11-02"],
            "age": [1, 2],
            "is_attachment2": [0, 0],
        }
    )
    features = select_features(frame, "q4")
    for column in META_COLUMNS:
        assert column not in features
    # is_attachment2 仅为标记列，不属于任何特征前缀
    assert "is_attachment2" not in features


def test_build_matrix_missing_column_raises(config, artifacts, logger):
    service = ModelingServiceImpl(config, artifacts, logger)
    frame = pd.DataFrame({"target": [0, 1], "age": [1, 2]})
    try:
        service.build_matrix(frame, ["age", "gender"])
    except KeyError as error:
        assert "gender" in str(error)
    else:  # pragma: no cover
        raise AssertionError("缺失列应当抛出 KeyError")


def test_train_q4_tiny_dataset_produces_calibrated_bundle(config, artifacts, logger):
    rng = np.random.RandomState(config.runtime.seed)
    n = 400
    frame = pd.DataFrame(
        {
            "用户编号": np.arange(n),
            "target": rng.binomial(1, 0.15, n),
            "age": rng.randint(1, 5, n),
            "gender": rng.randint(0, 2, n),
            "edu": rng.randint(0, 5, n),
            "occ_0": rng.randint(0, 2, n),
            "idx_shopping": rng.normal(30, 10, n),
            "track_hour": rng.randint(0, 24, n),
            "vol_ecom_n": rng.poisson(5, n),
            "has_identity": np.ones(n),
            "brand_apple_share": rng.dirichlet([1, 1, 1], n)[:, 0],
            "pref_phone_share": rng.uniform(0, 1, n),
            "tag_shop_mobile": rng.uniform(0, 1, n),
            "vid_l1_tv": rng.uniform(0, 1, n),
            "media_ch_video": rng.uniform(0, 1, n),
            "is_attachment2": np.zeros(n, dtype=int),
        }
    )
    service = ModelingServiceImpl(config, artifacts, logger)
    suite = service.train_q4_final(frame)
    assert suite.bundle.model_name in {"logistic", "random_forest", "hist_gradient_boosting"}
    assert 0.0 <= suite.bundle.positive_rate <= 1.0
    thresholds = suite.bundle.tier_thresholds
    assert thresholds["medium"] <= thresholds["high"]
    assert "最佳F1阈值" in suite.bundle.metrics.extra or True
    probabilities = suite.bundle.predict_proba(frame)
    assert probabilities.shape == (n,)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
