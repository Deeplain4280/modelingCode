# -*- coding: utf-8 -*-
"""特征口径测试：问题2/3/4 的特征分组与泄漏防护。"""
from __future__ import annotations

import pandas as pd
import pytest

from surpass_marketing.service_impl.feature_columns import (
    META_COLUMNS,
    select_features,
)


@pytest.fixture()
def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "用户编号": [1, 2],
            "target": [0, 1],
            "state_set": ["浏览", "购买"],
            "track_time": ["2015-11-02 10:00", "2015-11-03 20:00"],
            "age": [2, 3],
            "gender": [1, 0],
            "occ_0": [1, 0],
            "idx_shopping": [10.0, 55.0],
            "track_hour": [10, 20],
            "vol_ecom_n": [3, 12],
            "has_identity": [1, 1],
            "brand_apple_share": [0.2, 0.6],
            "pref_phone_share": [0.5, 0.9],
            "tag_shop_mobile": [0.4, 0.8],
            "vid_l1_tv": [0.1, 0.7],
            "media_ch_video": [0.3, 0.6],
            "is_attachment2": [0, 0],
        }
    )


def test_select_q2_excludes_preference(frame):
    columns = select_features(frame, "q2")
    assert "age" in columns and "idx_shopping" in columns
    assert "vol_ecom_n" in columns and "has_identity" in columns
    assert not any(column.startswith("brand_") for column in columns)


def test_select_q3_excludes_demographics(frame):
    columns = select_features(frame, "q3")
    assert "brand_apple_share" in columns
    assert "pref_phone_share" in columns
    assert "media_ch_video" in columns
    assert "age" not in columns and "idx_shopping" not in columns


def test_select_q4_is_union(frame):
    columns = select_features(frame, "q4")
    assert "age" in columns
    assert "brand_apple_share" in columns


def test_meta_columns_never_selected(frame):
    for group in ("q2", "q3", "q4"):
        for meta in META_COLUMNS:
            assert meta not in select_features(frame, group)


def test_unknown_group_raises(frame):
    with pytest.raises(ValueError):
        select_features(frame, "q9")
