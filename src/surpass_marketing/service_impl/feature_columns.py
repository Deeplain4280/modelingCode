# -*- coding: utf-8 -*-
"""特征列口径：问题2/3/4 分别使用哪些列，统一在此维护。

设计原则
--------
* **问题2（基本行为特征）**：人口学属性、7 个行为指数、活跃时间、行为量级、
  数据覆盖标记。回答“谁更可能买、活跃度是否提升购买”。
* **问题3（个人偏好）**：品牌结构、品类关注、视频题材、触媒渠道、促销触点。
  回答“偏好什么的人更可能买”，并据此构造 8 个可解释偏好指标。
* **问题4（综合预测）**：问题2 + 问题3 原始特征，交给模型做潜力评分。
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import pandas as pd

from ..domain import constants as C

# 元信息列，绝不进入模型
META_COLUMNS = {
    C.COL_USER_ID,
    "target",
    "state_set",
    "track_time",
}

Q2_PREFIXES: Sequence[str] = (
    "age",
    "gender",
    "edu",
    "occ_",
    "idx_",
    "track_",
    "vol_",
    "has_",
)

Q3_PREFIXES: Sequence[str] = (
    "brand_",
    "pref_",
    "tag_shop_",
    "tag_genre_",
    "tag_site_",
    "tag_domain_",
    "tag_promo_",
    "vid_",
    "genre_",
    "media_ch_",
    "media_domain_",
)


def select_features(frame: pd.DataFrame, group: str) -> List[str]:
    """按分组返回特征列名（保持原表顺序，便于复现）。"""
    if group == "q2":
        prefixes = Q2_PREFIXES
    elif group == "q3":
        prefixes = Q3_PREFIXES
    elif group == "q4":
        prefixes = tuple(Q2_PREFIXES) + tuple(Q3_PREFIXES)
    else:  # pragma: no cover - 防御式编程
        raise ValueError(f"未知特征分组: {group}")

    return [
        column
        for column in frame.columns
        if column not in META_COLUMNS and any(column.startswith(p) for p in prefixes)
    ]


def preference_indicator_columns(frame: pd.DataFrame) -> Dict[str, List[str]]:
    """返回每个偏好指标实际可用的成分列（缺失的自动剔除）。"""
    resolved: Dict[str, List[str]] = {}
    for indicator, columns in C.PREFERENCE_INDICATORS.items():
        available = [column for column in columns if column in frame.columns]
        if available:
            resolved[indicator] = available
    return resolved


__all__ = [
    "META_COLUMNS",
    "Q2_PREFIXES",
    "Q3_PREFIXES",
    "select_features",
    "preference_indicator_columns",
]
