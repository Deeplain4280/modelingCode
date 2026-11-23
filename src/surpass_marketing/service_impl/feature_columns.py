# -*- coding: utf-8 -*-
"""特征列口径：问题2/3/4 分别使用哪些列，统一在此维护。

设计原则
--------
* **问题2（基本行为特征）**：人口学属性、7 个行为指数、行为量级、数据覆盖标记。
  回答“谁更可能买、活跃度是否提升购买”。
* **问题3（个人偏好）**：品牌结构、品类关注、视频题材、触媒渠道、促销触点。
  回答“偏好什么的人更可能买”，并据此构造 8 个可解释偏好指标。
* **问题4（综合预测）**：问题2 + 问题3 原始特征，交给模型做潜力评分。
* **禁用字段**：跟踪状态表派生的 ``track_*``（尤其购买后才有值的字段）、
  用户编号、标签、状态/时间元信息一律不得进入模型。``assert_feature_safety``
  在建模入口做硬校验，违反即抛错。
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
    "purchase_time",
    "cutoff_time",
    "post_purchase_activity",
    "a2_cutoff_time",
    "is_attachment2",
    "known_target",
    "in_target_table",
}

# 明确禁用：来自跟踪状态表的时间/条数特征（部分取值只可能在购买后产生）
FORBIDDEN_FEATURE_NAMES = set(META_COLUMNS)
FORBIDDEN_FEATURE_PREFIXES: Sequence[str] = ("track_",)

# 不进入预测模型的列：视频表自 2015-11-13 才有记录，购买者按首次购买截断后
# 绝大多数购买前无可观测视频；若保留会退化为“观测覆盖/窗口长度”代理而非偏好。
# 视频仅用于问题一的数据质量描述，不参与 Q2/Q3/Q4 建模。
EXCLUDED_FEATURE_NAMES = {
    "has_video",
    "vol_total",
    "vol_source_count",
    "vol_video_n",
    "vol_video_days",
    "vol_video_unique",
    "vol_video_night_share",
    "vol_video_weekend_share",
    "vol_video_hour_mean",
}
EXCLUDED_FEATURE_PREFIXES: Sequence[str] = ("vid_", "genre_")

Q2_PREFIXES: Sequence[str] = (
    "age",
    "gender",
    "edu",
    "occ_",
    "idx_",
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
    "media_ch_",
    "media_domain_",
)


def forbidden_columns(columns: Sequence[str]) -> List[str]:
    """返回列名中属于禁用标签泄漏字段的项。"""
    return [
        column
        for column in columns
        if column in FORBIDDEN_FEATURE_NAMES
        or any(str(column).startswith(prefix) for prefix in FORBIDDEN_FEATURE_PREFIXES)
    ]


def assert_feature_safety(columns: Sequence[str]) -> None:
    """建模入口硬校验：禁用字段一旦进入特征矩阵立即报错。"""
    bad = forbidden_columns(columns)
    if bad:
        raise ValueError(
            "特征矩阵包含禁用字段（标签泄漏风险）：" + "、".join(str(column) for column in bad[:10])
        )


def excluded_columns(columns: Sequence[str]) -> List[str]:
    """返回因时序覆盖不足而不进入预测模型的列。"""
    return [
        column
        for column in columns
        if column in EXCLUDED_FEATURE_NAMES
        or any(str(column).startswith(prefix) for prefix in EXCLUDED_FEATURE_PREFIXES)
    ]


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

    selected = [
        column
        for column in frame.columns
        if column not in META_COLUMNS
        and any(column.startswith(p) for p in prefixes)
        and column not in EXCLUDED_FEATURE_NAMES
        and not any(str(column).startswith(p) for p in EXCLUDED_FEATURE_PREFIXES)
    ]
    assert_feature_safety(selected)
    return selected


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
    "FORBIDDEN_FEATURE_NAMES",
    "FORBIDDEN_FEATURE_PREFIXES",
    "EXCLUDED_FEATURE_NAMES",
    "EXCLUDED_FEATURE_PREFIXES",
    "Q2_PREFIXES",
    "Q3_PREFIXES",
    "forbidden_columns",
    "excluded_columns",
    "assert_feature_safety",
    "select_features",
    "preference_indicator_columns",
]
