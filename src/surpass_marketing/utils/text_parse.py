# -*- coding: utf-8 -*-
"""文本解析工具：行为标签长文本、手机品牌、时间特征。

行为标签格式示例::

    网络活跃指数:40,在线视频指数:19,笑话top5:xxsy.net,zongheng.com,
    常用网站top5:SNS,网页搜索,文学及电子书籍,综合视频,在线音乐

其中 ``键:值`` 以英文逗号分隔，但 ``top5`` 的值本身也以逗号分隔，
因此需要“遇到下一个 键: 之前的内容都属于当前键”的解析策略。
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

import pandas as pd

from ..domain.constants import BRAND_PATTERNS

_KEY_RE = re.compile(r"^([\u4e00-\u9fa5A-Za-z0-9_]+?(?:top5|指数)):(.*)$")
_INDEX_RE = re.compile(r"^([\u4e00-\u9fa5]+指数):(\d+(?:\.\d+)?)$")


def parse_behavior_tags(text: object) -> Dict[str, object]:
    """把单条行为标签长文本解析为 ``{"indices": {...}, "top5": {键: [值, ...]}}``。

    >>> parse_behavior_tags("网络活跃指数:40,常用网站top5:SNS,网页搜索,网页搜索top5:baidu.com")
    {'indices': {'网络活跃指数': 40.0}, 'top5': {'常用网站top5': ['SNS', '网页搜索'], '网页搜索top5': ['baidu.com']}}
    """
    result: Dict[str, object] = {"indices": {}, "top5": {}}
    if not isinstance(text, str) or not text.strip():
        return result

    indices: Dict[str, float] = {}
    top5: Dict[str, List[str]] = {}
    current_key: Optional[str] = None

    for raw_token in text.split(","):
        token = raw_token.strip()
        if not token:
            continue
        match = _KEY_RE.match(token)
        if match:
            key, value = match.group(1), match.group(2).strip()
            if key.endswith("指数"):
                current_key = None
                try:
                    indices[key] = float(value)
                except ValueError:
                    continue
            else:
                current_key = key
                top5.setdefault(key, [])
                if value:
                    top5[key].append(value)
        elif current_key is not None:
            top5.setdefault(current_key, []).append(token)
    result["indices"] = indices
    result["top5"] = top5
    return result


def match_brand(model_text: object, patterns: Optional[Dict[str, List[str]]] = None) -> Optional[str]:
    """按优先级返回型号文本命中的首个手机品牌（英文键）。

    一个型号文本可能同时提到多个品牌（例如对比贴），这里只保留优先级最高的
    一个，使品牌份额之和为 1，便于解释为“品牌关注结构”。
    """
    if not isinstance(model_text, str) or not model_text.strip():
        return None
    text = model_text.lower()
    patterns = patterns or BRAND_PATTERNS
    for brand, keywords in patterns.items():
        for keyword in keywords:
            if keyword.lower() in text:
                return brand
    return None


def match_keyword_groups(text: object, groups: Dict[str, List[str]]) -> List[str]:
    """返回文本命中的关键词组名列表（用于题材/品类/渠道归类）。"""
    if not isinstance(text, str) or not text.strip():
        return []
    text = text.lower()
    hits = []
    for group, keywords in groups.items():
        for keyword in keywords:
            if keyword.lower() in text:
                hits.append(group)
                break
    return hits


def add_time_features(frame: pd.DataFrame, column: str, prefix: str) -> pd.DataFrame:
    """从时间戳列派生小时、星期、是否周末等特征（不修改原表）。"""
    out = frame.copy()
    ts = pd.to_datetime(out[column], errors="coerce")
    out[f"{prefix}_hour"] = ts.dt.hour
    out[f"{prefix}_weekday"] = ts.dt.weekday
    out[f"{prefix}_is_weekend"] = (ts.dt.weekday >= 5).astype("float")
    # 夜间活跃：20:00-次日 02:00
    hour = ts.dt.hour
    out[f"{prefix}_is_night"] = (((hour >= 20) | (hour < 2)) & hour.notna()).astype("float")
    return out


__all__ = [
    "parse_behavior_tags",
    "match_brand",
    "match_keyword_groups",
    "add_time_features",
]
