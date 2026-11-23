# -*- coding: utf-8 -*-
"""文本解析工具测试：行为标签、品牌匹配、时间特征。"""
from __future__ import annotations

import pandas as pd

from surpass_marketing.utils.text_parse import (
    add_time_features,
    match_brand,
    match_keyword_groups,
    parse_behavior_tags,
)


def test_parse_behavior_tags_indices_and_top5():
    text = "网络活跃指数:40,常用网站top5:SNS,网页搜索,网页搜索top5:baidu.com"
    parsed = parse_behavior_tags(text)
    assert parsed["indices"]["网络活跃指数"] == 40.0
    assert parsed["top5"]["常用网站top5"] == ["SNS", "网页搜索"]
    assert parsed["top5"]["网页搜索top5"] == ["baidu.com"]


def test_parse_behavior_tags_empty():
    assert parse_behavior_tags(None) == {"indices": {}, "top5": {}}
    assert parse_behavior_tags("") == {"indices": {}, "top5": {}}


def test_match_brand_priority():
    # 同时提到苹果与华为时按 BRAND_PATTERNS 优先级返回（苹果在前）
    assert match_brand("苹果（Apple）iPhone6 与华为 Mate8 对比") is not None
    assert match_brand("【Surpass】Surpass16GB粉色") is None
    assert match_brand("华为荣耀8") == "huawei"


def test_match_keyword_groups():
    groups = {"search": ["beidu", "baidu"], "video": ["youku", "iqiyi"]}
    assert "video" in match_keyword_groups("www.youku.com", groups)
    assert match_keyword_groups("", groups) == []


def test_add_time_features():
    frame = pd.DataFrame(
        {"time": ["2015-11-02 21:30:00", "2015-11-07 10:00:00", None]}
    )
    out = add_time_features(frame, "time", "track")
    assert out.loc[0, "track_is_night"] == 1.0
    assert out.loc[1, "track_is_weekend"] == 1.0
    assert out.loc[0, "track_hour"] == 21
    assert "time" in out.columns
