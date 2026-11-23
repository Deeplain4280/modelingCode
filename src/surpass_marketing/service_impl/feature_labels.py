# -*- coding: utf-8 -*-
"""特征列的中文标签：论文表格、图表与营销建议共用一套口径。"""
from __future__ import annotations

from typing import Dict

from ..domain import constants as C

FEATURE_LABELS: Dict[str, str] = {
    # 人口学
    "age": "年龄",
    "gender": "性别",
    "edu": "学历",
    "occ_0": "职业-生产/运营/采购/物流",
    # 行为指数
    "idx_active": "网络活跃指数",
    "idx_shopping": "网络购物指数",
    "idx_video": "在线视频指数",
    "idx_travel": "出行指数",
    "idx_study": "学习指数",
    "idx_finance": "理财指数",
    "idx_baby": "母婴指数",
    "idx_count": "行为指数覆盖数",
    # 行为量级
    "vol_ecom_rate": "电商日均浏览事件数",
    "vol_ecom_models_rate": "日均浏览手机型号数",
    "vol_ecom_l2_rate": "日均浏览二级类目数",
    "vol_ecom_night_share": "电商夜间浏览占比",
    "vol_ecom_weekend_share": "电商周末浏览占比",
    "vol_ecom_hour_mean": "电商平均浏览时段",
    "vol_video_n": "视频浏览事件数",
    "vol_video_days": "视频活跃天数",
    "vol_video_unique": "视频内容数",
    "vol_video_night_share": "视频夜间浏览占比",
    "vol_video_weekend_share": "视频周末浏览占比",
    "vol_video_hour_mean": "视频平均浏览时段",
    "vol_media_n": "触媒浏览事件数",
    "vol_media_urls": "触媒网站数",
    "vol_media_sites": "触媒渠道数",
    "vol_media_search_share": "搜索行为占比",
    "vol_total": "总行为事件数",
    "vol_source_count": "数据来源覆盖数",
    # 覆盖标记
    "has_identity": "有身份标签",
    "has_tags": "有行为标签",
    "has_ecom": "有电商行为",
    "has_video": "有视频行为",
    "has_media": "有触媒行为",
    # 品类与 Surpass 关注
    "pref_phone_share": "手机品类浏览占比",
    "pref_accessory_share": "手机配件浏览占比",
    "pref_digital_share": "数码品类浏览占比",
    "pref_surpass_share": "Surpass 提及占比",
    "pref_surpass_count": "Surpass 提及次数(log1p)",
    "pref_surpass_rate": "日均 Surpass 提及次数(log1p)",
    # 视频一级分类
    "vid_l1_entertainment": "视频娱乐内容(剧/影/综艺)",
    "vid_l1_tv": "电视剧浏览占比",
    "vid_l1_movie": "电影浏览占比",
    "vid_l1_variety": "综艺浏览占比",
    "vid_l1_unknown": "未分类视频占比",
}

# 品牌
for _code, _name in C.BRAND_LABELS.items():
    FEATURE_LABELS[f"brand_{_code}_share"] = f"{_name}品牌浏览占比"
FEATURE_LABELS["brand_premium_share"] = "高端品牌浏览占比"
FEATURE_LABELS["brand_mass_share"] = "大众品牌浏览占比"

# 购物倾向 / 视频题材 / 网站类别 / 促销触点 / 域名
_SHOP_LABELS = {
    "mobile_digital": "手机数码",
    "apparel": "服饰鞋包",
    "beauty": "美妆个护",
    "mother_baby": "母婴",
    "food": "食品生鲜",
    "sports": "运动户外",
    "home": "家居家装",
    "books": "图书音像",
    "auto": "汽车用品",
}
_GENRE_LABELS = {
    "suspense": "悬疑",
    "comedy": "喜剧",
    "romance": "言情",
    "action": "动作",
    "costume": "古装",
    "war": "战争",
    "scifi": "科幻",
    "horror": "恐怖",
    "reality": "真人秀",
    "animation": "动漫",
    "other": "其它题材",
}
_SITE_LABELS = {
    "sns": "社交",
    "search": "搜索",
    "news": "新闻",
    "shopping": "购物",
    "video": "视频",
    "game": "游戏",
    "music": "音乐",
    "portal": "门户",
    "navigation": "导航",
    "finance": "财经",
    "auto": "汽车",
    "health": "健康",
    "literature": "文学",
}
_PROMO_LABELS = {
    "groupbuy": "团购",
    "ad_network": "广告联盟",
    "epay": "电子支付",
    "finance_portal": "银行门户",
    "finance_news": "财经资讯",
    "auto": "汽车网站",
    "job": "人才招聘",
    "fashion": "女性时尚",
    "military": "政史军事",
    "health": "医疗健康",
    "luxury": "奢侈品",
    "dating": "婚恋网站",
    "p2p": "P2P金融",
    "lottery": "彩票网站",
}
_DOMAIN_LABELS = {
    "baidu": "百度",
    "qq": "腾讯",
    "sina": "新浪",
    "sohu": "搜狐",
    "netease": "网易",
    "jd": "京东",
    "tmall": "天猫",
    "taobao": "淘宝",
    "suning": "苏宁",
    "youku": "优酷",
    "iqiyi": "爱奇艺",
    "letv_video": "乐视视频",
    "bilibili": "哔哩哔哩",
    "weibo": "微博",
    "zhihu": "知乎",
    "douban": "豆瓣",
}
_CHANNEL_LABELS = {code: site for site, code in C.MEDIA_SITE_CHANNELS.items()}


def feature_label(name: str) -> str:
    """返回特征的中文标签，未知特征回退为原列名。"""
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]
    if name.startswith("occ_"):
        try:
            index = int(name.split("_", 1)[1])
        except ValueError:
            return name
        if 0 <= index < len(C.OCCUPATIONS):
            return f"职业-{C.OCCUPATIONS[index]}"
    if name.startswith("tag_shop_"):
        return f"购物倾向-{_SHOP_LABELS.get(name[len('tag_shop_'):], name)}"
    if name.startswith("tag_genre_"):
        return f"视频题材-{_GENRE_LABELS.get(name[len('tag_genre_'):], name)}"
    if name.startswith("tag_site_"):
        return f"常用网站-{_SITE_LABELS.get(name[len('tag_site_'):], name)}"
    if name.startswith("tag_promo_"):
        return f"营销触点-{_PROMO_LABELS.get(name[len('tag_promo_'):], name)}"
    if name.startswith("tag_domain_"):
        return f"常用域名-{_DOMAIN_LABELS.get(name[len('tag_domain_'):], name)}"
    if name.startswith("genre_"):
        return f"视频题材-{_GENRE_LABELS.get(name[len('genre_'):], name)}"
    if name.startswith("media_ch_"):
        return f"触媒渠道-{_CHANNEL_LABELS.get(name[len('media_ch_'):], name)}"
    if name.startswith("media_domain_"):
        return f"触媒域名-{_DOMAIN_LABELS.get(name[len('media_domain_'):], name)}"
    return name


__all__ = ["FEATURE_LABELS", "feature_label"]
