# -*- coding: utf-8 -*-
"""领域常量：表名、列名、映射表、品牌与关键词规则。

所有“魔法字符串”集中在此，便于中英文对照与论文写作时统一术语。
"""
from __future__ import annotations

# --------------------------------------------------------------------------- #
# 工作表名称（附件1）
# --------------------------------------------------------------------------- #
SHEET_TARGET_USERS = "目标用户"
SHEET_ID_MAPPING = "IDmapping"
SHEET_IDENTITY = "目标用户身份标签"
SHEET_BEHAVIOR_TAGS = "目标用户行为标签"
SHEET_ECOMMERCE = "电商行为"
SHEET_VIDEO = "视频行为"
SHEET_MEDIA = "触媒行为"
SHEET_SURPASS_PARAMS = "Surpass参数"

ALL_SHEETS = [
    SHEET_TARGET_USERS,
    SHEET_ID_MAPPING,
    SHEET_IDENTITY,
    SHEET_BEHAVIOR_TAGS,
    SHEET_ECOMMERCE,
    SHEET_VIDEO,
    SHEET_MEDIA,
    SHEET_SURPASS_PARAMS,
]

# --------------------------------------------------------------------------- #
# 通用列名
# --------------------------------------------------------------------------- #
COL_USER_ID = "用户编号"
COL_SID = "SID"
COL_STATE = "最后一次跟踪的状态"
PURCHASE_STATE = "购买"
COL_TRACK_TIME = "在某购物平台上的平均每次停留的时间    分:秒"
COL_ECOM_TIME = "浏览时长"
COL_ECOM_MODEL = "在商务平台网页上浏览过的手机型号"
COL_ECOM_L1 = "产品一级分类"
COL_ECOM_L2 = "产品二级分类"
COL_VIDEO_TIME = "浏览时长"
COL_VIDEO_CONTENT = "浏览内容"
COL_VIDEO_L1 = "内容一级分类"
COL_VIDEO_L2 = "内容二级分类"
COL_MEDIA_SITE = "浏览网页名称"
COL_MEDIA_SEARCH = "搜索子类名称"
COL_MEDIA_URL = "网址"
COL_BEHAVIOR_TEXT = "目标用户行为标签"

COL_AGE = "年龄"
COL_GENDER = "性别"
COL_EDUCATION = "学历"
COL_OCCUPATION = "正在学习或从事的专业或职业"

# 附件2
COL_A2_TIME = "在某购物平台上的平均每次停留的时间    分:秒"
COL_A2_DECISION = "你认为客户会购买吗？填“购买”或“不购买”"

# --------------------------------------------------------------------------- #
# 有序类别编码
# --------------------------------------------------------------------------- #
AGE_ORDER = {"18-24": 1, "25-34": 2, "35-44": 3, "45-54": 4}
EDUCATION_ORDER = {
    "高中及以下": 1,
    "大学专科": 2,
    "大学本科": 3,
    "硕士及以上": 4,
    "其它": 0,
}
GENDER_MAP = {"女": 1, "男": 0}

OCCUPATIONS = [
    "生产_运营_采购_物流",
    "人事_行政_高级管理",
    "计算机_互联网_通信_电子",
    "服务业",
    "销售_客服_技术支持",
    "公务员_翻译_其他",
    "广告_市场_媒体_艺术",
    "会计_金融_银行_保险",
    "咨询_法律_教育_科研",
    "建筑_房地产",
    "生物_制药_医疗_护理",
]

# 行为标签中的指数名称
BEHAVIOR_INDICES = [
    "网络活跃指数",
    "网络购物指数",
    "在线视频指数",
    "出行指数",
    "学习指数",
    "理财指数",
    "母婴指数",
]

# 购物倾向中的“手机/数码”类关键词
SHOPPING_MOBILE_KEYWORDS = ["手机", "数码", "电脑", "家电"]

# --------------------------------------------------------------------------- #
# 手机品牌关键词（按优先级匹配，一个型号文本只归入首个命中的品牌）
# --------------------------------------------------------------------------- #
BRAND_PATTERNS = {
    "apple": ["iphone", "苹果", "apple"],
    "samsung": ["三星", "samsung"],
    "huawei": ["华为", "huawei"],
    "honor": ["荣耀", "honor"],
    "xiaomi": ["小米", "红米", "xiaomi", "redmi"],
    "meizu": ["魅族", "meizu"],
    "letv": ["乐视", "letv", "乐1", "乐2", "乐视手机"],
    "lenovo": ["联想", "lenovo", "zuk"],
    "zte": ["中兴", "zte"],
    "coolpad": ["酷派", "coolpad"],
    "nubia": ["努比亚", "nubia"],
    "smartisan": ["锤子", "smartisan"],
    "oppo": ["oppo"],
    "vivo": ["vivo"],
    "oneplus": ["一加", "oneplus"],
}

PREMIUM_BRANDS = ["apple", "samsung", "huawei"]
MASS_BRANDS = ["honor", "xiaomi", "meizu", "letv", "lenovo", "zte", "coolpad", "nubia", "smartisan", "oppo", "vivo", "oneplus"]

# 视频一级分类中的“娱乐内容”
VIDEO_ENTERTAINMENT_L1 = ["电视剧", "电影", "综艺", "动漫"]

# 触媒行为：网页名称关键词 -> 渠道类别
MEDIA_CHANNEL_KEYWORDS = {
    "search": ["搜索"],
    "sns": ["社区", "社交", "论坛", "微博", "SNS"],
    "news": ["新闻", "媒体", "资讯"],
    "shopping": ["购物", "电商", "商城", "团购"],
    "video": ["视频", "影视"],
    "game": ["游戏"],
    "finance": ["财经", "金融", "理财", "银行", "股票"],
    "auto": ["汽车", "车"],
    "life": ["生活", "服务", "旅游", "房产", "招聘"],
    "health": ["医疗", "健康"],
    "education": ["教育", "学习", "培训"],
    "mom": ["母婴", "亲子", "育儿"],
}

# 高频域名关键词（触媒/电商/视频标签中的域名偏好）
DOMAIN_KEYWORDS = {
    "baidu": ["baidu.com"],
    "qq": ["qq.com"],
    "sina": ["sina.com", "weibo.com"],
    "sohu": ["sohu.com"],
    "netease": ["163.com"],
    "jd": ["jd.com"],
    "tmall": ["tmall.com"],
    "taobao": ["taobao.com"],
    "suning": ["suning.com"],
    "youku": ["youku.com"],
    "iqiyi": ["iqiyi.com"],
    "letv_video": ["letv.com"],
    "bilibili": ["bilibili.com"],
    "weibo": ["weibo.com"],
    "zhihu": ["zhihu.com"],
    "douban": ["douban.com"],
}

# 偏好指标使用的“购物倾向”类别关键词
SHOPPING_TENDENCY_GROUPS = {
    "mobile_digital": ["手机", "数码", "电脑", "家电"],
    "apparel": ["男装", "女装", "服饰", "鞋", "箱包"],
    "beauty": ["美妆", "护肤", "化妆", "个护"],
    "mother_baby": ["母婴", "奶粉", "尿裤"],
    "food": ["食品", "生鲜", "零食", "酒"],
    "sports": ["运动", "户外"],
    "home": ["家居", "家装", "家具"],
    "books": ["图书", "书籍", "音像"],
    "auto": ["汽车", "车品"],
}

# 偏好指标使用的“在线视频题材”类别关键词
VIDEO_GENRE_GROUPS = {
    "suspense": ["悬疑", "推理", "犯罪"],
    "comedy": ["喜剧", "搞笑"],
    "romance": ["爱情", "言情", "偶像"],
    "action": ["动作", "武打", "武侠"],
    "costume": ["古装", "宫廷", "历史"],
    "war": ["战争", "军事"],
    "scifi": ["科幻", "奇幻"],
    "horror": ["恐怖", "惊悚"],
    "reality": ["真人秀", "综艺"],
    "animation": ["动漫", "动画"],
}

# 偏好指标使用的“常用网站/媒体”类别关键词
WEBSITE_CATEGORY_GROUPS = {
    "sns": ["SNS", "社交", "社区", "论坛"],
    "search": ["网页搜索", "搜索"],
    "news": ["新闻", "媒体", "资讯"],
    "shopping": ["B2C商城", "C2C平台", "购物", "电商"],
    "video": ["综合视频", "视频"],
    "game": ["游戏"],
    "music": ["在线音乐", "音乐"],
    "portal": ["综合门户"],
    "navigation": ["网址导航"],
    "finance": ["财经", "股票", "理财", "银行"],
    "auto": ["汽车"],
    "health": ["医疗", "健康"],
    "literature": ["文学", "电子书籍", "小说"],
}

# 品牌英文/中文对照（论文与图表使用）
BRAND_LABELS = {
    "apple": "苹果",
    "samsung": "三星",
    "huawei": "华为",
    "honor": "荣耀",
    "xiaomi": "小米",
    "meizu": "魅族",
    "letv": "乐视",
    "lenovo": "联想",
    "zte": "中兴",
    "coolpad": "酷派",
    "nubia": "努比亚",
    "smartisan": "锤子",
    "oppo": "OPPO",
    "vivo": "vivo",
    "oneplus": "一加",
}

# 营销建议：分层策略（渠道/频次/创意/核心指标）
MARKETING_PLAYBOOK = {
    "高潜力": {
        "触达渠道": "短信 + APP 推送 + 社交信息流",
        "建议频次": "每周 2~3 次，连续 2 周",
        "创意方向": "Surpass 卖点（5.5 英寸大屏、1300 万像素、3100mAh）直击换机需求",
        "核心指标": "CVR、CPA、ROAS、退订率",
        "预算占比": 0.50,
    },
    "中潜力": {
        "触达渠道": "信息流广告 + 电商站内推荐",
        "建议频次": "每周 1~2 次，持续 3 周",
        "创意方向": "以旧换新/分期免息，突出性价比与同价位对比",
        "核心指标": "CVR、CPA、ROAS、退订率",
        "预算占比": 0.35,
    },
    "低潜力": {
        "触达渠道": "低成本内容种草 + 私域触达",
        "建议频次": "每月 1 次",
        "创意方向": "品牌认知与口碑内容，培育长期兴趣",
        "核心指标": "曝光成本、互动率、退订率",
        "预算占比": 0.15,
    },
}

# --------------------------------------------------------------------------- #
# 行为标签中的指数英文别名（用于特征列名，避免中文列名进入模型矩阵）
# --------------------------------------------------------------------------- #
INDEX_ALIASES = {
    "网络活跃指数": "active",
    "网络购物指数": "shopping",
    "在线视频指数": "video",
    "出行指数": "travel",
    "学习指数": "study",
    "理财指数": "finance",
    "母婴指数": "baby",
}

# 行为标签中用于“营销触点偏好”的 top5 键（键名以原始数据中出现过的为准）
PROMO_TAG_KEYS = {
    "groupbuy": "团购top5",
    "ad_network": "广告联盟top5",
    "epay": "电子支付top5",
    "finance_portal": "银行门户top5",
    "finance_news": "财经资讯top5",
    "auto": "汽车网站top5",
    "job": "人才招聘top5",
    "fashion": "女性时尚top5",
    "military": "政史军事top5",
    "health": "医疗健康top5",
    "luxury": "奢侈品top5",
    "dating": "婚恋网站top5",
    "p2p": "p2p金融top5",
    "lottery": "彩票网站top5",
}

# 触媒行为：网页名称精确映射到渠道（原始表共 21 个取值）
MEDIA_SITE_CHANNELS = {
    "新闻媒体": "news",
    "在线视频": "video",
    "电子商务": "shopping",
    "搜索服务": "search",
    "社交网络和在线社区": "sns",
    "网址导航": "navigation",
    "网络服务应用": "other",
    "IT数码": "it",
    "游戏": "game",
    "投资金融": "finance",
    "生活服务": "life",
    "汽车": "auto",
    "音乐": "music",
    "房产家居": "home",
    "交通旅游": "travel",
    "休闲娱乐": "entertainment",
    "人才招聘": "job",
    "医疗保健": "health",
    "女性时尚": "fashion",
    "教学及考试": "education",
    "垂直行业": "vertical",
}

# --------------------------------------------------------------------------- #
# 问题3：8 个可解释偏好指标的构成（均为标准化后的成分均值）
# --------------------------------------------------------------------------- #
PREFERENCE_INDICATORS = {
    "phone_digital": [
        "pref_phone_share",
        "pref_surpass_share",
        "tag_shop_mobile_digital",
    ],
    "premium": [
        "brand_apple_share",
        "brand_samsung_share",
        "brand_huawei_share",
        "brand_premium_share",
    ],
    "value": [
        "brand_honor_share",
        "brand_xiaomi_share",
        "brand_meizu_share",
        "brand_letv_share",
        "brand_mass_share",
    ],
    "entertainment": [
        "media_ch_video",
        "tag_genre_reality",
        "tag_genre_comedy",
        "tag_genre_romance",
        "tag_site_video",
    ],
    "info_serious": [
        "media_ch_news",
        "media_ch_search",
        "tag_site_news",
        "tag_site_portal",
        "tag_domain_baidu",
    ],
    "social": [
        "media_ch_sns",
        "tag_site_sns",
        "tag_domain_weibo",
        "tag_domain_qq",
        "tag_domain_sina",
    ],
    "promotion": [
        "tag_promo_groupbuy",
        "tag_promo_ad_network",
        "tag_promo_epay",
        "media_ch_shopping",
    ],
    "lifestyle": [
        "media_ch_life",
        "media_ch_health",
        "media_ch_auto",
        "media_ch_finance",
        "tag_promo_auto",
        "tag_promo_finance_portal",
        "tag_promo_fashion",
        "tag_promo_luxury",
    ],
}

PREFERENCE_INDICATOR_LABELS = {
    "phone_digital": "手机数码关注",
    "premium": "高端品牌偏好",
    "value": "性价比偏好",
    "entertainment": "娱乐内容偏好",
    "info_serious": "资讯求知偏好",
    "social": "社交互动偏好",
    "promotion": "促销敏感",
    "lifestyle": "品质生活偏好",
}

__all__ = [name for name in dir() if not name.startswith("_")]
