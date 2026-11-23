# -*- coding: utf-8 -*-
"""数据服务实现：数据审计 + 用户级特征工程。

设计要点
--------
1. 原始 xlsx 只读；所有中间结果写入 ``outputs/interim``（parquet）。
2. 每张行为表先按用户聚合为“用户 × 特征”，再与主表左连接，保证一行一个用户。
3. 特征命名遵循前缀约定（见 ``service_impl/feature_columns.py``）：
   ``vol_*`` 活跃量级、``idx_*`` 行为指数、``brand_*/pref_*`` 电商偏好、
   ``vid_*/genre_*`` 视频偏好、``media_*`` 触媒渠道、``tag_*`` 行为标签偏好、
   ``has_*`` 数据覆盖标记。跟踪状态表派生的 ``track_*`` 一律不作为特征。
4. **时间口径（防标签泄漏）**：购买者的电商/视频行为只取“首次购买之前”的记录；
   未购买者取统一观察窗口截止（2015-11-30 23:59:59）；附件2 只取其待判时间之前的记录。
   购买标签取“任一记录出现购买”（与论文口径一致）。
5. 缺失策略：计数/份额类缺失填 0（无行为即 0），由 ``has_*`` 标记区分“未观测”；
   人口学与行为指数保留 NaN，交由模型管线中的中位数填补，避免把缺失编码成真实类别。
"""
from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..config import AppConfig
from ..domain import constants as C
from ..domain.entities import DataAudit
from ..repository.artifact_repository import ArtifactRepository
from ..repository.excel_repository import ExcelRepository
from ..service import DataService
from ..utils.text_parse import match_keyword_groups, parse_behavior_tags

# 观察窗口截止时间：目标用户表的记录时间范围为 2015-11-01 ~ 2015-11-30
WINDOW_END = pd.Timestamp("2015-11-30 23:59:59")
USER_FEATURE_CACHE = "user_features.parquet"
FEATURE_TABLE_SUFFIX = ".parquet"
FEATURE_CACHE_VERSION = "4"  # 口径变更（去 track_*、去视频特征、电商强度归一化）后递增
CACHE_VERSION_FILE = "feature_cache_version.txt"

# 各来源特征表名（缓存于 outputs/interim）
FEAT_IDENTITY = "feat_identity"
FEAT_TAGS = "feat_tags"
FEAT_ECOM = "feat_ecom"
FEAT_VIDEO = "feat_video"
FEAT_MEDIA = "feat_media"


class DataServiceImpl(DataService):
    """数据服务实现（含特征工程与缓存）。"""

    def __init__(
        self,
        config: AppConfig,
        excel_repository: ExcelRepository,
        artifact_repository: ArtifactRepository,
        logger: Optional[logging.Logger] = None,
    ):
        self.config = config
        self.excel = excel_repository
        self.artifacts = artifact_repository
        self.logger = logger or logging.getLogger(__name__)
        self._cache: Dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------ #
    # 基础工具
    # ------------------------------------------------------------------ #
    def _sheet(self, name: str, force: bool = False, cache: bool = True) -> pd.DataFrame:
        if cache and not force and name in self._cache:
            return self._cache[name]
        frame = self.excel.load_sheet(name, force=force)
        if cache:
            self._cache[name] = frame
        return frame

    @staticmethod
    def _numeric_user_ids(frame: pd.DataFrame) -> pd.DataFrame:
        """把用户编号统一为 int64，并丢弃无法解析的行（不修改原表）。"""
        ids = pd.to_numeric(frame[C.COL_USER_ID], errors="coerce")
        mask = ids.notna()
        out = frame.loc[mask].copy()
        out[C.COL_USER_ID] = ids.loc[mask].astype("int64")
        return out

    @staticmethod
    def _assign_brand(model_text: pd.Series) -> pd.Series:
        """按优先级把型号文本归入单一品牌；同一行只保留优先级最高的品牌。"""
        lower = model_text.fillna("").astype(str).str.lower()
        brand = pd.Series(pd.NA, index=model_text.index, dtype="object")
        for name, keywords in C.BRAND_PATTERNS.items():
            pattern = "|".join(re.escape(keyword.lower()) for keyword in keywords)
            hit = lower.str.contains(pattern, regex=True, na=False)
            brand = brand.where(~(brand.isna() & hit), name)
        return brand

    @staticmethod
    def _share_by_user(work: pd.DataFrame, flag_column: str, user_column: str = C.COL_USER_ID) -> pd.Series:
        return work.groupby(user_column)[flag_column].mean()

    def _feature_table_path(self, name: str) -> Path:
        return self.config.paths.interim_dir / f"{name}{FEATURE_TABLE_SUFFIX}"

    # ------------------------------------------------------------------ #
    # 主表（标签口径 + 时间截断口径）
    # ------------------------------------------------------------------ #
    def _build_master(self) -> pd.DataFrame:
        """主表：目标用户去重 + 购买标签 + 购买时间/观察截止时间（元信息，不进模型）。"""
        raw = self._sheet(C.SHEET_TARGET_USERS)
        raw = raw.drop(columns=[column for column in raw.columns if str(column).startswith("Unnamed")], errors="ignore")
        raw = self._numeric_user_ids(raw)
        raw["_purchase"] = (raw[C.COL_STATE].astype(str).str.strip() == C.PURCHASE_STATE).astype("int8")
        raw["_time"] = pd.to_datetime(raw[C.COL_TRACK_TIME], errors="coerce")

        grouped = raw.groupby(C.COL_USER_ID)
        master = pd.DataFrame(
            {
                "target": grouped["_purchase"].max(),
                "track_time": grouped["_time"].max(),
                "state_set": grouped[C.COL_STATE].apply(
                    lambda values: "|".join(sorted({str(value).strip() for value in values.dropna()}))
                ),
            }
        ).reset_index()
        master["target"] = master["target"].astype("int8")

        # 购买时间：该用户首次出现“购买”记录的时间；仅用于截断其行为数据
        first_purchase = raw.loc[raw["_purchase"] == 1].groupby(C.COL_USER_ID)["_time"].min()
        master["purchase_time"] = master[C.COL_USER_ID].map(first_purchase)
        master["cutoff_time"] = master["purchase_time"].where(
            (master["target"] == 1) & master["purchase_time"].notna(), WINDOW_END
        )

        # 诊断字段：购买后仍出现浏览/搜索记录的用户（说明“末状态购买”口径会漏标签）
        later = raw[[C.COL_USER_ID, "_time", C.COL_STATE]].merge(
            master[[C.COL_USER_ID, "purchase_time"]], on=C.COL_USER_ID, how="left"
        )
        post = later.loc[
            later["purchase_time"].notna()
            & (later["_time"] > later["purchase_time"])
            & later[C.COL_STATE].astype(str).str.strip().isin(["浏览", "搜索"])
        ]
        master["post_purchase_activity"] = master[C.COL_USER_ID].isin(set(post[C.COL_USER_ID])).astype("int8")

        self.logger.info(
            "主表构建完成：%d 位目标用户，购买者 %d 人（阳性率 %.2f%%），购买后仍有浏览/搜索记录 %d 人",
            len(master),
            int(master["target"].sum()),
            100 * master["target"].mean(),
            int(master["post_purchase_activity"].sum()),
        )
        return master

    @staticmethod
    def _cutoffs(master: pd.DataFrame) -> pd.Series:
        """用户编号 → 行为截断时间（购买者=首次购买时间，未购者=统一窗口截止）。"""
        return pd.Series(
            pd.to_datetime(master["cutoff_time"]).to_numpy(),
            index=master[C.COL_USER_ID].to_numpy(),
        )

    # ------------------------------------------------------------------ #
    # 各来源用户级特征
    # ------------------------------------------------------------------ #
    def _build_identity_features(self) -> pd.DataFrame:
        raw = self._sheet(C.SHEET_IDENTITY)
        raw = self._numeric_user_ids(raw)
        raw = raw.drop_duplicates()
        keep = {C.COL_USER_ID: "first", C.COL_AGE: "first", C.COL_GENDER: "first",
                C.COL_EDUCATION: "first", C.COL_OCCUPATION: "first"}
        frame = raw.groupby(C.COL_USER_ID, as_index=False).agg(keep)
        frame["age"] = frame[C.COL_AGE].map(C.AGE_ORDER).astype("float")
        frame["gender"] = frame[C.COL_GENDER].map(C.GENDER_MAP).astype("float")
        frame["edu"] = frame[C.COL_EDUCATION].map(C.EDUCATION_ORDER).astype("float")
        for index, occupation in enumerate(C.OCCUPATIONS):
            frame[f"occ_{index}"] = (frame[C.COL_OCCUPATION] == occupation).astype("float")
        frame["has_identity"] = 1.0
        frame = frame[[C.COL_USER_ID, "age", "gender", "edu", "has_identity"]
                      + [f"occ_{i}" for i in range(len(C.OCCUPATIONS))]]
        frame.to_parquet(self._feature_table_path(FEAT_IDENTITY), index=False)
        self.logger.info("身份特征完成：%d 位用户，覆盖率 %.1f%%", len(frame), 100 * len(frame) / 16295)
        return frame

    def _build_tag_features(self) -> pd.DataFrame:
        raw = self._sheet(C.SHEET_BEHAVIOR_TAGS)
        raw = self._numeric_user_ids(raw)
        raw = raw.dropna(subset=[C.COL_BEHAVIOR_TEXT]).drop_duplicates(subset=[C.COL_USER_ID])

        records: List[Dict[str, object]] = []
        for user_id, text in zip(raw[C.COL_USER_ID].to_numpy(), raw[C.COL_BEHAVIOR_TEXT].to_numpy()):
            parsed = parse_behavior_tags(text)
            indices: Dict[str, float] = parsed["indices"]  # type: ignore[assignment]
            top5: Dict[str, List[str]] = parsed["top5"]  # type: ignore[assignment]

            record: Dict[str, object] = {C.COL_USER_ID: int(user_id), "has_tags": 1.0}
            for zh_name, alias in C.INDEX_ALIASES.items():
                record[f"idx_{alias}"] = indices.get(zh_name, np.nan)
            record["idx_count"] = float(len(indices))

            all_values = [str(value).strip() for values in top5.values() for value in values if str(value).strip()]
            blob = ",".join(all_values)
            shopping_blob = ",".join(top5.get("购物倾向top5", []))
            video_blob = ",".join(top5.get("在线视频top5", []))

            shop_hits = set(match_keyword_groups(shopping_blob, C.SHOPPING_TENDENCY_GROUPS))
            for group in C.SHOPPING_TENDENCY_GROUPS:
                record[f"tag_shop_{group}"] = 1.0 if group in shop_hits else 0.0

            genre_hits = set(match_keyword_groups(video_blob, C.VIDEO_GENRE_GROUPS))
            for group in C.VIDEO_GENRE_GROUPS:
                record[f"tag_genre_{group}"] = 1.0 if group in genre_hits else 0.0

            site_hits = set(match_keyword_groups(blob, C.WEBSITE_CATEGORY_GROUPS))
            for group in C.WEBSITE_CATEGORY_GROUPS:
                record[f"tag_site_{group}"] = 1.0 if group in site_hits else 0.0

            domains = {value.lower() for value in all_values if "." in value and " " not in value}
            record["tag_domain_n"] = float(len(domains))
            for name, keywords in C.DOMAIN_KEYWORDS.items():
                record[f"tag_domain_{name}"] = (
                    1.0 if any(keyword in domain for domain in domains for keyword in keywords) else 0.0
                )

            for key, tag_key in C.PROMO_TAG_KEYS.items():
                record[f"tag_promo_{key}"] = 1.0 if top5.get(tag_key) else 0.0

            records.append(record)

        frame = pd.DataFrame.from_records(records)
        frame.to_parquet(self._feature_table_path(FEAT_TAGS), index=False)
        self.logger.info("行为标签特征完成：%d 位用户，%d 列", *frame.shape)
        return frame

    def _truncate_by_cutoff(self, work: pd.DataFrame, cutoffs: pd.Series) -> pd.DataFrame:
        """只保留时间不晚于用户截断时间的记录（购买者=首次购买前，未购者=窗口截止）。"""
        cut = pd.DataFrame(
            {C.COL_USER_ID: cutoffs.index.to_numpy(), "_cutoff": cutoffs.to_numpy()}
        )
        merged = work.merge(cut, on=C.COL_USER_ID, how="left")
        before = len(merged)
        merged = merged.loc[merged["_ts"].notna() & (merged["_ts"] <= merged["_cutoff"])].copy()
        self.logger.info("按用户时间截断：%d → %d 行（剔除 %d 行）", before, len(merged), before - len(merged))
        return merged

    def _ecommerce_frame(self, cutoffs: pd.Series) -> pd.DataFrame:
        raw = self._sheet(C.SHEET_ECOMMERCE, cache=False)
        raw = self._numeric_user_ids(raw)
        l1 = raw[C.COL_ECOM_L1].fillna("").astype(str)
        l2 = raw[C.COL_ECOM_L2].fillna("").astype(str)
        model = raw[C.COL_ECOM_MODEL].fillna("").astype(str)
        ts = pd.to_datetime(raw[C.COL_ECOM_TIME], errors="coerce")

        work = pd.DataFrame(
            {
                C.COL_USER_ID: raw[C.COL_USER_ID].to_numpy(),
                "_l1": l1.to_numpy(),
                "_l2": l2.to_numpy(),
                "_model": model.to_numpy(),
                "_ts": ts.to_numpy(),
            }
        )
        work["_brand"] = self._assign_brand(pd.Series(work["_model"], index=work.index)).to_numpy()
        has_model = work["_model"].str.len() > 0
        work["_has_model"] = has_model.astype("int8")
        work["_surpass"] = work["_model"].str.lower().str.contains("surpass", regex=False).astype("int8")
        l1l2 = work["_l1"] + "|" + work["_l2"]
        work["_phone_row"] = work["_l1"].str.contains("手机", regex=False).astype("int8")
        work["_accessory_row"] = l1l2.str.contains("配件|手机壳|贴膜|电池|充电", regex=True).astype("int8")
        work["_digital_row"] = l1l2.str.contains("摄影|数码|相机", regex=True).astype("int8")
        hour = work["_ts"].dt.hour
        work["_night"] = ((hour >= 20) | (hour < 2)).astype("float")
        work["_weekend"] = (work["_ts"].dt.weekday >= 5).astype("float")
        work["_hour"] = hour.astype("float")
        work["_date"] = work["_ts"].dt.normalize()
        work = self._truncate_by_cutoff(work, cutoffs)

        grouped = work.groupby(C.COL_USER_ID)
        size = grouped.size().astype("float")
        active_days = grouped["_date"].nunique().astype("float").clip(lower=1.0)
        models = grouped["_model"].apply(lambda s: s[s.str.len() > 0].nunique()).astype("float")
        l2_count = grouped["_l2"].nunique().astype("float")
        surpass_count = grouped["_surpass"].sum().astype("float")
        frame = pd.DataFrame(
            {
                # 观测窗口长度因购买时点而异，绝对频次会把“窗口长度”当成购买信号；
                # 因此强度类特征统一用“每个活跃日的平均事件数”，构成类特征用份额。
                "vol_ecom_rate": size / active_days,
                "vol_ecom_models_rate": models / active_days,
                "vol_ecom_l2_rate": l2_count / active_days,
                "vol_ecom_night_share": grouped["_night"].mean(),
                "vol_ecom_weekend_share": grouped["_weekend"].mean(),
                "vol_ecom_hour_mean": grouped["_hour"].mean(),
                "pref_phone_share": grouped["_phone_row"].mean(),
                "pref_accessory_share": grouped["_accessory_row"].mean(),
                "pref_digital_share": grouped["_digital_row"].mean(),
                "pref_surpass_share": (
                    grouped["_surpass"].sum() / grouped["_has_model"].sum().replace(0, np.nan)
                ).fillna(0.0),
                "pref_surpass_rate": np.log1p(surpass_count / active_days),
                "has_ecom": 1.0,
            }
        ).reset_index()

        known = work.loc[work["_brand"].notna(), [C.COL_USER_ID, "_brand"]]
        brand_counts = known.groupby([C.COL_USER_ID, "_brand"]).size().unstack(fill_value=0)
        brand_share = brand_counts.div(brand_counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        premium_columns = [brand for brand in C.PREMIUM_BRANDS if brand in brand_share.columns]
        mass_columns = [brand for brand in C.MASS_BRANDS if brand in brand_share.columns]
        for brand in C.BRAND_PATTERNS:
            frame[f"brand_{brand}_share"] = frame[C.COL_USER_ID].map(
                brand_share[brand] if brand in brand_share.columns else pd.Series(dtype="float")
            ).fillna(0.0)
        frame["brand_premium_share"] = frame[C.COL_USER_ID].map(
            brand_share[premium_columns].sum(axis=1) if premium_columns else pd.Series(dtype="float")
        ).fillna(0.0)
        frame["brand_mass_share"] = frame[C.COL_USER_ID].map(
            brand_share[mass_columns].sum(axis=1) if mass_columns else pd.Series(dtype="float")
        ).fillna(0.0)
        return frame

    def _build_ecommerce_features(self) -> pd.DataFrame:
        cutoffs = self._cutoffs(self._build_master())
        frame = self._ecommerce_frame(cutoffs)
        frame.to_parquet(self._feature_table_path(FEAT_ECOM), index=False)
        self.logger.info("电商行为特征完成：%d 位用户，%d 列（已按用户时间截断）", *frame.shape)
        return frame

    def _video_frame(self, cutoffs: pd.Series) -> pd.DataFrame:
        raw = self._sheet(C.SHEET_VIDEO, cache=False)
        raw = self._numeric_user_ids(raw)
        l1 = raw[C.COL_VIDEO_L1].fillna("").astype(str)
        l2 = raw[C.COL_VIDEO_L2].fillna("").astype(str)
        content = raw[C.COL_VIDEO_CONTENT].fillna("").astype(str)
        ts = pd.to_datetime(raw[C.COL_VIDEO_TIME], errors="coerce")

        work = pd.DataFrame(
            {
                C.COL_USER_ID: raw[C.COL_USER_ID].to_numpy(),
                "_l1": l1.to_numpy(),
                "_l2": l2.to_numpy(),
                "_content": content.to_numpy(),
                "_ts": ts.to_numpy(),
            }
        )
        genre_map: Dict[str, str] = {}
        for value in pd.unique(work["_l2"]):
            hits = match_keyword_groups(value, C.VIDEO_GENRE_GROUPS)
            genre_map[value] = hits[0] if hits else "other"
        work["_genre"] = work["_l2"].map(genre_map)
        work["_entertainment"] = work["_l1"].isin(C.VIDEO_ENTERTAINMENT_L1).astype("int8")
        for short, category in (("tv", "电视剧"), ("movie", "电影"), ("variety", "综艺"), ("unknown", "未分类")):
            work[f"_l1_{short}"] = (work["_l1"] == category).astype("int8")
        hour = work["_ts"].dt.hour
        work["_night"] = ((hour >= 20) | (hour < 2)).astype("float")
        work["_weekend"] = (work["_ts"].dt.weekday >= 5).astype("float")
        work["_hour"] = hour.astype("float")
        work["_date"] = work["_ts"].dt.normalize()
        work = self._truncate_by_cutoff(work, cutoffs)

        grouped = work.groupby(C.COL_USER_ID)
        frame = pd.DataFrame(
            {
                "vol_video_n": grouped.size().astype("float"),
                "vol_video_days": grouped["_date"].nunique().astype("float"),
                "vol_video_unique": grouped["_content"].apply(lambda s: s[s.str.len() > 0].nunique()).astype("float"),
                "vol_video_night_share": grouped["_night"].mean(),
                "vol_video_weekend_share": grouped["_weekend"].mean(),
                "vol_video_hour_mean": grouped["_hour"].mean(),
                "vid_l1_entertainment": grouped["_entertainment"].mean(),
                "vid_l1_tv": grouped["_l1_tv"].mean(),
                "vid_l1_movie": grouped["_l1_movie"].mean(),
                "vid_l1_variety": grouped["_l1_variety"].mean(),
                "vid_l1_unknown": grouped["_l1_unknown"].mean(),
                "has_video": 1.0,
            }
        )
        for group in list(C.VIDEO_GENRE_GROUPS) + ["other"]:
            column = f"_genre_{group}"
            work[column] = (work["_genre"] == group).astype("int8")
            frame[f"genre_{group}"] = grouped[column].mean()
        frame = frame.reset_index()
        return frame

    def _build_video_features(self) -> pd.DataFrame:
        cutoffs = self._cutoffs(self._build_master())
        frame = self._video_frame(cutoffs)
        frame.to_parquet(self._feature_table_path(FEAT_VIDEO), index=False)
        self.logger.info("视频行为特征完成：%d 位用户，%d 列（已按用户时间截断）", *frame.shape)
        return frame

    def _build_media_features(self) -> pd.DataFrame:
        raw = self._sheet(C.SHEET_MEDIA, cache=False)
        raw = self._numeric_user_ids(raw)
        site = raw[C.COL_MEDIA_SITE].fillna("").astype(str)
        search = raw[C.COL_MEDIA_SEARCH].fillna("").astype(str)
        url = raw[C.COL_MEDIA_URL].fillna("").astype(str)

        work = pd.DataFrame(
            {
                C.COL_USER_ID: raw[C.COL_USER_ID].to_numpy(),
                "_site": site.to_numpy(),
                "_search": search.to_numpy(),
                "_url": url.to_numpy(),
            }
        )
        work["_channel"] = work["_site"].map(C.MEDIA_SITE_CHANNELS).fillna("other")
        work["_has_search"] = (work["_search"].str.len() > 0).astype("float")

        grouped = work.groupby(C.COL_USER_ID)
        frame = pd.DataFrame(
            {
                "vol_media_n": grouped.size().astype("float"),
                "vol_media_urls": grouped["_url"].apply(lambda s: s[s.str.len() > 0].nunique()).astype("float"),
                "vol_media_sites": grouped["_site"].nunique().astype("float"),
                "vol_media_search_share": grouped["_has_search"].mean(),
                "has_media": 1.0,
            }
        ).reset_index()

        channel_share = pd.crosstab(work[C.COL_USER_ID], work["_channel"], normalize="index")
        for channel in channel_share.columns:
            frame[f"media_ch_{channel}"] = frame[C.COL_USER_ID].map(channel_share[channel]).fillna(0.0)

        lower_url = work["_url"].str.lower()
        domain_columns: List[str] = []
        for name, keywords in C.DOMAIN_KEYWORDS.items():
            pattern = "|".join(re.escape(keyword) for keyword in keywords)
            column = f"_dm_{name}"
            work[column] = lower_url.str.contains(pattern, regex=True, na=False).astype("int8")
            domain_columns.append(column)
        domain_share = work.groupby(C.COL_USER_ID)[domain_columns].mean()
        for name in C.DOMAIN_KEYWORDS:
            frame[f"media_domain_{name}"] = frame[C.COL_USER_ID].map(domain_share[f"_dm_{name}"]).fillna(0.0)

        frame.to_parquet(self._feature_table_path(FEAT_MEDIA), index=False)
        self.logger.info("触媒行为特征完成：%d 位用户，%d 列", *frame.shape)
        return frame

    # ------------------------------------------------------------------ #
    # 特征缓存与合并
    # ------------------------------------------------------------------ #
    def _cache_version_path(self) -> Path:
        return self.config.paths.interim_dir / CACHE_VERSION_FILE

    def _cache_is_current(self) -> bool:
        path = self._cache_version_path()
        if not path.exists():
            return False
        return path.read_text(encoding="utf-8").strip() == FEATURE_CACHE_VERSION

    def ensure_feature_tables(self, force: bool = False) -> None:
        version_current = self._cache_is_current()
        if not version_current:
            self.logger.info("特征缓存版本不一致或缺失（期望 %s），将强制重建", FEATURE_CACHE_VERSION)
        builders = {
            FEAT_IDENTITY: self._build_identity_features,
            FEAT_TAGS: self._build_tag_features,
            FEAT_ECOM: self._build_ecommerce_features,
            FEAT_VIDEO: self._build_video_features,
            FEAT_MEDIA: self._build_media_features,
        }
        for name, builder in builders.items():
            path = self._feature_table_path(name)
            if path.exists() and not force and version_current:
                continue
            self.logger.info("构建特征表：%s", name)
            builder()
        self._cache_version_path().write_text(FEATURE_CACHE_VERSION, encoding="utf-8")

    def _load_feature_table(self, name: str) -> pd.DataFrame:
        return pd.read_parquet(self._feature_table_path(name))

    @staticmethod
    def _zero_fill_columns(frame: pd.DataFrame) -> List[str]:
        prefixes = (
            "vol_",
            "brand_",
            "pref_",
            "tag_",
            "vid_",
            "genre_",
            "media_ch_",
            "media_domain_",
            "has_",
        )
        return [column for column in frame.columns if column.startswith(prefixes)]

    def _merge_feature_tables(self, base: pd.DataFrame) -> pd.DataFrame:
        frame = base.copy()
        for name in (FEAT_IDENTITY, FEAT_TAGS, FEAT_ECOM, FEAT_VIDEO, FEAT_MEDIA):
            table = self._load_feature_table(name)
            frame = frame.merge(table, on=C.COL_USER_ID, how="left")
        zero_columns = self._zero_fill_columns(frame)
        frame[zero_columns] = frame[zero_columns].fillna(0.0)
        return frame

    def _attachment2_raw(self) -> pd.DataFrame:
        raw = self.excel.load_attachment2()
        return self._numeric_user_ids(raw)

    def build_user_features(self, force: bool = False) -> pd.DataFrame:
        cache = self.config.paths.interim_dir / USER_FEATURE_CACHE
        if cache.exists() and not force and self._cache_is_current():
            return pd.read_parquet(cache)

        self.ensure_feature_tables(force=force)
        master = self._build_master()
        frame = self._merge_feature_tables(master)
        attachment2_ids = set(self._attachment2_raw()[C.COL_USER_ID].tolist())
        frame["is_attachment2"] = frame[C.COL_USER_ID].isin(attachment2_ids).astype("int8")
        frame.to_parquet(cache, index=False)
        self.logger.info("用户特征矩阵完成：%d 位用户 × %d 列", *frame.shape)
        return frame

    def build_attachment2_features(self) -> pd.DataFrame:
        """附件2 特征：与训练同口径（无 track_* 特征），行为按待判时间截断。"""
        self.ensure_feature_tables()
        raw = self._attachment2_raw()
        base = raw[[C.COL_USER_ID]].copy()
        base["a2_cutoff_time"] = pd.to_datetime(raw[C.COL_A2_TIME], errors="coerce")
        frame = base.copy()
        # 静态画像：身份、行为标签、触媒（触媒表无时间戳，只能作静态渠道画像）
        for name in (FEAT_IDENTITY, FEAT_TAGS, FEAT_MEDIA):
            frame = frame.merge(self._load_feature_table(name), on=C.COL_USER_ID, how="left")
        # 行为表：只取附件2 待判时间之前的记录，避免使用待判时点之后的信息
        cutoffs = pd.Series(
            base["a2_cutoff_time"].fillna(WINDOW_END).to_numpy(),
            index=base[C.COL_USER_ID].to_numpy(),
        )
        for table in (self._ecommerce_frame(cutoffs),):
            frame = frame.merge(table, on=C.COL_USER_ID, how="left")
        zero_columns = self._zero_fill_columns(frame)
        frame[zero_columns] = frame[zero_columns].fillna(0.0)
        frame["is_attachment2"] = 1

        master = self._load_user_feature_labels()
        frame = frame.merge(master, on=C.COL_USER_ID, how="left")
        frame["in_target_table"] = frame["known_target"].notna().astype("int8")
        frame["known_target"] = frame["known_target"].fillna(-1).astype("int8")
        self.logger.info("附件2 特征矩阵完成：%d 位用户 × %d 列（行为按待判时间截断）", *frame.shape)
        return frame

    def _load_user_feature_labels(self) -> pd.DataFrame:
        """从主表读取已知标签（仅供附件2 的“已知标签”诊断列，不参与预测）。"""
        master = self._build_master()
        return master[[C.COL_USER_ID, "target"]].rename(columns={"target": "known_target"})

    # ------------------------------------------------------------------ #
    # 数据审计
    # ------------------------------------------------------------------ #
    def audit(self) -> DataAudit:
        target_users = self._sheet(C.SHEET_TARGET_USERS)
        target_ids = set(self._numeric_user_ids(target_users)[C.COL_USER_ID].tolist())
        target_unique = len(target_ids)

        sheet_rows: Dict[str, int] = {}
        sheet_unique: Dict[str, int] = {}
        duplicate_rows: Dict[str, int] = {}
        coverage: Dict[str, float] = {}
        time_ranges: Dict[str, str] = {}

        time_columns = {
            C.SHEET_TARGET_USERS: C.COL_TRACK_TIME,
            C.SHEET_ECOMMERCE: C.COL_ECOM_TIME,
            C.SHEET_VIDEO: C.COL_VIDEO_TIME,
        }

        for sheet in C.ALL_SHEETS:
            frame = self._sheet(sheet, cache=False)
            sheet_rows[sheet] = int(len(frame))
            if C.COL_USER_ID in frame.columns:
                ids = pd.to_numeric(frame[C.COL_USER_ID], errors="coerce").dropna().astype("int64")
                sheet_unique[sheet] = int(ids.nunique())
                duplicate_rows[sheet] = int(len(ids) - ids.nunique())
                coverage[sheet] = float(ids.nunique() / target_unique) if target_unique else float("nan")
            else:
                sheet_unique[sheet] = 0
                duplicate_rows[sheet] = 0
                coverage[sheet] = 0.0
            if sheet in time_columns and time_columns[sheet] in frame.columns:
                ts = pd.to_datetime(frame[time_columns[sheet]], errors="coerce").dropna()
                if len(ts):
                    time_ranges[sheet] = f"{ts.min():%Y-%m-%d %H:%M} ~ {ts.max():%Y-%m-%d %H:%M}"

        purchase_state = self._sheet(C.SHEET_TARGET_USERS)
        state_counts = purchase_state[C.COL_STATE].astype(str).str.strip().value_counts().to_dict()
        target_raw = self._numeric_user_ids(purchase_state)
        target_raw["_state"] = target_raw[C.COL_STATE].astype(str).str.strip()
        target_raw["_time"] = pd.to_datetime(target_raw[C.COL_TRACK_TIME], errors="coerce")
        target_raw["_purchase"] = (target_raw["_state"] == C.PURCHASE_STATE).astype("int8")
        positive = target_raw.groupby(C.COL_USER_ID)["_purchase"].max()
        positive_rate = float(positive.mean()) if len(positive) else 0.0

        # 标签口径对照：末次记录状态 vs 任一记录购买（论文口径说明的依据）
        ordered = target_raw.sort_values([C.COL_USER_ID, "_time"])
        last_state = ordered.groupby(C.COL_USER_ID)["_state"].last()
        last_purchase_users = int((last_state == C.PURCHASE_STATE).sum())
        multi_state_users = int(target_raw.groupby(C.COL_USER_ID)["_state"].nunique().gt(1).sum())
        first_purchase = (
            target_raw.loc[target_raw["_purchase"] == 1].groupby(C.COL_USER_ID)["_time"].min()
        )
        post = target_raw.merge(first_purchase.rename("_first_purchase"), on=C.COL_USER_ID, how="left")
        post_purchase_users = int(
            post.loc[
                post["_first_purchase"].notna()
                & (post["_time"] > post["_first_purchase"])
                & post["_state"].isin(["浏览", "搜索"]),
                C.COL_USER_ID,
            ].nunique()
        )
        label_caliber = pd.DataFrame(
            [
                {"口径": "原始记录数", "数值": int(len(target_raw)), "说明": "目标用户表行数（同一用户可多行）"},
                {"口径": "去重用户数", "数值": int(target_raw[C.COL_USER_ID].nunique()), "说明": "按用户编号去重"},
                {
                    "口径": "任一记录购买（本文采用）",
                    "数值": int(positive.sum()),
                    "说明": "任一状态为购买即正例，避免漏掉购买后仍有浏览/搜索的用户",
                },
                {
                    "口径": "末次记录状态为购买",
                    "数值": last_purchase_users,
                    "说明": "仅按最后一条记录判定，会漏标签",
                },
                {"口径": "多状态用户数", "数值": multi_state_users, "说明": "同一用户出现两种及以上状态"},
                {
                    "口径": "购买后仍有浏览/搜索",
                    "数值": post_purchase_users,
                    "说明": "购买后行为记录存在，末状态口径不完整",
                },
            ]
        )

        attachment2 = self._attachment2_raw()
        decision_column = C.COL_A2_DECISION if C.COL_A2_DECISION in attachment2.columns else None
        decision_filled = int(attachment2[decision_column].notna().sum()) if decision_column else 0

        issues: List[str] = []
        issues.append(
            f"目标用户表存在同一用户多行（重复 {duplicate_rows.get(C.SHEET_TARGET_USERS, 0)} 行），"
            "对应“浏览/搜索/购买”多状态记录。购买标签统一取“任一记录出现购买”："
            f"任一购买 {int(positive.sum())} 人，末次记录为购买仅 {last_purchase_users} 人，"
            f"差 {int(positive.sum()) - last_purchase_users} 人来自购买后仍有浏览/搜索记录。"
        )
        issues.append(
            "“在某购物平台上的平均每次停留的时间    分:秒”列实际存储的是时间戳"
            f"（{time_ranges.get(C.SHEET_TARGET_USERS, '未知')}），并非时长；"
            "本文仅将其用于确定观察截止时间与描述性统计，不作为模型特征。"
        )
        for sheet in (C.SHEET_IDENTITY, C.SHEET_BEHAVIOR_TAGS, C.SHEET_ECOMMERCE, C.SHEET_VIDEO, C.SHEET_MEDIA):
            issues.append(
                f"表“{sheet}”仅覆盖 {coverage.get(sheet, 0):.1%} 的目标用户，"
                "缺失用户不删除，而是保留并用 has_* 标记与中位数填补处理。"
            )
        issues.append(
            f"附件2 的“{C.COL_A2_DECISION}”列非空值 {decision_filled} 个，"
            "即待判用户没有真实标签，因此不能计算其预测精度，只能给出概率与分层。"
        )
        issues.append(
            "电商/视频行为的时间范围超出 11 月（最晚至 12-07），存在观察窗口不一致。"
            "本文按用户截断行为数据：购买者只取首次购买之前的记录，未购买者取统一窗口截止"
            f"（{WINDOW_END:%Y-%m-%d %H:%M}），附件2 只取待判时间之前的记录；"
            "触媒表无时间戳，只能作为静态渠道画像，不能做时点截断。"
        )

        audit = DataAudit(
            generated_at=datetime.now().isoformat(timespec="seconds"),
            sheet_rows=sheet_rows,
            sheet_unique_users=sheet_unique,
            duplicate_rows=duplicate_rows,
            coverage=coverage,
            quality_issues=issues,
            extra={
                "target_unique_users": target_unique,
                "positive_users": int(positive.sum()),
                "positive_rate": positive_rate,
                "state_counts": state_counts,
                "label_caliber": label_caliber.to_dict(orient="records"),
                "time_ranges": time_ranges,
                "attachment2_users": int(len(attachment2)),
                "attachment2_decision_filled": decision_filled,
            },
        )
        self.logger.info(
            "数据审计完成：附件1 八张表，目标用户 %d 人，购买者 %d 人（%.2f%%）",
            target_unique,
            int(positive.sum()),
            100 * positive_rate,
        )
        return audit

    def save_data_quality_report(self, audit: DataAudit) -> pd.DataFrame:
        report = audit.to_frame()
        report["说明"] = ""
        for index, row in report.iterrows():
            sheet = row["数据表"]
            if sheet in audit.extra.get("time_ranges", {}):
                report.at[index, "说明"] = f"时间范围：{audit.extra['time_ranges'][sheet]}"
        self.artifacts.save_table(report, "data_quality_report.csv")
        caliber = audit.extra.get("label_caliber") or []
        if caliber:
            self.artifacts.save_table(pd.DataFrame(caliber), "表1-1B_购买标签口径.csv")
        issue_text = "\n".join(f"{i + 1}. {issue}" for i, issue in enumerate(audit.quality_issues))
        self.artifacts.save_text(
            "数据质量问题清单\n" + "=" * 30 + "\n" + issue_text + "\n",
            "data_quality_issues.txt",
        )
        return report


__all__ = ["DataServiceImpl"]
