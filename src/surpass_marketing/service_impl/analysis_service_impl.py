# -*- coding: utf-8 -*-
"""问题 1~3 的分析服务实现：描述统计、统计检验、可解释模型与用户画像。"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import AppConfig
from ..domain import constants as C
from ..domain.entities import QuestionResult
from ..domain.enums import PotentialTier
from ..repository.artifact_repository import ArtifactRepository
from ..service import AnalysisService, DataService, ModelingService
from ..utils.plotting import HEATMAP_CMAP, PALETTE, setup_plot_style, style_axes
from .feature_columns import preference_indicator_columns, select_features
from .feature_labels import feature_label
from .modeling_service_impl import ModelSuite


class AnalysisServiceImpl(AnalysisService):
    """统计分析与可解释建模实现。"""

    def __init__(
        self,
        config: AppConfig,
        data_service: DataService,
        modeling_service: ModelingService,
        artifact_repository: ArtifactRepository,
        logger: Optional[logging.Logger] = None,
        prediction_service: Optional[Any] = None,
    ):
        self.config = config
        self.data = data_service
        self.modeling = modeling_service
        self.artifacts = artifact_repository
        self.logger = logger or logging.getLogger(__name__)
        self.prediction = prediction_service
        self.seed = config.runtime.seed
        self.test_size = config.runtime.test_size
        self.cv_folds = config.runtime.cv_folds

    # ------------------------------------------------------------------ #
    # 通用小工具
    # ------------------------------------------------------------------ #
    def _table(self, frame: pd.DataFrame, filename: str) -> str:
        return self.artifacts.save_table(frame, filename).name

    def _figure(self, fig, filename: str) -> str:
        name = self.artifacts.save_figure(fig, filename).name
        plt.close(fig)
        return name

    def _logistic_pipeline(self) -> Pipeline:
        logistic_cfg = self.config.model.get("logistic", {})
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=float(logistic_cfg.get("C", 0.5)),
                        max_iter=int(logistic_cfg.get("max_iter", 3000)),
                        class_weight=logistic_cfg.get("class_weight", "balanced"),
                        solver="lbfgs",
                    ),
                ),
            ]
        )

    def _coefficient_frame(self, coefficients: pd.Series) -> pd.DataFrame:
        if coefficients is None or coefficients.empty:
            return pd.DataFrame(columns=["特征", "中文含义", "标准化系数", "机会比OR", "影响方向"])
        frame = coefficients.rename("标准化系数").to_frame()
        frame.insert(0, "特征", frame.index)
        frame.insert(1, "中文含义", [feature_label(name) for name in frame["特征"]])
        frame["机会比OR"] = np.exp(frame["标准化系数"])
        frame["影响方向"] = np.where(frame["标准化系数"] >= 0, "正向", "负向")
        frame["绝对系数"] = frame["标准化系数"].abs()
        frame = frame.sort_values("绝对系数", ascending=False).drop(columns="绝对系数")
        return frame.reset_index(drop=True)

    @staticmethod
    def _purchase_rate(frame: pd.DataFrame, mask: pd.Series) -> float:
        subset = frame.loc[mask]
        return float(subset["target"].mean()) if len(subset) else float("nan")

    # ------------------------------------------------------------------ #
    # 问题 1：数据审计与描述统计
    # ------------------------------------------------------------------ #
    def run_question1(self, users: pd.DataFrame) -> QuestionResult:
        setup_plot_style()
        tables: List[str] = []
        figures: List[str] = []
        total = len(users)
        base_rate = float(users["target"].mean())

        audit = self.data.audit()
        tables.append(self._table(audit.to_frame(), "表1-1_数据审计.csv"))

        # ---------- 表1-2 跟踪状态分布 ----------
        state_rows: List[Dict[str, Any]] = []
        for state in ["浏览", "搜索", "购买"]:
            mask = users["state_set"].fillna("").str.contains(state)
            count = int(mask.sum())
            buys = int(users.loc[mask, "target"].sum())
            state_rows.append(
                {
                    "状态/组合": state,
                    "人数": count,
                    "占目标用户比例": count / total,
                    "其中购买人数": buys,
                    "该组购买率": buys / count if count else float("nan"),
                }
            )
        state_set = users["state_set"].fillna("")
        for combo, count in state_set.value_counts().items():
            if "|" in combo:
                mask = state_set == combo
                buys = int(users.loc[mask, "target"].sum())
                state_rows.append(
                    {
                        "状态/组合": f"多状态:{combo}",
                        "人数": int(count),
                        "占目标用户比例": int(count) / total,
                        "其中购买人数": buys,
                        "该组购买率": buys / int(count),
                    }
                )
        state_rows.append(
            {
                "状态/组合": "合计（唯一用户）",
                "人数": total,
                "占目标用户比例": 1.0,
                "其中购买人数": int(users["target"].sum()),
                "该组购买率": base_rate,
            }
        )
        state_frame = pd.DataFrame(state_rows)
        tables.append(self._table(state_frame, "表1-2_跟踪状态分布.csv"))

        # ---------- 表1-3 人口属性分布 ----------
        age_reverse = {value: key for key, value in C.AGE_ORDER.items()}
        edu_reverse = {value: key for key, value in C.EDUCATION_ORDER.items()}
        age_label = users["age"].map(age_reverse).fillna("未知")
        gender_label = users["gender"].map({1.0: "女", 0.0: "男"}).fillna("未知")
        edu_label = users["edu"].map(edu_reverse).fillna("未知")
        occupation_columns = [column for column in users.columns if column.startswith("occ_")]
        occupation_label = pd.Series("未知", index=users.index, dtype="object")
        if occupation_columns:
            known = users[occupation_columns].fillna(0).sum(axis=1) > 0
            if known.any():
                indices = users.loc[known, occupation_columns].fillna(0).values.argmax(axis=1)
                occupation_label.loc[known] = [C.OCCUPATIONS[int(i)] for i in indices]

        population_rows: List[Dict[str, Any]] = []

        def add_dimension(dimension: str, labels: pd.Series) -> None:
            subset = pd.DataFrame({"类别": labels.values, "target": users["target"].values})
            grouped = subset.groupby("类别")["target"].agg(["count", "sum", "mean"])
            grouped = grouped.sort_values("count", ascending=False)
            contingency = pd.crosstab(subset["类别"], subset["target"])
            p_value = float("nan")
            if contingency.shape[1] == 2 and contingency.shape[0] > 1:
                p_value = float(stats.chi2_contingency(contingency)[1])
            for category, row in grouped.iterrows():
                population_rows.append(
                    {
                        "维度": dimension,
                        "类别": category,
                        "人数": int(row["count"]),
                        "占比": float(row["count"]) / total,
                        "购买人数": int(row["sum"]),
                        "购买率": float(row["mean"]),
                        "卡方p值": p_value,
                    }
                )

        add_dimension("年龄", age_label)
        add_dimension("性别", gender_label)
        add_dimension("学历", edu_label)
        add_dimension("职业", occupation_label)
        population_frame = pd.DataFrame(population_rows)
        tables.append(self._table(population_frame, "表1-3_人口属性分布.csv"))

        # ---------- 表1-4 数据覆盖率 ----------
        coverage_names = {
            "has_identity": "身份标签",
            "has_tags": "行为标签",
            "has_ecom": "电商行为",
            "has_video": "视频行为",
            "has_media": "触媒行为",
        }
        coverage_rows: List[Dict[str, Any]] = []
        for column, label in coverage_names.items():
            if column not in users.columns:
                continue
            covered = users[column].fillna(0) > 0
            uncovered = ~covered
            contingency = pd.crosstab(covered, users["target"])
            p_value = float("nan")
            if contingency.shape == (2, 2):
                p_value = float(stats.chi2_contingency(contingency)[1])
            coverage_rows.append(
                {
                    "数据来源": label,
                    "覆盖人数": int(covered.sum()),
                    "覆盖率": float(covered.mean()),
                    "覆盖组购买率": self._purchase_rate(users, covered),
                    "未覆盖组购买率": self._purchase_rate(users, uncovered),
                    "卡方p值": p_value,
                }
            )
        coverage_frame = pd.DataFrame(coverage_rows)
        tables.append(self._table(coverage_frame, "表1-4_数据覆盖率.csv"))

        # ---------- 表1-5 行为指数描述统计 ----------
        indicator_names = {
            "idx_active": "网络活跃指数",
            "idx_shopping": "网络购物指数",
            "idx_video": "在线视频指数",
            "idx_travel": "出行指数",
            "idx_study": "学习指数",
            "idx_finance": "理财指数",
            "idx_baby": "母婴指数",
            "idx_count": "行为指数覆盖数",
        }
        index_rows: List[Dict[str, Any]] = []
        for column, label in indicator_names.items():
            if column not in users.columns:
                continue
            series = users[column]
            valid = series.notna()
            purchase = series[valid & (users["target"] == 1)]
            non_purchase = series[valid & (users["target"] == 0)]
            p_value = float("nan")
            effect = float("nan")
            if len(purchase) and len(non_purchase):
                statistic, p_value = stats.mannwhitneyu(purchase, non_purchase, alternative="two-sided")
                effect = 2.0 * statistic / (len(purchase) * len(non_purchase)) - 1.0
            index_rows.append(
                {
                    "指数": label,
                    "样本量": int(valid.sum()),
                    "均值": float(series[valid].mean()),
                    "标准差": float(series[valid].std()),
                    "中位数": float(series[valid].median()),
                    "P25": float(series[valid].quantile(0.25)),
                    "P75": float(series[valid].quantile(0.75)),
                    "购买组均值": float(purchase.mean()) if len(purchase) else float("nan"),
                    "未购买组均值": float(non_purchase.mean()) if len(non_purchase) else float("nan"),
                    "差值": float(purchase.mean() - non_purchase.mean())
                    if len(purchase) and len(non_purchase)
                    else float("nan"),
                    "MWU_p值": float(p_value),
                    "效应量rank_biserial": float(effect),
                }
            )
        index_frame = pd.DataFrame(index_rows)
        tables.append(self._table(index_frame, "表1-5_行为指数描述统计.csv"))

        # ---------- 表1-6 时间行为分布 ----------
        time_rows: List[Dict[str, Any]] = []
        if "track_hour" in users.columns:
            bins = [-0.5, 5.5, 8.5, 11.5, 13.5, 17.5, 22.5, 23.5]
            labels = ["0-5时", "6-8时", "9-11时", "12-13时", "14-17时", "18-22时", "23时"]
            hour_group = pd.cut(users["track_hour"], bins=bins, labels=labels)
            for label, subset in users.groupby(hour_group, observed=True):
                time_rows.append(
                    {
                        "维度": "最后跟踪时段",
                        "类别": str(label),
                        "人数": int(len(subset)),
                        "购买率": float(subset["target"].mean()),
                    }
                )
        if "track_weekday" in users.columns:
            weekday_labels = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
            weekday_group = users["track_weekday"].map(
                lambda value: weekday_labels[int(value)] if pd.notna(value) and 0 <= int(value) <= 6 else "未知"
            )
            for label, subset in users.groupby(weekday_group):
                time_rows.append(
                    {
                        "维度": "最后跟踪星期",
                        "类别": str(label),
                        "人数": int(len(subset)),
                        "购买率": float(subset["target"].mean()),
                    }
                )
        time_frame = pd.DataFrame(time_rows)
        tables.append(self._table(time_frame, "表1-6_时间行为分布.csv"))

        # ---------- 图1-1 状态分布 ----------
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        single = state_frame[state_frame["状态/组合"].isin(["浏览", "搜索", "购买"])]
        bars = ax.bar(single["状态/组合"], single["人数"], color=[PALETTE[0], PALETTE[1], PALETTE[2]])
        for bar, rate in zip(bars, single["该组购买率"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{int(bar.get_height())}\n购买率{rate:.1%}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ax.set_ylabel("人数")
        ax.set_title(f"目标用户跟踪状态分布（N={total}，总体购买率 {base_rate:.2%}）")
        style_axes(ax)
        figures.append(self._figure(fig, "图1-1_跟踪状态分布.png"))

        # ---------- 图1-2 人口结构 ----------
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
        for ax, dimension in zip(axes, ["年龄", "学历"]):
            subset = population_frame[population_frame["维度"] == dimension].sort_values("人数")
            ax.barh(subset["类别"], subset["人数"], color=PALETTE[3])
            for y, (count, rate) in enumerate(zip(subset["人数"], subset["购买率"])):
                ax.text(count, y, f" {count}（{rate:.1%}）", va="center", fontsize=8)
            ax.set_title(f"{dimension}结构（括号内为购买率）")
            ax.set_xlabel("人数")
            style_axes(ax)
        fig.tight_layout()
        figures.append(self._figure(fig, "图1-2_人口结构.png"))

        # ---------- 图1-3 数据覆盖率 ----------
        fig, ax = plt.subplots(figsize=(7.2, 3.8))
        bars = ax.bar(coverage_frame["数据来源"], coverage_frame["覆盖率"], color=PALETTE[4])
        for bar, rate in zip(bars, coverage_frame["覆盖率"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{rate:.1%}", ha="center", va="bottom")
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("目标用户覆盖率")
        ax.set_title("各类数据对目标用户的覆盖率")
        style_axes(ax)
        figures.append(self._figure(fig, "图1-3_数据覆盖率.png"))

        # ---------- 图1-4 行为指数箱线图 ----------
        box_columns = [column for column in indicator_names if column in users.columns and column != "idx_count"]
        fig, ax = plt.subplots(figsize=(9.0, 4.2))
        data_non_purchase = [users.loc[users["target"] == 0, column].dropna().values for column in box_columns]
        data_purchase = [users.loc[users["target"] == 1, column].dropna().values for column in box_columns]
        positions = np.arange(len(box_columns))
        ax.boxplot(data_non_purchase, positions=positions - 0.18, widths=0.3, showfliers=False, patch_artist=True,
                   boxprops={"facecolor": PALETTE[0], "alpha": 0.65}, medianprops={"color": "white"})
        ax.boxplot(data_purchase, positions=positions + 0.18, widths=0.3, showfliers=False, patch_artist=True,
                   boxprops={"facecolor": PALETTE[2], "alpha": 0.8}, medianprops={"color": "white"})
        ax.set_xticks(positions)
        ax.set_xticklabels([indicator_names[column] for column in box_columns], rotation=15)
        ax.set_ylabel("指数取值")
        ax.set_title("未购买（蓝）与购买（橙）用户的行为指数分布")
        ax.legend([plt.Rectangle((0, 0), 1, 1, color=PALETTE[0], alpha=0.65),
                   plt.Rectangle((0, 0), 1, 1, color=PALETTE[2], alpha=0.8)],
                  ["未购买", "购买"], frameon=False)
        style_axes(ax)
        figures.append(self._figure(fig, "图1-4_行为指数箱线图.png"))

        # ---------- 图1-5 活跃时段 ----------
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
        hours = np.arange(24)
        counts_all = users["track_hour"].value_counts().reindex(hours, fill_value=0)
        rate_by_hour = users.groupby(users["track_hour"].round())["target"].mean().reindex(hours)
        axes[0].bar(hours, counts_all.values, color=PALETTE[0])
        axes[0].set_title("最后跟踪时段人数分布")
        axes[0].set_xlabel("小时")
        axes[0].set_ylabel("人数")
        style_axes(axes[0])
        axes[1].plot(hours, rate_by_hour.values, marker="o", color=PALETTE[2])
        axes[1].axhline(base_rate, linestyle="--", color="gray", linewidth=1, label=f"总体 {base_rate:.1%}")
        axes[1].set_title("各时段购买率")
        axes[1].set_xlabel("小时")
        axes[1].set_ylabel("购买率")
        axes[1].legend(frameon=False, fontsize=8)
        style_axes(axes[1])
        fig.tight_layout()
        figures.append(self._figure(fig, "图1-5_活跃时段分布.png"))

        notes = (
            f"目标用户 {total} 人，购买者 {int(users['target'].sum())} 人（{base_rate:.2%}）；"
            "发现重复行、时间列语义错位、各行为表覆盖率差异等数据质量问题。"
        )
        self.logger.info("问题1完成：%d 张表、%d 张图", len(tables), len(figures))
        return QuestionResult(
            name="问题1：数据预处理与描述性统计",
            tables=tables,
            figures=figures,
            metrics={
                "目标用户数": total,
                "购买人数": int(users["target"].sum()),
                "总体购买率": base_rate,
                "数据覆盖率": dict(zip(coverage_frame["数据来源"], coverage_frame["覆盖率"])),
            },
            notes=notes,
        )

    # ------------------------------------------------------------------ #
    # 问题 2：基本行为特征对购买行为的影响
    # ------------------------------------------------------------------ #
    def _univariate_tests(self, users: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        target = users["target"]
        for column in features:
            if column not in users.columns:
                continue
            series = users[column]
            valid = series.notna()
            if int(valid.sum()) < 100:
                continue
            if series[valid].nunique() <= 2:
                subset = pd.DataFrame({"x": series[valid].astype(int), "y": target[valid]})
                contingency = pd.crosstab(subset["x"], subset["y"])
                if contingency.shape != (2, 2):
                    continue
                statistic, p_value, _, _ = stats.chi2_contingency(contingency)
                n = int(contingency.values.sum())
                effect = math.sqrt(statistic / (n * (min(contingency.shape) - 1)))
                rates = subset.groupby("x")["y"].mean()
                direction = f"取值0：{rates.get(0, 0):.2%}，取值1：{rates.get(1, 0):.2%}"
                rows.append(
                    {
                        "特征": column,
                        "中文含义": feature_label(column),
                        "检验方法": "卡方检验",
                        "统计量": float(statistic),
                        "p值": float(p_value),
                        "效应量": float(effect),
                        "效应量说明": "Cramér's V",
                        "方向说明": direction,
                    }
                )
            else:
                purchase = series[valid & (target == 1)]
                non_purchase = series[valid & (target == 0)]
                if len(purchase) < 10 or len(non_purchase) < 10:
                    continue
                statistic, p_value = stats.mannwhitneyu(purchase, non_purchase, alternative="two-sided")
                effect = 2.0 * statistic / (len(purchase) * len(non_purchase)) - 1.0
                rows.append(
                    {
                        "特征": column,
                        "中文含义": feature_label(column),
                        "检验方法": "Mann-Whitney U",
                        "统计量": float(statistic),
                        "p值": float(p_value),
                        "效应量": float(effect),
                        "效应量说明": "rank-biserial",
                        "方向说明": f"购买组中位数 {purchase.median():.3f}，未购组 {non_purchase.median():.3f}",
                    }
                )
        frame = pd.DataFrame(rows)
        if not frame.empty:
            frame = frame.sort_values("p值").reset_index(drop=True)
        return frame

    def _group_lift(self, users: pd.DataFrame, binary_columns: Sequence[str], continuous_columns: Sequence[str]) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        base_rate = float(users["target"].mean())
        for column in binary_columns:
            if column not in users.columns:
                continue
            series = users[column]
            for value in sorted(series.dropna().unique()):
                mask = series == value
                count = int(mask.sum())
                if count == 0:
                    continue
                rate = float(users.loc[mask, "target"].mean())
                rows.append(
                    {
                        "特征": column,
                        "中文含义": feature_label(column),
                        "分组": f"{'有/是' if value else '无/否'}（={int(value)}）",
                        "人数": count,
                        "购买人数": int(users.loc[mask, "target"].sum()),
                        "购买率": rate,
                        "相对整体提升": rate / base_rate - 1.0,
                    }
                )
        for column in continuous_columns:
            if column not in users.columns:
                continue
            series = users[column]
            valid = series.notna()
            if int(valid.sum()) < 100:
                continue
            median = float(series[valid].median())
            for label, mask in [("低（≤中位数）", series <= median), ("高（>中位数）", series > median)]:
                mask = mask & valid
                count = int(mask.sum())
                rate = float(users.loc[mask, "target"].mean())
                rows.append(
                    {
                        "特征": column,
                        "中文含义": feature_label(column),
                        "分组": label,
                        "人数": count,
                        "购买人数": int(users.loc[mask, "target"].sum()),
                        "购买率": rate,
                        "相对整体提升": rate / base_rate - 1.0,
                    }
                )
        return pd.DataFrame(rows)

    def _model_comparison_figure(self, comparison: pd.DataFrame, filename: str, title: str) -> str:
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
        models = comparison["模型"].tolist()
        axes[0].bar(models, comparison["5折CV_PR_AUC均值"], color=PALETTE[0])
        axes[0].set_title("5 折交叉验证 PR-AUC")
        axes[0].set_ylim(max(0.0, comparison["5折CV_PR_AUC均值"].min() - 0.05), 1.0)
        axes[0].tick_params(axis="x", labelrotation=10)
        style_axes(axes[0])
        axes[1].bar(models, comparison["测试集ROC_AUC"], color=PALETTE[2])
        axes[1].set_title("测试集 ROC-AUC")
        axes[1].set_ylim(max(0.0, comparison["测试集ROC_AUC"].min() - 0.05), 1.0)
        axes[1].tick_params(axis="x", labelrotation=10)
        style_axes(axes[1])
        fig.suptitle(title)
        fig.tight_layout()
        return self._figure(fig, filename)

    def _curve_figure(self, y_true: pd.Series, prob: np.ndarray, filename: str, title: str) -> str:
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
        fpr, tpr, _ = roc_curve(y_true, prob)
        axes[0].plot(fpr, tpr, color=PALETTE[2], linewidth=2, label=f"AUC={roc_auc_score(y_true, prob):.4f}")
        axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        axes[0].set_xlabel("假正率 FPR")
        axes[0].set_ylabel("真正率 TPR")
        axes[0].set_title("ROC 曲线")
        axes[0].legend(frameon=False, fontsize=8)
        style_axes(axes[0])
        precision, recall, _ = precision_recall_curve(y_true, prob)
        axes[1].plot(recall, precision, color=PALETTE[2], linewidth=2,
                     label=f"AP={average_precision_score(y_true, prob):.4f}")
        axes[1].axhline(float(y_true.mean()), linestyle="--", color="gray", linewidth=1, label="基准阳性率")
        axes[1].set_xlabel("召回率 Recall")
        axes[1].set_ylabel("精确率 Precision")
        axes[1].set_title("PR 曲线")
        axes[1].legend(frameon=False, fontsize=8)
        style_axes(axes[1])
        fig.suptitle(title)
        fig.tight_layout()
        return self._figure(fig, filename)

    def run_question2(self, users: pd.DataFrame) -> QuestionResult:
        setup_plot_style()
        features = select_features(users, "q2")
        self.logger.info("问题2使用 %d 个基本行为特征", len(features))
        suite: ModelSuite = self.modeling.run_model_suite(users, features, "Q2")

        tables: List[str] = [self._table(suite.comparison, "表2-1_Q2模型对比.csv")]
        coefficient_frame = self._coefficient_frame(suite.coefficients)
        tables.append(self._table(coefficient_frame, "表2-2_Q2_Logistic系数.csv"))
        tests = self._univariate_tests(users, features)
        tables.append(self._table(tests, "表2-3_Q2单变量统计检验.csv"))

        binary_columns = [
            "gender",
            "has_identity",
            "has_tags",
            "has_ecom",
            "has_video",
            "has_media",
            "track_is_weekend",
            "track_is_night",
        ]
        continuous_columns = ["idx_active", "idx_shopping", "vol_ecom_n", "vol_media_n", "vol_video_n", "idx_count"]
        lift = self._group_lift(users, binary_columns, continuous_columns)
        tables.append(self._table(lift, "表2-4_Q2分组购买率.csv"))

        figures = [
            self._model_comparison_figure(suite.comparison, "图2-1_Q2模型对比.png", "问题2 候选模型性能对比"),
            self._curve_figure(suite.y_test, suite.prob_test, "图2-3_Q2_ROC与PR曲线.png", "问题2 入选模型判别能力"),
        ]

        fig, ax = plt.subplots(figsize=(8.0, 5.4))
        top = coefficient_frame.head(15).iloc[::-1]
        colors = [PALETTE[2] if value > 0 else PALETTE[0] for value in top["标准化系数"]]
        ax.barh(top["中文含义"], top["标准化系数"], color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("标准化 Logistic 系数（正=提升购买概率）")
        ax.set_title("影响购买行为的 Top15 基本行为特征")
        style_axes(ax)
        figures.append(self._figure(fig, "图2-2_Q2系数.png"))

        metrics = {
            "入选模型": suite.bundle.model_name,
            "测试集PR_AUC": suite.bundle.metrics.pr_auc,
            "测试集ROC_AUC": suite.bundle.metrics.roc_auc,
            "测试集F1": suite.bundle.metrics.f1,
            "测试集Brier": suite.bundle.metrics.brier,
            "阳性率": suite.bundle.positive_rate,
            "特征数": len(features),
        }
        notes = (
            f"以 5 折交叉验证 PR-AUC 选型，入选 {suite.bundle.model_name}；"
            f"PR-AUC={suite.bundle.metrics.pr_auc:.4f}，ROC-AUC={suite.bundle.metrics.roc_auc:.4f}。"
        )
        self.logger.info("问题2完成：%d 张表、%d 张图", len(tables), len(figures))
        return QuestionResult(
            name="问题2：基本行为特征对购买行为的影响",
            tables=tables,
            figures=figures,
            metrics=metrics,
            notes=notes,
        )

    # ------------------------------------------------------------------ #
    # 问题 3：个人偏好的影响、降维与聚类画像
    # ------------------------------------------------------------------ #
    def build_preference_indicators(self, frame: pd.DataFrame) -> pd.DataFrame:
        """构造 8 个可解释偏好指标：成分先 z 标准化再取均值。"""
        result = pd.DataFrame(index=frame.index)
        result[C.COL_USER_ID] = frame[C.COL_USER_ID] if C.COL_USER_ID in frame.columns else np.nan
        result["target"] = frame["target"]
        for name, columns in C.PREFERENCE_INDICATORS.items():
            available = [column for column in columns if column in frame.columns]
            if not available:
                continue
            parts: List[pd.Series] = []
            for column in available:
                series = frame[column].astype(float)
                std = float(series.std())
                parts.append((series - series.mean()) / std if std and not np.isnan(std) else series * 0.0)
            result[f"ind_{name}"] = pd.concat(parts, axis=1).mean(axis=1)
        return result

    def _indicator_pca(self, indicators: pd.DataFrame) -> Tuple[PCA, np.ndarray, int, pd.DataFrame, np.ndarray]:
        columns = [column for column in indicators.columns if column.startswith("ind_")]
        imputed = SimpleImputer(strategy="median").fit_transform(indicators[columns])
        scaled = StandardScaler().fit_transform(imputed)
        pca = PCA().fit(scaled)
        eigenvalues = pca.explained_variance_
        kaiser = int((eigenvalues > 1.0).sum())
        n_components = max(2, min(len(columns), kaiser)) if kaiser >= 2 else max(2, min(len(columns), 3))
        scores = pca.transform(scaled)[:, :n_components]
        loading = pd.DataFrame(
            pca.components_[:n_components],
            columns=[feature_label(column) for column in columns],
        )
        loading.insert(0, "主成分", [f"PC{i + 1}" for i in range(n_components)])
        return pca, scores, n_components, loading, scaled

    def build_personas(
        self, indicators: pd.DataFrame, k_range: Sequence[int] = (2, 3, 4, 5, 6)
    ) -> Tuple[np.ndarray, int, float, pd.DataFrame]:
        """KMeans 画像：轮廓系数选 k，返回标签、k、轮廓系数与画像表。"""
        columns = [column for column in indicators.columns if column.startswith("ind_")]
        imputed = SimpleImputer(strategy="median").fit_transform(indicators[columns])
        scaled = StandardScaler().fit_transform(imputed)
        best: Optional[Tuple[float, int, KMeans]] = None
        for k in k_range:
            if k >= len(scaled):
                continue
            model = KMeans(n_clusters=k, n_init=10, random_state=self.seed).fit(scaled)
            score = silhouette_score(
                scaled, model.labels_, sample_size=min(4000, len(scaled)), random_state=self.seed
            )
            if best is None or score > best[0]:
                best = (float(score), int(k), model)
        if best is None:  # pragma: no cover - 防御式编程
            raise RuntimeError("KMeans 未找到可行解")
        silhouette, k, model = best
        labels = model.labels_
        profile = pd.DataFrame(scaled, columns=columns).groupby(labels).mean()
        profile.columns = [feature_label(column) for column in profile.columns]
        sizes = pd.Series(labels).value_counts().sort_index()
        purchase_rate = indicators.assign(集群=labels).groupby("集群")["target"].mean()
        profile.insert(0, "人数", sizes.values)
        profile.insert(1, "占比", sizes.values / len(labels))
        profile.insert(2, "购买率", purchase_rate.reindex(profile.index).values)
        profile.insert(3, "画像标签", [
            f"{self._persona_label(row)}·群体{i + 1}" for i, (_, row) in enumerate(profile.iterrows())
        ])
        profile.index = [f"群体{i + 1}" for i in range(len(profile))]
        profile.index.name = "类别"
        return labels, k, silhouette, profile

    @staticmethod
    def _persona_label(row: pd.Series) -> str:
        values = row.drop(labels=["人数", "占比", "购买率", "画像标签"], errors="ignore")
        values = pd.to_numeric(values, errors="coerce").dropna()
        if values.empty:
            return "综合型"
        top = values.sort_values(ascending=False).head(2)
        return "·".join([f"{name}突出" if value > 0.15 else name for name, value in top.items()])

    def _compare_logistic(
        self, frame: pd.DataFrame, schemes: Dict[str, Sequence[str]]
    ) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, pd.Series], pd.Series, pd.Series]:
        target = frame["target"].astype(int)
        indices = np.arange(len(frame))
        train_index, test_index = train_test_split(
            indices, test_size=self.test_size, stratify=target, random_state=self.seed
        )
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.seed)
        rows: Dict[str, Dict[str, float]] = {}
        probabilities: Dict[str, np.ndarray] = {}
        coefficients: Dict[str, pd.Series] = {}
        for name, columns in schemes.items():
            available = [column for column in columns if column in frame.columns]
            matrix = frame[available].astype(float)
            pipeline = self._logistic_pipeline()
            cv_pr = cross_val_score(
                pipeline, matrix.iloc[train_index], target.iloc[train_index],
                cv=cv, scoring="average_precision", error_score=np.nan,
            )
            cv_roc = cross_val_score(
                pipeline, matrix.iloc[train_index], target.iloc[train_index],
                cv=cv, scoring="roc_auc", error_score=np.nan,
            )
            pipeline.fit(matrix.iloc[train_index], target.iloc[train_index])
            probability = pipeline.predict_proba(matrix.iloc[test_index])[:, 1]
            y_test = target.iloc[test_index]
            rows[name] = {
                "特征数": len(available),
                "5折CV_PR_AUC": float(np.nanmean(cv_pr)),
                "5折CV_ROC_AUC": float(np.nanmean(cv_roc)),
                "测试集PR_AUC": float(average_precision_score(y_test, probability)),
                "测试集ROC_AUC": float(roc_auc_score(y_test, probability)),
                "测试集F1": float(f1_score(y_test, (probability >= 0.5).astype(int), zero_division=0)),
                "测试集Brier": float(brier_score_loss(y_test, np.clip(probability, 1e-6, 1 - 1e-6))),
            }
            probabilities[name] = probability
            coefficients[name] = pd.Series(
                pipeline.named_steps["clf"].coef_.ravel(), index=available, name="标准化系数"
            )
        comparison = pd.DataFrame(rows).T.reset_index().rename(columns={"index": "模型方案"})
        return comparison, probabilities, coefficients, target.iloc[test_index].reset_index(drop=True), target

    def run_question3(self, users: pd.DataFrame) -> QuestionResult:
        setup_plot_style()
        indicators = self.build_preference_indicators(users)
        indicator_columns = [column for column in indicators.columns if column.startswith("ind_")]
        # 给指标列一个可读名
        readable = {column: feature_label(column) for column in indicator_columns}
        for column in indicator_columns:
            readable.setdefault(column, C.PREFERENCE_INDICATOR_LABELS.get(column.replace("ind_", ""), column))

        tables: List[str] = []
        figures: List[str] = []

        # ---------- 表3-1 偏好指标描述统计 ----------
        descript_rows: List[Dict[str, Any]] = []
        target = users["target"]
        for column in indicator_columns:
            series = indicators[column]
            purchase = series[target == 1].dropna()
            non_purchase = series[target == 0].dropna()
            p_value = float("nan")
            if len(purchase) and len(non_purchase):
                p_value = float(stats.mannwhitneyu(purchase, non_purchase, alternative="two-sided")[1])
            descript_rows.append(
                {
                    "偏好指标": readable[column],
                    "样本量": int(series.notna().sum()),
                    "均值": float(series.mean()),
                    "标准差": float(series.std()),
                    "购买组均值": float(purchase.mean()),
                    "未购买组均值": float(non_purchase.mean()),
                    "差值": float(purchase.mean() - non_purchase.mean()),
                    "MWU_p值": p_value,
                }
            )
        tables.append(self._table(pd.DataFrame(descript_rows), "表3-1_偏好指标描述统计.csv"))

        # ---------- PCA ----------
        pca, scores, n_components, loading, scaled = self._indicator_pca(indicators)
        variance = pd.DataFrame(
            {
                "主成分": [f"PC{i + 1}" for i in range(len(pca.explained_variance_))],
                "特征值": pca.explained_variance_,
                "方差解释率": pca.explained_variance_ratio_,
                "累计方差解释率": np.cumsum(pca.explained_variance_ratio_),
            }
        )
        tables.append(self._table(variance, "表3-2_偏好主成分方差解释.csv"))
        tables.append(self._table(loading, "表3-3_偏好主成分载荷.csv"))

        # ---------- 模型对比：M1 对照 / M2 指标 / M3 主成分 ----------
        q2_features = select_features(users, "q2")
        frame = users[[C.COL_USER_ID, "target"] + q2_features].copy()
        for column in indicator_columns:
            frame[column] = indicators[column].values
        for index in range(n_components):
            frame[f"pc_{index + 1}"] = scores[:, index]
        pc_columns = [f"pc_{index + 1}" for index in range(n_components)]
        schemes = {
            "M1_基本行为(对照组)": q2_features,
            "M2_基本行为+偏好指标": q2_features + indicator_columns,
            "M3_基本行为+主成分": q2_features + pc_columns,
        }
        comparison, probabilities, coefficients, y_test, _ = self._compare_logistic(frame, schemes)
        tables.append(self._table(comparison, "表3-4_Q3模型方案对比.csv"))

        # ---------- 表3-5 M2 偏好指标系数 ----------
        main_coefficients = coefficients["M2_基本行为+偏好指标"].reindex(indicator_columns).dropna()
        coef_frame = pd.DataFrame(
            {
                "偏好指标": [readable[column] for column in main_coefficients.index],
                "标准化系数": main_coefficients.values,
                "机会比OR": np.exp(main_coefficients.values),
                "影响方向": np.where(main_coefficients.values >= 0, "正向", "负向"),
            }
        ).sort_values("标准化系数", key=lambda series: series.abs(), ascending=False)
        tables.append(self._table(coef_frame.reset_index(drop=True), "表3-5_Q3偏好指标系数.csv"))

        # ---------- 聚类画像 ----------
        labels, k, silhouette, profile = self.build_personas(indicators)
        tables.append(self._table(profile.reset_index(), "表3-6_用户画像聚类结果.csv"))

        # ---------- 图3-1 指标相关热图 ----------
        correlation = indicators[indicator_columns].corr()
        fig, ax = plt.subplots(figsize=(7.4, 5.8))
        image = ax.imshow(correlation.values, cmap=HEATMAP_CMAP, vmin=-1, vmax=1)
        ax.set_xticks(range(len(indicator_columns)))
        ax.set_xticklabels([readable[column] for column in indicator_columns], rotation=35, ha="right")
        ax.set_yticks(range(len(indicator_columns)))
        ax.set_yticklabels([readable[column] for column in indicator_columns])
        for i in range(len(indicator_columns)):
            for j in range(len(indicator_columns)):
                ax.text(j, i, f"{correlation.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
        fig.colorbar(image, ax=ax, shrink=0.82)
        ax.set_title("偏好指标相关系数矩阵")
        figures.append(self._figure(fig, "图3-1_偏好指标相关热图.png"))

        # ---------- 图3-2 主成分 ----------
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
        axes[0].bar(variance["主成分"], variance["方差解释率"], color=PALETTE[0], label="方差解释率")
        axes[0].plot(variance["主成分"], variance["累计方差解释率"], marker="o", color=PALETTE[2], label="累计")
        axes[0].axhline(0.75, linestyle="--", color="gray", linewidth=1)
        axes[0].set_title("偏好主成分方差解释")
        axes[0].tick_params(axis="x", labelrotation=30)
        axes[0].legend(frameon=False, fontsize=8)
        style_axes(axes[0])
        loadings_k = loading.set_index("主成分")
        image = axes[1].imshow(loadings_k.values, cmap=HEATMAP_CMAP, aspect="auto")
        axes[1].set_yticks(range(len(loadings_k.index)))
        axes[1].set_yticklabels(loadings_k.index)
        axes[1].set_xticks(range(len(loadings_k.columns)))
        axes[1].set_xticklabels([str(column) for column in loadings_k.columns], rotation=35, ha="right")
        axes[1].set_title("主成分载荷")
        for i in range(loadings_k.shape[0]):
            for j in range(loadings_k.shape[1]):
                axes[1].text(j, i, f"{loadings_k.values[i, j]:.2f}", ha="center", va="center", fontsize=6)
        fig.colorbar(image, ax=axes[1], shrink=0.82)
        fig.tight_layout()
        figures.append(self._figure(fig, "图3-2_偏好主成分分析.png"))

        # ---------- 图3-3 聚类画像 ----------
        profile_values = profile.drop(columns=["人数", "占比", "购买率", "画像标签"])
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
        image = axes[0].imshow(profile_values.values, cmap=HEATMAP_CMAP, aspect="auto")
        axes[0].set_yticks(range(len(profile_values.index)))
        axes[0].set_yticklabels(profile_values.index)
        axes[0].set_xticks(range(len(profile_values.columns)))
        axes[0].set_xticklabels(profile_values.columns, rotation=35, ha="right")
        axes[0].set_title(f"KMeans 画像（k={k}，轮廓系数={silhouette:.3f}）")
        for i in range(profile_values.shape[0]):
            for j in range(profile_values.shape[1]):
                axes[0].text(j, i, f"{profile_values.values[i, j]:.2f}", ha="center", va="center", fontsize=6)
        fig.colorbar(image, ax=axes[0], shrink=0.82)
        axes[1].bar(profile.index, profile["购买率"], color=PALETTE[2])
        for index, (rate, size) in enumerate(zip(profile["购买率"], profile["人数"])):
            axes[1].text(index, rate, f"{rate:.2%}\n(n={size})", ha="center", va="bottom", fontsize=8)
        axes[1].set_title("各画像群体购买率")
        axes[1].tick_params(axis="x", labelrotation=20)
        style_axes(axes[1])
        fig.tight_layout()
        figures.append(self._figure(fig, "图3-3_用户画像聚类.png"))

        metrics = {
            "指标数": len(indicator_columns),
            "主成分数": n_components,
            "聚类数k": k,
            "轮廓系数": silhouette,
            "M2_PR_AUC": float(comparison.loc[comparison["模型方案"] == "M2_基本行为+偏好指标", "测试集PR_AUC"].iloc[0]),
            "M2_ROC_AUC": float(comparison.loc[comparison["模型方案"] == "M2_基本行为+偏好指标", "测试集ROC_AUC"].iloc[0]),
        }
        notes = (
            f"构造 {len(indicator_columns)} 个偏好指标，Kaiser 准则保留 {n_components} 个主成分；"
            f"KMeans 按轮廓系数选 k={k}（轮廓系数 {silhouette:.3f}）。"
        )
        self.logger.info("问题3完成：%d 张表、%d 张图", len(tables), len(figures))
        return QuestionResult(
            name="问题3：个人偏好对购买行为的影响",
            tables=tables,
            figures=figures,
            metrics=metrics,
            notes=notes,
        )

    # ------------------------------------------------------------------ #
    # 问题 4：潜力评估、附件2 预测、Top100 与营销策略
    # ------------------------------------------------------------------ #
    def run_question4(
        self, users: pd.DataFrame, attachment2_features: pd.DataFrame
    ) -> QuestionResult:
        if self.prediction is None:
            raise RuntimeError("问题4需要注入 PredictionService 实现")
        setup_plot_style()
        tables: List[str] = []
        figures: List[str] = []

        # ---------- 训练与选型 ----------
        suite: ModelSuite = self.modeling.train_q4_final(users)
        bundle = suite.bundle
        model_path = self.artifacts.save_model(bundle, "q4_calibrated_model.joblib")
        self.logger.info("问题4 模型已保存：%s", model_path.name)
        thresholds = dict(bundle.tier_thresholds or {})
        tables.append(self._table(suite.comparison, "表4-1_Q4模型对比.csv"))
        if not suite.feature_importance.empty:
            importance_frame = (
                suite.feature_importance.rename("特征重要性")
                .rename_axis("特征")
                .reset_index()
                .sort_values("特征重要性", ascending=False)
            )
            importance_frame["中文含义"] = importance_frame["特征"].map(feature_label)
            tables.append(self._table(importance_frame, "表4-1B_Q4特征重要性.csv"))

        calibrated = suite.extra.get("calibrated_metrics")
        uncalibrated = suite.extra.get("uncalibrated_metrics")
        rows = [
            {"项目": "训练集正样本比例", "数值": float(bundle.positive_rate)},
            {"项目": "测试集ROC_AUC（未校准）", "数值": getattr(uncalibrated, "roc_auc", np.nan)},
            {"项目": "测试集PR_AUC（未校准）", "数值": getattr(uncalibrated, "pr_auc", np.nan)},
            {"项目": "测试集Brier（未校准）", "数值": getattr(uncalibrated, "brier", np.nan)},
            {"项目": "测试集ROC_AUC（校准后）", "数值": getattr(calibrated, "roc_auc", np.nan)},
            {"项目": "测试集PR_AUC（校准后）", "数值": getattr(calibrated, "pr_auc", np.nan)},
            {"项目": "测试集Brier（校准后）", "数值": getattr(calibrated, "brier", np.nan)},
            {"项目": "测试集F1（校准后，按校准后F1阈值）", "数值": getattr(calibrated, "f1", np.nan)},
            {"项目": "OOF最大F1阈值（未校准）", "数值": float(suite.extra.get("threshold", np.nan))},
            {"项目": "校准后F1最优阈值", "数值": float(suite.extra.get("calibrated_threshold", np.nan))},
            {"项目": "高潜力阈值（0.90分位）", "数值": float(thresholds.get("high", np.nan))},
            {"项目": "中潜力阈值（0.70分位）", "数值": float(thresholds.get("medium", np.nan))},
            {"项目": "成本收益平衡概率", "数值": float(thresholds.get("break_even", np.nan))},
        ]
        tables.append(self._table(pd.DataFrame(rows), "表4-2_校准与阈值.csv"))

        # ---------- 全体用户打分与画像 ----------
        scored_users = self.prediction.score_users(bundle, users)
        scored_users["target"] = users["target"].to_numpy()

        combined_raw = pd.concat(
            [users, attachment2_features.assign(target=np.nan)], ignore_index=True
        )
        combined_indicators = self.build_preference_indicators(combined_raw)
        labels, k, silhouette, profile = self.build_personas(combined_indicators)
        persona_names = profile["画像标签"].tolist()
        user_persona = [persona_names[int(label)] for label in labels[: len(users)]]
        scored_users["画像标签"] = user_persona

        # ---------- 附件2 预测 ----------
        attachment2_scored = self.prediction.predict_attachment2(bundle, attachment2_features)
        a2_persona = pd.Series(
            [persona_names[int(label)] for label in labels[len(users):]],
            index=attachment2_features[C.COL_USER_ID].to_numpy(),
        )
        attachment2_scored["画像标签"] = attachment2_scored[C.COL_USER_ID].map(a2_persona)
        tables.append(self._table(attachment2_scored, "表4-3_附件2预测结果.csv"))

        # ---------- 未购用户 Top100 ----------
        top100 = self.prediction.rank_unpurchased_topn(bundle, users, self.config.runtime.top_n)
        persona_by_user = pd.Series(user_persona, index=users[C.COL_USER_ID].to_numpy())
        top100["画像标签"] = top100[C.COL_USER_ID].map(persona_by_user)
        tables.append(self._table(top100, "表4-4_未购Top100.csv"))

        # ---------- 分层营销矩阵 ----------
        plan_frame = pd.concat(
            [
                scored_users[
                    [C.COL_USER_ID, "预测购买概率", "潜力评分", "潜力等级", "画像标签", "target"]
                ],
                attachment2_scored[
                    [C.COL_USER_ID, "预测购买概率", "潜力评分", "潜力等级", "画像标签"]
                ].assign(target=np.nan),
            ],
            ignore_index=True,
        )
        marketing_plan, persona_summary = self.prediction.build_marketing_plan(plan_frame, profile)
        tables.append(self._table(marketing_plan, "表4-5_分层营销矩阵.csv"))
        tables.append(self._table(persona_summary, "表4-6_潜力人群画像.csv"))

        # ---------- 图4-1 概率分布与累积增益 ----------
        prob = np.asarray(suite.prob_test, dtype=float)
        y_test = np.asarray(suite.y_test, dtype=int)
        fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2))
        axes[0].hist(prob[y_test == 0], bins=30, alpha=0.75, label="未购买", color=PALETTE[0])
        axes[0].hist(prob[y_test == 1], bins=30, alpha=0.75, label="购买", color=PALETTE[2])
        if "high" in thresholds:
            axes[0].axvline(thresholds["high"], color="black", ls="--", lw=1.0, label="高潜力阈值")
        axes[0].set_xlabel("预测购买概率")
        axes[0].set_ylabel("人数")
        axes[0].set_title("测试集预测概率分布")
        axes[0].legend(fontsize=8)
        style_axes(axes[0])

        order = np.argsort(-prob)
        sorted_y = y_test[order]
        cumulative = np.cumsum(sorted_y) / max(1, sorted_y.sum())
        ratio = np.arange(1, len(sorted_y) + 1) / len(sorted_y)
        axes[1].plot(ratio, cumulative, color=PALETTE[2], label="校准模型")
        axes[1].plot([0, 1], [0, 1], ls="--", color="gray", label="随机基线")
        axes[1].set_xlabel("按评分从高到低覆盖的用户比例")
        axes[1].set_ylabel("累计捕获购买者比例")
        axes[1].set_title("累积增益曲线")
        axes[1].legend(fontsize=8)
        style_axes(axes[1])
        fig.tight_layout()
        figures.append(self._figure(fig, "图4-1_概率分布与累积增益.png"))

        # ---------- 图4-2 校准曲线 ----------
        bins = np.linspace(0.0, 1.0, 11)
        bin_ids = np.clip(np.digitize(prob, bins) - 1, 0, 9)
        reliability = (
            pd.DataFrame({"概率": prob, "标签": y_test, "区间": bin_ids})
            .groupby("区间")
            .agg(平均预测概率=("概率", "mean"), 实际购买率=("标签", "mean"), 人数=("标签", "size"))
        )
        brier = getattr(calibrated, "brier", getattr(bundle.metrics, "brier", np.nan))
        fig, ax = plt.subplots(figsize=(5.8, 4.6))
        ax.plot([0, 1], [0, 1], ls="--", color="gray", label="完美校准")
        ax.plot(
            reliability["平均预测概率"],
            reliability["实际购买率"],
            marker="o",
            color=PALETTE[2],
            label="校准模型",
        )
        ax.set_xlabel("平均预测概率")
        ax.set_ylabel("实际购买率")
        ax.set_title(f"校准曲线（Brier={brier:.4f}）")
        ax.legend(fontsize=8)
        style_axes(ax)
        fig.tight_layout()
        figures.append(self._figure(fig, "图4-2_校准曲线.png"))

        # ---------- 图4-3 分层规模与 Top100 评分 ----------
        tier_order = [
            PotentialTier.HIGH.value,
            PotentialTier.MEDIUM.value,
            PotentialTier.LOW.value,
        ]
        tier_stats = (
            scored_users.groupby("潜力等级")
            .agg(人数=(C.COL_USER_ID, "size"), 平均评分=("潜力评分", "mean"))
            .reindex(tier_order)
            .dropna(how="all")
        )
        fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2))
        colors = [PALETTE[2], PALETTE[1], PALETTE[0]][: len(tier_stats)]
        axes[0].bar(tier_stats.index, tier_stats["人数"], color=colors)
        for index, (count, score) in enumerate(zip(tier_stats["人数"], tier_stats["平均评分"])):
            axes[0].text(index, count, f"{int(count)}人\n均分{score:.1f}", ha="center", va="bottom", fontsize=8)
        axes[0].set_ylabel("用户人数")
        axes[0].set_title("附件1 全体用户潜力分层")
        style_axes(axes[0])

        axes[1].plot(
            np.arange(1, len(top100) + 1), top100["潜力评分"], color=PALETTE[2], linewidth=1.6
        )
        axes[1].set_xlabel("未购用户排名")
        axes[1].set_ylabel("潜力评分")
        axes[1].set_title(f"未购用户 Top{len(top100)} 评分")
        style_axes(axes[1])
        fig.tight_layout()
        figures.append(self._figure(fig, "图4-3_分层规模与未购Top100.png"))

        # ---------- 图4-4 附件2 潜力评分 ----------
        a2_sorted = attachment2_scored.sort_values("潜力评分", ascending=True)
        tier_color = {
            PotentialTier.HIGH.value: PALETTE[2],
            PotentialTier.MEDIUM.value: PALETTE[1],
            PotentialTier.LOW.value: PALETTE[0],
        }
        bar_colors = [tier_color.get(tier, PALETTE[0]) for tier in a2_sorted["潜力等级"]]
        fig, ax = plt.subplots(figsize=(6.4, 7.2))
        ax.barh([str(uid) for uid in a2_sorted[C.COL_USER_ID]], a2_sorted["潜力评分"], color=bar_colors)
        ax.set_xlabel("潜力评分（0~100）")
        ax.set_ylabel("用户编号")
        ax.set_title("附件2 50 位待判用户潜力评分")
        ax.tick_params(axis="y", labelsize=7)
        style_axes(ax, grid=False)
        fig.tight_layout()
        figures.append(self._figure(fig, "图4-4_附件2潜力评分排序.png"))

        metrics = {
            "入选模型": bundle.model_name,
            "测试集PR_AUC": float(getattr(calibrated, "pr_auc", np.nan)),
            "测试集ROC_AUC": float(getattr(calibrated, "roc_auc", np.nan)),
            "测试集Brier": float(brier),
            "F1阈值": float(suite.extra.get("threshold", np.nan)),
            "高潜力阈值": float(thresholds.get("high", np.nan)),
            "中潜力阈值": float(thresholds.get("medium", np.nan)),
            "附件1高潜力人数": int((scored_users["潜力等级"] == PotentialTier.HIGH.value).sum()),
            "附件1中潜力人数": int((scored_users["潜力等级"] == PotentialTier.MEDIUM.value).sum()),
            "附件2人数": int(len(attachment2_scored)),
            "聚类数k": int(k),
            "轮廓系数": float(silhouette),
        }
        notes = (
            f"问题4 入选模型 {bundle.model_name}，校准后 PR-AUC={metrics['测试集PR_AUC']:.4f}、"
            f"Brier={metrics['测试集Brier']:.4f}；高/中潜力阈值由训练集分数分位数确定，"
            f"附件2 共 {len(attachment2_scored)} 人（无真实标签，仅作模型推断）。"
        )
        self.logger.info("问题4完成：%d 张表、%d 张图", len(tables), len(figures))
        return QuestionResult(
            name="问题4：目标用户潜力评估与精准营销",
            tables=tables,
            figures=figures,
            metrics=metrics,
            notes=notes,
        )


__all__ = ["AnalysisServiceImpl"]
