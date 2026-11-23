# -*- coding: utf-8 -*-
"""绘图工具：统一 SCI/Nature 风格、中文字体、300 dpi 输出。

约束：每张图最多 2 个子图（1x2 或 2x1），多组图表拆成多张图片。
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 服务器/无界面环境安全
import matplotlib.pyplot as plt  # noqa: E402

__all__ = ["setup_plot_style", "save_figure", "style_axes"]

# 配色：主色 + 强调色 + 灰度
PALETTE = ["#2E5A88", "#D1495B", "#66A182", "#EDAE49", "#8D96A3", "#5B8FA8"]
HEATMAP_CMAP = "RdBu_r"


def setup_plot_style() -> None:
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
    plt.rcParams.update(
        {
            "font.sans-serif": ["SimHei", "Microsoft YaHei", "Noto Sans SC", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 1.2,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "lines.linewidth": 1.8,
            "grid.linestyle": "--",
            "grid.alpha": 0.5,
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def style_axes(ax, grid: bool = True, despine: bool = True) -> None:
    if grid:
        ax.grid(True, linestyle="--", alpha=0.45)
        ax.set_axisbelow(True)
    if despine:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)


def save_figure(fig, path: Path, dpi: int = 300) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path
