# -*- coding: utf-8 -*-
"""程序入口：TUI 交互模式与命令行批处理模式。"""
from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from .config import AppConfig
from .controller.app_controller import TASK_NAMES, AppController


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="品牌手机目标用户的精准营销 —— 建模分析系统",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=None, help="配置文件路径（默认 代码/configs/default.yaml）")
    parser.add_argument(
        "--run",
        default=None,
        choices=list(TASK_NAMES),
        help="批处理模式：audit / q1 / q2 / q3 / q4 / all；不指定则进入 TUI",
    )
    parser.add_argument(
        "--force-features", action="store_true", help="强制重算用户级特征（默认使用 parquet 缓存）"
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = AppConfig.load(args.config)
    controller = AppController(config)

    if args.run is None:
        from .ui.tui import TUI

        TUI(controller).run()
        return 0

    if args.force_features:
        controller.ensure_features(force=True)
    results = controller.run(args.run)
    for question, result in results.items():
        print(f"[{question}] {result.name}")
        print("  表:", "、".join(result.tables) if result.tables else "无")
        print("  图:", "、".join(result.figures) if result.figures else "无")
        if result.metrics:
            print("  指标:", result.metrics)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
