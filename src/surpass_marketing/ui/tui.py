# -*- coding: utf-8 -*-
"""Rich 文本界面（TUI）：仅负责交互与展示，不承载任何建模逻辑。"""
from __future__ import annotations

import traceback
from typing import List, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from ..controller.app_controller import AppController

MENU_ITEMS: List[Tuple[str, str]] = [
    ("0", "数据审计与质量报告（8 张表、重复、覆盖率、标签分布）"),
    ("1", "问题1：数据预处理与描述性统计"),
    ("2", "问题2：基本行为特征对购买行为的影响"),
    ("3", "问题3：个人偏好的影响（偏好指标、PCA、用户画像）"),
    ("4", "问题4：附件2 潜力评估、未购 Top100 与营销策略"),
    ("5", "一键完整运行（数据审计 + 问题1~4）"),
    ("6", "查看输出文件与最近日志"),
    ("q", "退出"),
]


class TUI:
    """基于 Rich 的编号菜单界面。"""

    def __init__(self, controller: AppController):
        self.controller = controller
        self.console = Console()

    # ------------------------------------------------------------------ #
    # 界面
    # ------------------------------------------------------------------ #
    def _render_menu(self) -> None:
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("按键", style="bold cyan", width=6)
        table.add_column("功能")
        for key, label in MENU_ITEMS:
            table.add_row(key, label)
        self.console.print(
            Panel(
                table,
                title="[bold]品牌手机目标用户的精准营销 —— 建模系统[/bold]",
                subtitle="Controller → Service → ServiceImpl",
                border_style="cyan",
            )
        )

    def _render_result(self, result) -> None:
        self.console.print(f"[bold green]✓ {result.name}[/bold green]")
        if result.tables:
            self.console.print("  数据表：" + "、".join(result.tables))
        if result.figures:
            self.console.print("  图    ：" + "、".join(result.figures))
        if result.metrics:
            self.console.print("  指   标：")
            for key, value in result.metrics.items():
                text = f"{value:.4f}" if isinstance(value, float) else str(value)
                self.console.print(f"    · {key}: {text}")
        if result.notes:
            self.console.print(f"  [dim]{result.notes}[/dim]")

    def _show_outputs(self) -> None:
        frame = self.controller.list_outputs()
        if frame.empty:
            self.console.print("[yellow]暂无输出文件。[/yellow]")
            return
        table = Table(title="输出文件")
        table.add_column("目录")
        table.add_column("文件")
        table.add_column("大小(KB)", justify="right")
        for _, row in frame.iterrows():
            table.add_row(str(row["目录"]), str(row["文件"]), f"{row['大小(KB)']:.1f}")
        self.console.print(table)
        self.console.print(Panel(self.controller.latest_logs(limit=2), title="最近日志", border_style="dim"))

    # ------------------------------------------------------------------ #
    # 主循环
    # ------------------------------------------------------------------ #
    def run(self) -> None:
        self.console.print("[bold]欢迎使用 Surpass 目标用户精准营销建模系统[/bold]\n")
        while True:
            try:
                self._render_menu()
                choice = Prompt.ask(
                    "请选择功能", choices=[key for key, _ in MENU_ITEMS], default="q"
                )
                if choice == "q":
                    self.console.print("[cyan]再见！[/cyan]")
                    return
                if choice == "6":
                    self._show_outputs()
                    continue
                task = {"0": "audit", "1": "q1", "2": "q2", "3": "q3", "4": "q4", "5": "all"}[choice]
                with self.console.status(f"[cyan]正在执行 {task}，请稍候（首次运行需读取 117MB 原始 Excel）..."):
                    results = self.controller.run(task)
                for result in results.values():
                    self._render_result(result)
                if results:
                    self.console.print(
                        "[green]结果已写入 代码/outputs/{tables,figures,models,logs}[/green]"
                    )
            except KeyboardInterrupt:
                self.console.print("\n[yellow]已中断，返回菜单。[/yellow]")
            except Exception as error:  # noqa: BLE001 - TUI 需要兜底展示
                self.console.print(f"[bold red]执行失败：{error}[/bold red]")
                self.controller.logger.error("TUI 执行失败：%s\n%s", error, traceback.format_exc())


__all__ = ["TUI"]
