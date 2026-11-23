# -*- coding: utf-8 -*-
"""领域枚举：状态、潜力分层、任务名。"""
from __future__ import annotations

from enum import Enum


class PurchaseState(str, Enum):
    """目标用户最后一次跟踪状态。"""

    BROWSE = "浏览"
    SEARCH = "搜索"
    PURCHASE = "购买"


class PotentialTier(str, Enum):
    """潜力分层。"""

    HIGH = "高潜力"
    MEDIUM = "中潜力"
    LOW = "低潜力"


class TaskName(str, Enum):
    """可执行的建模任务。"""

    AUDIT = "audit"
    Q1 = "q1"
    Q2 = "q2"
    Q3 = "q3"
    Q4 = "q4"
    ALL = "all"


__all__ = ["PurchaseState", "PotentialTier", "TaskName"]
