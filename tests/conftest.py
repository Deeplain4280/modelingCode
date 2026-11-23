# -*- coding: utf-8 -*-
"""pytest 公共夹具：路径注入、临时输出目录与轻量日志。"""
from __future__ import annotations

import dataclasses
import logging
import sys
from pathlib import Path

import pytest

CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = CODE_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from surpass_marketing.config import AppConfig  # noqa: E402


@pytest.fixture()
def logger() -> logging.Logger:
    instance = logging.getLogger("surpass-test")
    instance.handlers = [logging.NullHandler()]
    instance.setLevel(logging.INFO)
    return instance


@pytest.fixture()
def config(logger) -> AppConfig:
    """输出目录放在工作区内（沙箱下系统临时目录不可枚举）。"""
    import shutil
    import uuid

    base = AppConfig.load()
    tmp_path = CODE_ROOT / "outputs" / "_test_tmp" / uuid.uuid4().hex[:8]
    tmp_path.mkdir(parents=True, exist_ok=True)
    dirs = {}
    for name in ("interim_dir", "table_dir", "figure_dir", "model_dir", "log_dir"):
        dirs[name] = tmp_path / name
        dirs[name].mkdir(parents=True, exist_ok=True)
    paths = dataclasses.replace(base.paths, **dirs)
    yield dataclasses.replace(base, paths=paths)
    shutil.rmtree(tmp_path, ignore_errors=True)


@pytest.fixture()
def artifacts(config, logger):
    from surpass_marketing.repository.artifact_repository import ArtifactRepository

    return ArtifactRepository(config, logger)
