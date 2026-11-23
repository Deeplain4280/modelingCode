# -*- coding: utf-8 -*-
"""配置与业务阈值测试。"""
from __future__ import annotations

import pytest

from surpass_marketing.config import AppConfig


def test_config_defaults(config):
    assert config.runtime.seed == 42
    assert config.runtime.test_size == pytest.approx(0.2)
    assert config.runtime.cv_folds == 5
    assert config.runtime.top_n == 100
    assert config.business.attachment2_expected_users == 50
    assert config.business.break_even_probability == pytest.approx(
        config.business.contact_cost / config.business.conversion_value
    )


def test_config_paths_point_to_workspace(config):
    assert config.paths.attachment1.name == "附件1 Surpass目标用户数据1.xlsx"
    assert config.paths.attachment2.name == "附件2 待判数据.xlsx"
    assert config.paths.raw_dir.name == "B题附件"
    assert config.paths.code_root.name == "code"


def test_config_file_exists(config):
    assert config.paths.config_file.exists()
