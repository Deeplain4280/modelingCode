# -*- coding: utf-8 -*-
"""产物仓库测试：表格、文本、模型、输出清单。"""
from __future__ import annotations

import pandas as pd

from surpass_marketing.domain.entities import ModelBundle


def test_save_and_read_table(artifacts):
    frame = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    path = artifacts.save_table(frame, "测试表.csv")
    assert path.exists()
    loaded = pd.read_csv(path, encoding="utf-8-sig")
    assert loaded.shape == (2, 2)


def test_save_text_and_json(artifacts):
    text_path = artifacts.save_text("你好", "说明.txt")
    assert text_path.read_text(encoding="utf-8") == "你好"
    json_path = artifacts.save_json({"score": 0.5}, "指标.json")
    assert "0.5" in json_path.read_text(encoding="utf-8")


def test_save_and_load_model(artifacts):
    bundle = ModelBundle(
        model_name="dummy",
        pipeline=None,
        feature_names=["f1"],
        metrics=None,
        tier_thresholds={"high": 0.5},
        positive_rate=0.1,
        trained_at="2026-01-01",
        calibration_method="sigmoid",
        notes="",
    )
    path = artifacts.save_model(bundle, "dummy.joblib")
    assert path.exists()
    loaded = artifacts.load_model("dummy.joblib")
    assert loaded.model_name == "dummy"
    assert path.with_suffix(".json").exists()


def test_list_outputs_returns_existing_files(artifacts):
    artifacts.save_table(pd.DataFrame({"a": [1]}), "表.csv")
    files = artifacts.list_outputs()
    assert any(path.name == "表.csv" for path in files)
