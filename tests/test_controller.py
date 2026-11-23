# -*- coding: utf-8 -*-
"""Controller 层测试：只做用例转发，不承载建模逻辑。"""
from __future__ import annotations

import pandas as pd

from surpass_marketing.controller.data_controller import AnalysisController, DataController
from surpass_marketing.domain.entities import QuestionResult


class FakeDataService:
    def __init__(self):
        self.audited = 0
        self.report_saved = 0
        self.feature_calls = []
        self.attachment_calls = 0

    def audit(self):
        self.audited += 1
        return {"audit": True}

    def save_data_quality_report(self, audit):
        self.report_saved += 1
        return pd.DataFrame()

    def build_user_features(self, force=False):
        self.feature_calls.append(force)
        return pd.DataFrame({"用户编号": [1, 2, 3], "target": [0, 1, 0]})

    def build_attachment2_features(self):
        self.attachment_calls += 1
        return pd.DataFrame({"用户编号": [9, 10], "is_attachment2": [1, 1]})


class FakeAnalysisService:
    def __init__(self):
        self.calls = []
        self.q4_args = None

    def _result(self, name):
        self.calls.append(name)
        return QuestionResult(name=name, tables=[], figures=[], metrics={}, notes="")

    def run_question1(self, users):
        return self._result("q1")

    def run_question2(self, users):
        return self._result("q2")

    def run_question3(self, users):
        return self._result("q3")

    def run_question4(self, users, attachment2_features):
        self.q4_args = (users, attachment2_features)
        return self._result("q4")


def test_data_controller_audit_saves_report(logger):
    fake = FakeDataService()
    result = DataController(fake, logger).audit()
    assert result == {"audit": True}
    assert fake.audited == 1
    assert fake.report_saved == 1


def test_analysis_controller_forwarding(logger):
    data, analysis = FakeDataService(), FakeAnalysisService()
    controller = AnalysisController(data, analysis, logger)
    for question in ("q1", "q2", "q3"):
        controller.run(question)
    assert analysis.calls == ["q1", "q2", "q3"]


def test_analysis_controller_q4_builds_attachment2(logger):
    data, analysis = FakeDataService(), FakeAnalysisService()
    controller = AnalysisController(data, analysis, logger)
    result = controller.run("q4")
    assert result.name == "q4"
    assert data.attachment_calls == 1
    users, attachment2 = analysis.q4_args
    assert len(attachment2) == 2
    assert len(users) == 3


def test_analysis_controller_rejects_unknown(logger):
    import pytest

    controller = AnalysisController(FakeDataService(), FakeAnalysisService(), logger)
    with pytest.raises(ValueError):
        controller.run("q9")
