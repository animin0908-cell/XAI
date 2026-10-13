"""Validate audience sentences against their numeric SHAP evidence."""

import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from explainer.stakeholder_output import render_stakeholders, stakeholder_text


def explanation(kind, value, features):
    unit = "score" if kind == "risk" else "load_fraction"
    name = "Adams County 的未校准问题风险分数" if kind == "risk" else "Adams County 的受影响负荷比例"
    return {"dataset": "ieee9500", "model": "gin", "task": "county_risk", "sample_index": 25,
            "scenario_id": 28, "partition": "test", "outage_line": "line.example",
            "target": {"key": kind + "_0", "name": name, "kind": kind, "unit": unit,
                       "region": "Adams County", "status": "insufficient_labels" if kind == "risk" else "model_prediction"},
            "prediction": value, "base_value": value - sum(f["shap_value"] for f in features),
            "shap_sum": sum(f["shap_value"] for f in features), "features": features}


def feature(key, name, value):
    return {"key": key, "name": name, "unit": "dataset_native", "shap_value": value, "feature_mean": 0.99}


class StakeholderTests(unittest.TestCase):
    def results(self):
        return [explanation("risk", 0.999887, [feature("load_q", "无功负荷", -0.0000414)]),
                explanation("extent", 0.5043786, [feature("voltage", "节点停运前电压", 0.0121302),
                                                 feature("log_rating", "支路额定值特征", -0.00518426)])]

    def test_admin_and_manager_get_one_group_with_real_features_and_units(self):
        results = self.results()
        saved = copy.deepcopy(results)
        report = render_stakeholders(results)
        self.assertEqual(len(report["groups"]), 1)
        roles = report["groups"][0]["audiences"]
        self.assertEqual(list(roles), ["admin", "asset_manager"])
        self.assertIn("0.999887", roles["admin"]["sentence"])
        self.assertIn("不能按停电概率解读", roles["admin"]["sentence"])
        manager = roles["asset_manager"]["sentence"]
        for expected in ("50.44%", "voltage", "log_rating", "提高约 1.21 个百分点", "降低约 0.518 个百分点"):
            self.assertIn(expected, manager)
        self.assertNotIn("99.99%", manager)
        self.assertNotIn("$", manager)
        self.assertIn("未提供停运成本", manager)
        self.assertEqual(results, saved)
        self.assertEqual(roles["asset_manager"]["evidence"][1]["features"][0]["shap_value"], 0.0121302)
        text = stakeholder_text(report)
        self.assertIn("For Admin:", text)
        self.assertIn("For Asset Manager:", text)
        self.assertIn("假设 line.example 停运", text)

    def test_optional_roles_have_distinct_details(self):
        report = render_stakeholders(self.results(), ["engineer", "executive", "domain_expert"])
        roles = report["groups"][0]["audiences"]
        self.assertIn("整列均值", roles["engineer"]["sentence"])
        self.assertIn("投资收益", roles["executive"]["sentence"])
        self.assertIn("因果效果", roles["domain_expert"]["sentence"])
        self.assertIn("未纳入", roles["domain_expert"]["sentence"])

    def test_ieee24_probability_uses_selected_class_without_claiming_predicted_class(self):
        result = explanation("risk", 0.3, [feature("net_active_power", "净有功功率", -0.02)])
        result.update(dataset="ieee24", task="multi_classification", outage_line=None)
        result["target"] = {"key": "class_0", "name": "A 类：失供电且发生级联故障", "kind": "class",
                            "unit": "probability", "status": "uncalibrated", "region": ""}
        report = render_stakeholders([result])
        sentence = report["groups"][0]["audiences"]["admin"]["sentence"]
        self.assertIn("30.00%（未校准）", sentence)
        self.assertIn("降低约 2 个百分点", sentence)
        self.assertNotIn("预测为 A", sentence)

    def test_counties_and_samples_do_not_merge(self):
        first = self.results()[0]
        second = copy.deepcopy(first)
        second["sample_index"] = 26
        third = copy.deepcopy(first)
        third["target"]["region"] = "Franklin County"
        self.assertEqual(len(render_stakeholders([first, second, third])["groups"]), 3)

    def test_invalid_roles_and_nonfinite_contributions_are_rejected(self):
        for roles in ([], ["unknown"], ["admin", "admin"]):
            with self.subTest(roles=roles), self.assertRaises(ValueError):
                render_stakeholders(self.results(), roles)
        result = self.results()[0]
        result["features"][0]["shap_value"] = float("nan")
        with self.assertRaises(ValueError):
            render_stakeholders([result])


if __name__ == "__main__":
    unittest.main()
