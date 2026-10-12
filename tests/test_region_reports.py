"""County reports must preserve uncertainty, geography, and model status."""

import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import train_region_risk as pipeline


def report_inputs():
    regions = [
        {"name": "Example", "geoid": "53001", "kind": "county", "bus_count": 2,
         "area_km2": 100., "modeled_bounds": [-119., 46., -118.9, 46.1],
         "modeled_bbox_area_km2": 1., "scope_note": "Only modeled feeder loads.",
         "geometry": {"type": "Polygon", "coordinates": [
             [[-119., 46.], [-118.9, 46.], [-118.9, 46.1], [-119., 46.1], [-119., 46.]]]}},
        {"name": "Unknown", "geoid": None, "kind": "unknown", "bus_count": 1,
         "area_km2": None, "modeled_bounds": None, "modeled_bbox_area_km2": None,
         "scope_note": "County assignment unavailable.", "geometry": None},
        {"name": "Empty", "geoid": "53003", "kind": "county", "bus_count": 1,
         "area_km2": 200., "modeled_bounds": None, "modeled_bbox_area_km2": None,
         "scope_note": "No modeled positive load.", "geometry": None},
    ]
    graphs = []
    for i in range(2):
        y = torch.tensor([[float(i), 1., 1.]])
        extent = torch.tensor([[i * 0.6, 1., 0.5 + i * 0.4]])
        graphs.append(Data(y=y, extent=extent, dns_fraction=extent,
                           voltage_fraction=extent, overload_fraction=y,
                           node_affected=torch.tensor([bool(i), False, True, True]),
                           scenario_id=torch.tensor([i]), outage_group=f"line.{i}"))
    scenarios = [{"scenario_id": i, "branch_name": f"Line.{i}", "load_multiplier": 1.,
                  "converged": i < 2, "status": "valid" if i < 2 else "outage_nonconverged"}
                 for i in range(4)]
    metadata = {
        "attempted_scenarios": 4, "invalid_scenarios": 2, "eligible_candidate_count": 20,
        "region_valid": [True, True, False], "region_nominal_load_kw": [100., 10., 0.],
        "node_nominal_load_kw": [50., 50., 10., 0.],
        "bus_names": ["load_a", "load_b", "unknown", "unloaded"],
        "node_region": [0, 0, 1, 2], "scenarios": scenarios,
        "sampling": "uniform_without_replacement", "problem_type": "any",
        "probability_scope": "Conditional on a sampled single-line outage.",
        "definitions": {"extent": "Affected nominal-load share."},
        "thresholds": {"dns_fraction": 0.01, "affected_load_fraction": 0.01},
        "baseline": {"y": [1., 0., 0.], "extent": [0.2, 0., 0.]},
        "model_skipped_reason": "GNN training skipped: fewer than 12 distinct resolved outage lines.",
    }
    geography = {"regions": regions, "coverage": {"county_count": 2, "unassigned_buses": 1}}
    coordinates = {"load_a": (-119., 46.), "load_b": (-118.9, 46.1),
                   "unknown": None, "unloaded": (-119., 46.1)}
    return graphs, metadata, geography, coordinates


class RegionReportTests(unittest.TestCase):
    def report(self, output, model_output=None, inputs=None):
        graphs, metadata, geography, coordinates = inputs or report_inputs()
        with patch.object(pipeline, "plot_risk_map") as plotting:
            report = pipeline.make_reports(graphs, metadata, geography, coordinates,
                                           Path(output), model_output=model_output)
        plotting.assert_called_once()
        return report

    def test_unresolved_scenarios_create_bounds_and_remain_in_export(self):
        with tempfile.TemporaryDirectory() as output:
            report = self.report(output)
            county = report["counties"][0]
            self.assertIsNone(county["simulation_probability"])
            self.assertEqual(county["simulation_frequency_resolved"], 0.5)
            self.assertEqual(county["risk_probability_lower_bound"], 0.25)
            self.assertEqual(county["risk_probability_upper_bound"], 0.75)
            for threshold in [1, 5, 10, 25, 50]:
                label = f"affected_load_ge_{threshold}pct"
                self.assertIsNone(county[f"{label}_probability"])
                self.assertEqual(county[f"{label}_lower_bound"], 0.25)
                self.assertEqual(county[f"{label}_upper_bound"], 0.75)
            self.assertAlmostEqual(county["expected_affected_load_fraction_resolved"], 0.3)
            self.assertIn("not confidence intervals", report["probability_bounds_scope"])
            with (Path(output) / "scenario_predictions.csv").open(encoding="utf-8-sig") as stream:
                scenarios = list(csv.DictReader(stream))
            self.assertEqual(len(scenarios), 12)
            unresolved = [row for row in scenarios if int(row["scenario_id"]) >= 2]
            self.assertEqual(len(unresolved), 6)
            self.assertTrue(all(row["status"] == "outage_nonconverged" for row in unresolved))
            self.assertTrue(all(row["simulated_problem"] == "" for row in unresolved))
            saved = json.loads((Path(output) / "risk_report.json").read_text())
            self.assertEqual(saved, report)

    def test_unknown_county_and_unloaded_county_do_not_get_county_risk(self):
        with tempfile.TemporaryDirectory() as output:
            report = self.report(output)
            for county in report["counties"][1:]:
                for key in ["simulation_probability", "simulation_frequency_resolved",
                            "risk_probability_lower_bound", "risk_probability_upper_bound"]:
                    self.assertIsNone(county[key])
            self.assertFalse(report["counties"][1]["is_county"])
            self.assertTrue(report["counties"][2]["is_county"])
            with (Path(output) / "bus_problem_probability.csv").open(encoding="utf-8-sig") as stream:
                buses = {row["bus"]: row for row in csv.DictReader(stream)}
            self.assertEqual(float(buses["load_a"]["problem_frequency_resolved"]), 0.5)
            self.assertEqual(float(buses["load_b"]["problem_frequency_resolved"]), 0.)
            self.assertEqual(buses["unloaded"]["problem_frequency_resolved"], "")
            geojson = json.loads((Path(output) / "county_risk.geojson").read_text())
            self.assertEqual([f["properties"]["county"] for f in geojson["features"]], ["Example"])
            self.assertIsNone(geojson["features"][0]["properties"]["simulation_probability"])

    def test_baseline_problem_and_training_skip_reason_are_visible(self):
        inputs = report_inputs()
        with tempfile.TemporaryDirectory() as output:
            report = self.report(output, inputs=inputs)
        county = report["counties"][0]
        self.assertTrue(county["baseline_already_has_problem"])
        self.assertEqual(county["baseline_affected_load_fraction"], 0.2)
        self.assertEqual(report["baseline"], inputs[1]["baseline"])
        self.assertIn(inputs[1]["model_skipped_reason"], report["warnings"])
        self.assertTrue(any("pre-existing problems" in warning for warning in report["warnings"]))
        self.assertIsNone(report["model_report"])

    def test_calibration_status_distinguishes_probability_from_uncalibrated_score(self):
        model_output = {
            "probabilities": np.array([[0.2, 0.7, 0.8], [0.8, 0.9, 0.8]]),
            "extent": np.array([[0.1, 0.2, 0.3], [0.5, 0.2, 0.3]]),
            "calibration": [{"status": "calibrated"}] * 3,
            "test_metrics": [{"brier_score": 0.1}, {"samples": 0}, {"samples": 0}],
            "split_names": ["train", "test"],
            "report": {"summary_scope": "Includes training scenarios."},
        }
        with tempfile.TemporaryDirectory() as output:
            calibrated = self.report(output, model_output=model_output)
        county = calibrated["counties"][0]
        self.assertEqual(county["gnn_probability_resolved"], 0.5)
        self.assertIsNone(county["gnn_uncalibrated_score"])
        self.assertIsNone(county["simulation_probability"])
        for row in calibrated["counties"][1:]:
            self.assertIsNone(row["gnn_probability_resolved"])
            self.assertIsNone(row["gnn_uncalibrated_score"])
        uncalibrated = copy.deepcopy(model_output)
        uncalibrated["calibration"][0] = {"status": "insufficient_labels"}
        with tempfile.TemporaryDirectory() as output:
            report = self.report(output, model_output=uncalibrated)
        self.assertIsNone(report["counties"][0]["gnn_probability_resolved"])
        self.assertEqual(report["counties"][0]["gnn_uncalibrated_score"], 0.5)
        self.assertTrue(any("uncalibrated score" in warning for warning in report["warnings"]))

    def test_fully_resolved_ensemble_exposes_conditional_probability(self):
        inputs = report_inputs()
        inputs[1]["attempted_scenarios"] = 2
        inputs[1]["invalid_scenarios"] = 0
        inputs[1]["scenarios"] = inputs[1]["scenarios"][:2]
        with tempfile.TemporaryDirectory() as output:
            report = self.report(output, inputs=inputs)
        county = report["counties"][0]
        self.assertEqual(county["simulation_probability"], 0.5)
        self.assertEqual(county["risk_probability_lower_bound"], 0.5)
        self.assertEqual(county["risk_probability_upper_bound"], 0.5)
        self.assertEqual(county["affected_load_ge_50pct_probability"], 0.5)

    def test_explicit_bus_county_csv_does_not_add_default_boundary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            master = Path(directory) / "master.dss"
            county_csv = Path(directory) / "county.csv"
            master.write_text("clear\n")
            county_csv.write_text("bus,county\nb,Example\n")
            args = pipeline.parse_args(["--dss-master-file", str(master),
                                        "--bus-county-csv", str(county_csv)])
        self.assertEqual(args.bus_county_csv, county_csv)
        self.assertIsNone(args.county_geojson)
        self.assertIsNone(args.bus_coordinates)


if __name__ == "__main__":
    unittest.main()
