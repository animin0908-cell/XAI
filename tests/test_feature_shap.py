"""Check shared SHAP math, masking, model routing and checkpoint compatibility."""

import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.nn import global_mean_pool

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

entry_spec = importlib.util.spec_from_file_location("prediction_main", ROOT / "main.py")
entry = importlib.util.module_from_spec(entry_spec)
entry_spec.loader.exec_module(entry)
main, parse_args, select_targets = entry.main, entry.parse_args, entry.select_targets
from explainer.feature_shap import explain_features, masked_predictions, write_explanations
from gnn.regional_risk import RegionalRiskGNN
from workflows.common import (
    ModelContext, Target, IEEE24_NODE_FEATURES, IEEE24_EDGE_FEATURES,
)
from workflows.ieee9500 import load_context as load_regional_context
from workflows.ieee24 import load_context as load_graph_context


def graph(value=1.0):
    return Data(x=torch.full((4, 3), value),
                edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]),
                edge_attr=torch.full((4, 4), value), y=torch.tensor([0.0]))


class AdditiveModel(nn.Module):
    def forward(self, batch):
        node = global_mean_pool(batch.x, batch.batch)
        edge = global_mean_pool(batch.edge_attr, batch.batch[batch.edge_index[0]])
        features = torch.cat((node, edge), dim=1)
        value = (features * torch.arange(1, 8)).sum(dim=1)
        return torch.stack((value, -2 * value), dim=1)


def additive_context():
    model = AdditiveModel()
    return ModelContext("ieee24", "toy", "regression", model,
                        [graph(0.0), graph(1.0)], {"train": [0], "test": [1]},
                        Path("toy.pth"), IEEE24_NODE_FEATURES, IEEE24_EDGE_FEATURES,
                        (Target("a", "输出 A", "dns", "MW"), Target("b", "输出 B", "dns", "MW")),
                        model, torch.device("cpu"))


class FeatureShapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_kernel_shap_matches_known_multi_output_shapley_values(self):
        context = additive_context()
        results = explain_features(context, 1, [0, 1], context.reference())
        for result, factor in zip(results, (1, -2)):
            by_key = {row["key"]: row for row in result["features"]}
            for i, feature in enumerate(context.node_features + context.edge_features, start=1):
                self.assertAlmostEqual(by_key[feature.key]["shap_value"], factor * i, places=5)
                self.assertIn(feature.name, by_key[feature.key]["sentence"])
            self.assertAlmostEqual(result["prediction"], factor * 28, places=5)
            self.assertAlmostEqual(result["additivity_error"], 0, places=5)

    def test_masks_keep_topology_and_do_not_read_labels_or_mutate_sample(self):
        context = additive_context()
        sample = context.graphs[1]
        saved = sample.clone()
        sample.y = torch.tensor([float("nan")])
        values = masked_predictions(context, sample, np.array([[0.] * 7, [1.] * 7]),
                                    context.reference(), [0], batch_size=1)
        np.testing.assert_allclose(values[:, 0], [0, 28])
        torch.testing.assert_close(sample.x, saved.x)
        torch.testing.assert_close(sample.edge_attr, saved.edge_attr)
        torch.testing.assert_close(sample.edge_index, saved.edge_index)

    def test_reference_excludes_test_data(self):
        context = additive_context()
        before = context.reference()
        context.graphs[1].x *= 1000
        context.graphs[1].edge_attr *= 1000
        after = context.reference()
        for a, b in zip(before, after):
            torch.testing.assert_close(a, b)

    def test_shared_report_contains_sentences_units_and_additivity(self):
        context = additive_context()
        reference = context.reference()
        results = explain_features(context, 1, [1], reference)
        with tempfile.TemporaryDirectory() as directory:
            write_explanations(context, results, reference, directory, 256, 0)
            report = json.loads((Path(directory) / "shap_report.json").read_text())
            self.assertEqual(report["method"], "KernelSHAP")
            with (Path(directory) / "shap_features.csv").open(encoding="utf-8-sig") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 7)
            self.assertTrue(all(row["output_unit"] == "MW" for row in rows))
            self.assertIn("推动输出下降", (Path(directory) / "feature_explanations.txt").read_text())
            sentences = (Path(directory) / "explanations.txt").read_text()
            self.assertIn("For Admin:", sentences)
            self.assertIn("For Asset Manager:", sentences)
            self.assertTrue((Path(directory) / "stakeholder_explanations.json").is_file())

    def test_ieee24_trains_and_reloads_all_tasks_with_shared_shap_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for task, name in (("multi_classification", "transformer"),
                               ("binary_classification", "gat"), ("regression", "gin")):
                with self.subTest(task=task):
                    cache = root / "data/ieee24/processed" / task
                    cache.mkdir(parents=True)
                    graphs = [graph(0.1 + i / 20) for i in range(20)]
                    for i, sample in enumerate(graphs):
                        sample.y = torch.tensor([float(i % (4 if task == "multi_classification" else 2))])
                    torch.save(InMemoryDataset.collate(graphs), cache / "data.pt")
                    output = root / task
                    args = parse_args(["--dataset", "ieee24", "--model", name, "--task", task,
                                       "--data_save_dir", str(root / "data"), "--model_save_dir", str(root / "model"),
                                       "--output_dir", str(output), "--num_epochs", "1", "--threads", "1"])
                    main(args)
                    self.assertTrue((output / "shap_features.csv").is_file())
                    self.assertTrue((output / "explanations.txt").is_file())
                    args.predict_only = True
                    args.skip_shap = True
                    context = main(args)
                    self.assertEqual(context.model_name, name)
                    self.assertEqual(len(context.targets), 4 if task == "multi_classification" else 2 if task == "binary_classification" else 1)

    def regional_fixture(self, root):
        torch.manual_seed(5)
        config = dict(input_dim=4, edge_dim=4, num_regions=2, model_name="gin",
                      hidden_dim=8, num_layers=2, dropout=0.)
        model = RegionalRiskGNN(**config).eval()
        graphs = []
        for i in range(4):
            sample = graph(1.0 + i / 10)
            sample.x = torch.cat((sample.x, torch.tensor([[1.], [0.], [0.], [1.]])), dim=1)
            sample.region_id = torch.tensor([0, 0, 1, 1])
            sample.y = torch.tensor([[0., 1.]])
            sample.extent = torch.tensor([[0.1, 0.2]])
            sample.region_valid = torch.tensor([[True, True]])
            sample.scenario_id = torch.tensor([i])
            graphs.append(sample)
        mean = torch.tensor([0.1, 0.2, 0.3, 0.])
        scale = torch.tensor([2., 3., 4., 1.])
        calibration = [{"status": "insufficient_labels", "slope": 1., "intercept": 0.},
                       {"status": "calibrated", "slope": 0.5, "intercept": -0.1}]
        checkpoint = root / "regional_gnn_best.pt"
        torch.save({"config": config, "state_dict": model.state_dict(), "normalization_mean": mean,
                    "normalization_scale": scale, "calibration": calibration, "county_names": ["A County", "B County"],
                    "dataset_cache_key": "fixture", "splits": {"train": [0], "tune": [1], "calibration": [2], "test": [3]}}, checkpoint)
        cache = root / "dataset/fixture"
        cache.mkdir(parents=True)
        torch.save(graphs, cache / "graphs.pt")
        (cache / "metadata.json").write_text(json.dumps({"cache_key": "fixture", "region_valid": [True, True],
            "regions": [{"name": "A County", "kind": "county"}, {"name": "B County", "kind": "county"}]}))
        return checkpoint, model, graphs, mean, scale

    def test_ieee9500_reuses_weights_normalization_calibration_and_shared_shap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, model, graphs, mean, scale = self.regional_fixture(root)
            args = parse_args(["--dataset", "ieee9500", "--model", "gin", "--checkpoint", str(checkpoint),
                               "--target", "both", "--output_dir", str(root / "output"), "--threads", "1"])
            context = load_regional_context(args, torch.device("cpu"))
            torch.testing.assert_close(context.graphs[0].x, (graphs[0].x - mean) / scale)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(context.model.state_dict()[key], value)
            self.assertEqual(context.targets[0].unit, "score")
            self.assertEqual(context.targets[1].unit, "probability")
            self.assertEqual(context.reference()[0][3].item(), 0.0)
            report = main(args)
            self.assertEqual(len(report["explanations"]), 4)
            self.assertTrue(all(abs(r["additivity_error"]) < 1e-5 for r in report["explanations"]))
            args.county = "B County"
            self.assertEqual(select_targets(context, context.predict([context.graphs[3]])[0], args), [1, 3])

    def test_ieee9500_rejects_different_model_or_graph_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, *_ = self.regional_fixture(root)
            args = parse_args(["--dataset", "ieee9500", "--model", "transformer", "--checkpoint", str(checkpoint)])
            with self.assertRaisesRegex(ValueError, "differs from checkpoint"):
                load_regional_context(args, torch.device("cpu"))
            args.model_name = "gin"
            path = root / "dataset/fixture/metadata.json"
            metadata = json.loads(path.read_text())
            metadata["cache_key"] = "other"
            path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "different regions or cache keys"):
                load_regional_context(args, torch.device("cpu"))

    def test_datasets_reject_each_others_checkpoint_formats(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            regional, *_ = self.regional_fixture(root)
            graph_checkpoint = root / "graph.pth"
            torch.save({"net": {}}, graph_checkpoint)
            args = parse_args(["--dataset", "ieee9500", "--checkpoint", str(graph_checkpoint)])
            with self.assertRaisesRegex(ValueError, "IEEE9500 county-risk workflow"):
                load_regional_context(args, torch.device("cpu"))
            cache = root / "data/ieee24/processed/multi_classification"
            cache.mkdir(parents=True)
            torch.save(InMemoryDataset.collate([graph() for _ in range(20)]), cache / "data.pt")
            args = parse_args(["--dataset", "ieee24", "--data_save_dir", str(root / "data"),
                               "--checkpoint", str(regional)])
            with self.assertRaisesRegex(ValueError, "IEEE24 graph-level workflow"):
                load_graph_context(args, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
