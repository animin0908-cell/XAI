"""Numerical and data-isolation checks for county GNN risk predictions."""

from pathlib import Path
import sys
import unittest

import torch
from torch import nn
from torch_geometric.data import Batch, Data

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gnn.regional_risk import (
    RegionalRiskGNN, fit_calibration, grouped_split, normalize_graphs,
    predict, probabilities, probability_metrics, train_regional_model,
)


def sample_graph(number=0, num_regions=4):
    generator = torch.Generator().manual_seed(number)
    x = torch.rand((6, 4), generator=generator)
    x[:, 3] = torch.tensor([1., 0., 0., 0., 0., 1.])
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5],
                               [1, 0, 2, 1, 3, 2, 4, 3, 5, 4]])
    region_ids = torch.tensor([0, 0, 1, 1, 2, 2])
    y = torch.tensor([[float(number % 2), 0., 1., 0.]])
    return Data(x=x, edge_index=edge_index,
                edge_attr=torch.rand((10, 6), generator=generator),
                region_id=region_ids,
                region_valid=torch.tensor([[True, True, True, False]]),
                y=y, extent=y * 0.6, node_affected=torch.zeros(6, dtype=torch.bool),
                dns_fraction=y * 0.4, voltage_fraction=y * 0.3,
                overload_fraction=y * 0.2, outage_group=f"line.outage_{number}")


class _IdentityEmbeddings(nn.Module):
    def get_emb(self, data):
        return data.x[:, :3]


class _CaptureReadout(nn.Module):
    def forward(self, features):
        self.features = features.detach().clone()
        return features[:, :2]


class RegionalRiskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def model(self, name="gin"):
        torch.manual_seed(10)
        return RegionalRiskGNN(4, 6, 4, model_name=name, hidden_dim=8,
                               num_layers=2, dropout=0.).eval()

    def test_county_pooling_is_separate_for_each_graph(self):
        first, second = sample_graph(0), sample_graph(1)
        second.x[:, :3] += 100
        model = self.model()
        model.backbone = _IdentityEmbeddings()
        model.head = _CaptureReadout()
        batch = Batch.from_data_list([first, second])
        model(batch)
        actual = model.head.features
        expected = []
        for graph in [first, second]:
            for region in range(4):
                members = graph.x[graph.region_id == region]
                if len(members):
                    mean, maximum = members[:, :3].mean(0), members[:, :3].max(0).values
                    inputs = members.mean(0)
                    inputs[3:] = members[:, 3:].sum(0)
                else:
                    mean, maximum, inputs = torch.zeros(3), torch.zeros(3), torch.zeros(4)
                expected.append(torch.cat((mean, maximum, graph.x[:, :3].mean(0), inputs)))
        torch.testing.assert_close(actual, torch.stack(expected))
        self.assertEqual(batch.region_id.tolist(), first.region_id.tolist() * 2)

    def test_all_backbones_batch_independently_and_return_finite_probabilities(self):
        graphs = [sample_graph(2), sample_graph(3)]
        for name in ["gin", "gat", "transformer"]:
            with self.subTest(model=name), torch.no_grad():
                model = self.model(name)
                together = model(Batch.from_data_list(graphs))
                separately = [model(Batch.from_data_list([g])) for g in graphs]
                for output in range(2):
                    torch.testing.assert_close(together[output], torch.cat([v[output] for v in separately]),
                                               rtol=1e-5, atol=1e-6)
                    self.assertEqual(together[output].shape, (2, 4))
                    self.assertTrue(torch.isfinite(together[output]).all())
                probs = probabilities(together[0], [{"slope": 1., "intercept": 0.}] * 4)
                self.assertTrue(((probs >= 0) & (probs <= 1)).all())
                self.assertTrue(((together[1] >= 0) & (together[1] <= 1)).all())

    def test_labels_and_post_outage_measurements_are_not_model_inputs(self):
        original = sample_graph(5)
        changed = original.clone()
        for key in ["y", "extent", "dns_fraction", "voltage_fraction", "overload_fraction"]:
            changed[key] = torch.full_like(changed[key], float("nan"))
        changed.node_affected = ~changed.node_affected
        changed.region_valid = ~changed.region_valid
        for name in ["gin", "gat", "transformer"]:
            with self.subTest(model=name), torch.no_grad():
                model = self.model(name)
                before = model(Batch.from_data_list([original]))
                after = model(Batch.from_data_list([changed]))
                torch.testing.assert_close(before[0], after[0], rtol=0, atol=0)
                torch.testing.assert_close(before[1], after[1], rtol=0, atol=0)

    def test_split_keeps_every_repetition_of_outage_together(self):
        graphs = [sample_graph(i) for i in range(20)]
        graphs += [g.clone() for g in graphs] + [g.clone() for g in graphs]
        splits = grouped_split(graphs, seed=9)
        self.assertEqual(splits, grouped_split(graphs, seed=9))
        all_indices = [i for indices in splits.values() for i in indices]
        self.assertEqual(sorted(all_indices), list(range(len(graphs))))
        groups = {name: {graphs[i].outage_group for i in indices} for name, indices in splits.items()}
        for first, left in groups.items():
            self.assertTrue(left)
            for second, right in groups.items():
                if first != second:
                    self.assertFalse(left & right)
        with self.assertRaises(ValueError):
            grouped_split(graphs[:11])

    def test_normalization_uses_training_nodes_and_preserves_outage_marker(self):
        graphs = [sample_graph(i) for i in range(3)]
        training_values = torch.cat([graphs[0].x, graphs[1].x])
        graphs[2].x[:, :3] += 10000
        original_markers = [g.x[:, 3].clone() for g in graphs]
        mean, scale = normalize_graphs(graphs, [0, 1])
        torch.testing.assert_close(mean[:3], training_values[:, :3].mean(0))
        torch.testing.assert_close(scale[:3], training_values[:, :3].std(0))
        for graph, marker in zip(graphs, original_markers):
            torch.testing.assert_close(graph.x[:, 3], marker, rtol=0, atol=0)

    def test_calibration_reports_missing_classes_and_ignores_invalid_counties(self):
        logits = torch.stack((torch.linspace(-2, 2, 20), torch.zeros(20), torch.zeros(20)), dim=1)
        labels = torch.zeros_like(logits)
        labels[10:, 0] = 1
        labels[:, 1] = 1
        valid = torch.ones_like(labels, dtype=torch.bool)
        valid[:, 2] = False
        fitted = fit_calibration(logits, labels, valid)
        self.assertEqual(fitted[0]["status"], "calibrated")
        self.assertEqual((fitted[0]["positive_count"], fitted[0]["negative_count"]), (10, 10))
        self.assertEqual(fitted[1]["status"], "insufficient_labels")
        self.assertEqual((fitted[1]["positive_count"], fitted[1]["negative_count"]), (20, 0))
        self.assertEqual(fitted[2]["status"], "insufficient_labels")
        self.assertEqual((fitted[2]["positive_count"], fitted[2]["negative_count"]), (0, 0))
        result = probabilities(logits, fitted)
        self.assertTrue(torch.isfinite(result).all())
        self.assertTrue(((result >= 0) & (result <= 1)).all())

    def test_single_class_metrics_have_no_auc_and_empty_counties_have_no_score(self):
        labels = torch.tensor([[0., 1., 0.], [1., 1., 1.]])
        probs = torch.tensor([[0.1, 0.8, 0.4], [0.9, 0.6, 0.5]])
        valid = torch.tensor([[True, True, False], [True, True, False]])
        scores = probability_metrics(labels, probs, labels, labels * 0.8, valid)
        self.assertAlmostEqual(scores[0]["roc_auc"], 1.)
        self.assertAlmostEqual(scores[0]["brier_score"], 0.01, places=6)
        self.assertIsNone(scores[1]["roc_auc"])
        self.assertEqual(scores[2], {"samples": 0})
        for score in scores[:2]:
            self.assertEqual(sum(b["count"] for b in score["reliability_bins"]), score["samples"])

    def test_reliability_bins_include_every_boundary_probability(self):
        probs = torch.arange(11, dtype=torch.float64).reshape(-1, 1) / 10
        labels = torch.zeros_like(probs)
        score = probability_metrics(labels, probs, labels, labels,
                                    torch.ones_like(labels, dtype=torch.bool))[0]
        self.assertEqual(sum(b["count"] for b in score["reliability_bins"]), len(probs))
        self.assertAlmostEqual(score["expected_calibration_error"], 0.5)

    def test_training_and_prediction_with_invalid_county_complete(self):
        graphs = [sample_graph(i) for i in range(12)]
        splits = grouped_split(graphs, seed=4)
        model = self.model()
        history = train_regional_model(model, graphs, splits, "cpu", num_epochs=2,
                                       batch_size=2, patience=1)
        self.assertGreaterEqual(len(history), 1)
        self.assertTrue(all(torch.isfinite(torch.tensor(row["tune_loss"])) for row in history))
        logits, extents = predict(model, graphs, splits["test"], "cpu", batch_size=1)
        self.assertEqual(logits.shape, (len(splits["test"]), 4))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(extents).all())


if __name__ == "__main__":
    unittest.main()
