"""OpenDSS integration checks for regional contingency dataset labels."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch_geometric.data import Batch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import dataset.ieee9500_risk as risk


class IEEE9500RiskTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="xai-risk-test-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.output = self.root / "output"

    def feeder(self, normal_amps=500, open_tie=False, unloaded_stub=False):
        electrical = [
            "new circuit.test basekv=12.47 pu=1 phases=3 bus1=sourcebus",
            f"new line.a bus1=sourcebus bus2=mid phases=3 r1=.01 x1=.02 r0=.02 x0=.04 length=.1 units=km normamps={normal_amps}",
            f"new line.b bus1=mid bus2=end phases=3 r1=.01 x1=.02 r0=.02 x0=.04 length=.1 units=km normamps={normal_amps}",
            "new load.a bus1=mid phases=3 conn=wye kv=12.47 kw=100 kvar=20",
            "new load.b bus1=end phases=3 conn=wye kv=12.47 kw=100 kvar=20",
        ]
        if open_tie:
            electrical.extend([
                "new line.tie bus1=sourcebus bus2=end phases=3 switch=yes normamps=500",
                "open line.tie 1",
            ])
        if unloaded_stub:
            electrical.append("new line.stub bus1=mid bus2=stub phases=3 r1=.01 x1=.02 r0=.02 x0=.04 length=.1 units=km normamps=500")
        electrical.extend(["set voltagebases=[12.47]", "calcvoltagebases"])
        (self.source / "electrical.dss").write_text("\n".join(electrical) + "\n")
        (self.source / "busxy.dss").write_text("sourcebus, 0, 0\nmid, 1, 0\nend, 2, 0\n")
        master = self.source / "master.dss"
        master.write_text("clear\nredirect electrical.dss\nbuscoords busxy.dss\nexport voltages\n")
        mapping = {"sourcebus": 0, "mid": 0, "end": 1}
        if unloaded_stub:
            mapping["stub"] = 0
        geography = {"bus_to_region": mapping,
                     "regions": [{"name": "West"}, {"name": "East"}, {"name": "Empty"}]}
        return master, geography

    def source_snapshot(self):
        return {str(path.relative_to(self.source)): path.read_bytes()
                for path in self.source.rglob("*") if path.is_file()}

    def test_disconnected_load_labels_use_pre_outage_features_and_preserve_source(self):
        master, geography = self.feeder()
        source_before = self.source_snapshot()
        working_directory = Path.cwd()
        graphs, metadata = risk.generate_risk_dataset(master, self.output, geography, num_scenarios=0)
        self.assertEqual(metadata["num_converged"], 2)
        self.assertEqual(metadata["baseline"]["y"], [0., 0., 0.])
        self.assertEqual(metadata["baseline"]["extent"], [0., 0., 0.])
        by_outage = {graph.outage_group: graph for graph in graphs}
        self.assertEqual(by_outage["line.a"].y.tolist(), [[1., 1., 0.]])
        self.assertEqual(by_outage["line.b"].y.tolist(), [[0., 1., 0.]])
        torch.testing.assert_close(by_outage["line.a"].extent, torch.tensor([[1., 1., 0.]]))
        torch.testing.assert_close(by_outage["line.b"].extent, torch.tensor([[0., 1., 0.]]))
        bus_index = {name: index for index, name in enumerate(metadata["bus_names"])}
        self.assertTrue(by_outage["line.b"].node_affected[bus_index["end"]])
        self.assertFalse(by_outage["line.b"].node_affected[bus_index["mid"]])
        self.assertFalse(by_outage["line.a"].node_affected[bus_index["sourcebus"]])
        for graph in graphs:
            self.assertEqual(graph.x.shape, (3, 4))
            self.assertGreater(float(graph.x[:, 2].min()), 0.99)
            self.assertEqual(float(graph.x[:, 3].sum()), 2.)
            self.assertEqual(graph.region_valid.tolist(), [[True, True, False]])
        batch = Batch.from_data_list(graphs)
        self.assertEqual(batch.y.shape, (2, 3))
        self.assertEqual(batch.region_id.tolist(), graphs[0].region_id.tolist() * 2)
        self.assertEqual(self.source_snapshot(), source_before)
        self.assertEqual(Path.cwd(), working_directory)

    def test_event_type_changes_baseline_and_cache_without_changing_union_extent(self):
        master, geography = self.feeder(normal_amps=1)
        results = {}
        for event in ["any", "dns", "voltage", "overload"]:
            graphs, metadata = risk.generate_risk_dataset(master, self.output, geography,
                                                          num_scenarios=0, problem_type=event)
            results[event] = graphs, metadata
            expected = [1., 1., 0.] if event in {"any", "overload"} else [0., 0., 0.]
            self.assertEqual(metadata["baseline"]["y"], expected)
            self.assertEqual(metadata["problem_type"], event)
        self.assertEqual(len({metadata["cache_key"] for _, metadata in results.values()}), 4)
        for event, (graphs, _) in results.items():
            for graph, reference in zip(graphs, results["any"][0]):
                torch.testing.assert_close(graph.extent, reference.extent)
        cached_graphs, cached_metadata = risk.generate_risk_dataset(
            master, self.output, geography, num_scenarios=0, problem_type="dns")
        self.assertEqual(cached_metadata["cache_key"], results["dns"][1]["cache_key"])
        for graph, reference in zip(cached_graphs, results["dns"][0]):
            torch.testing.assert_close(graph.y, reference.y)

    def test_open_switch_is_absent_from_graph_and_outage_candidates(self):
        master, geography = self.feeder(open_tie=True)
        prepared = risk.prepare_feeder(master, self.output)
        self.assertEqual({branch["name"].lower() for branch in prepared["branches"]},
                         {"line.a", "line.b"})
        graphs, metadata = risk.generate_risk_dataset(master, self.output, geography, num_scenarios=0)
        self.assertEqual(metadata["num_eligible_lines"], 2)
        self.assertEqual({graph.outage_group for graph in graphs}, {"line.a", "line.b"})
        self.assertTrue(all(graph.edge_index.shape == (2, 2) for graph in graphs))
        self.assertEqual(next(graph.y.tolist() for graph in graphs if graph.outage_group == "line.b"),
                         [[0., 1., 0.]])

    def test_disappearing_unloaded_bus_keeps_fixed_graph_without_affecting_load_extent(self):
        master, geography = self.feeder(unloaded_stub=True)
        graphs, metadata = risk.generate_risk_dataset(master, self.output, geography, num_scenarios=0)
        self.assertEqual(metadata["num_converged"], 3)
        stub_index = metadata["bus_names"].index("stub")
        graph = next(graph for graph in graphs if graph.outage_group == "line.stub")
        self.assertEqual(graph.x.shape, (4, 4))
        self.assertEqual(graph.y.tolist(), [[0., 0., 0.]])
        self.assertEqual(graph.extent.tolist(), [[0., 0., 0.]])
        self.assertFalse(graph.node_affected[stub_index])
        self.assertEqual(metadata["node_nominal_load_kw"][stub_index], 0.)
        # The final candidate need not be the stub, so reproduce that outage
        # directly and inspect the same fixed-order voltage label routine.
        self.assertTrue(risk._compile(Path(metadata["staged_master_file"])))
        risk.dss.Circuit.SetActiveElement("line.stub")
        risk.dss.CktElement.Enabled(False)
        risk.dss.Solution.Solve()
        self.assertLess(risk.dss.Circuit.SetActiveBus("stub"), 0)
        violations, deenergized = risk._bus_voltage_flags(metadata["bus_names"], .95, 1.05)
        self.assertTrue(violations[stub_index])
        self.assertTrue(deenergized[stub_index])

    def test_unresolved_solve_is_recorded_without_safe_label(self):
        master, geography = self.feeder()
        compile_feeder = risk._compile
        compile_calls = 0
        def one_failed_baseline(*args, **kwargs):
            nonlocal compile_calls
            compile_calls += 1
            if compile_calls == 2:
                return False
            return compile_feeder(*args, **kwargs)
        with patch.object(risk, "_compile", side_effect=one_failed_baseline):
            graphs, metadata = risk.generate_risk_dataset(master, self.output, geography, num_scenarios=0)
        self.assertEqual(metadata["num_attempted"], 2)
        self.assertEqual(metadata["num_converged"], 1)
        self.assertEqual(metadata["invalid_scenarios"], 1)
        unresolved = next(scenario for scenario in metadata["scenarios"] if not scenario["converged"])
        self.assertEqual(unresolved["status"], "baseline_nonconverged")
        self.assertNotIn("y", unresolved)
        self.assertNotIn(unresolved["scenario_id"], [int(graph.scenario_id.item()) for graph in graphs])


if __name__ == "__main__":
    unittest.main()
