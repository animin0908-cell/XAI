"""OpenDSS N-1 outage graphs for IEEE 9500 feeder."""

from pathlib import Path
import hashlib
import math
import re
import shutil

import numpy as np
from opendssdirect import dss
import torch
from torch_geometric.data import Data, InMemoryDataset


class IEEE9500N1(InMemoryDataset):
    """Build graph-level DNS regression samples from single-line outages."""

    def __init__(self, root, master_file=None, max_scenarios=128, seed=0):
        self.master_file = Path(master_file) if master_file else Path(root) / "raw" / "Master-unbal-initial-config.dss"
        self.max_scenarios = int(max_scenarios)
        self.seed = int(seed)
        if self.max_scenarios < 0:
            raise ValueError("max_scenarios must be 0 (all lines) or a positive integer.")
        if not self.master_file.is_file():
            raise FileNotFoundError(
                f"IEEE 9500 master DSS file not found: {self.master_file}. "
                "Pass --dss_master_file or place the complete DSS model under ieee9500/raw/."
            )
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        scenario_tag = "all" if self.max_scenarios == 0 else str(self.max_scenarios)
        source_files = sorted(self.master_file.parent.glob("*.dss"))
        source_state = "|".join(
            f"{path.name}:{path.stat().st_size}:{path.stat().st_mtime_ns}"
            for path in source_files
        )
        source_tag = hashlib.sha1(source_state.encode()).hexdigest()[:10]
        return f"ieee9500_n1_{self.master_file.stem}_{source_tag}_{scenario_tag}_seed{self.seed}.pt"

    def download(self):
        return

    def process(self):
        dss.Basic.ClearAll()
        master_file = self._prepare_compatible_master()
        dss.Text.Command(f'compile "{master_file.resolve()}"')
        dss.Solution.Solve()
        if not dss.Solution.Converged():
            raise RuntimeError(f"OpenDSS failed to solve base case: {self.master_file}")

        bus_names = list(dss.Circuit.AllBusNames())
        bus_to_idx = {name.lower(): idx for idx, name in enumerate(bus_names)}
        if not bus_names:
            raise RuntimeError("OpenDSS returned no buses for the IEEE 9500 feeder.")

        branches = self._read_branches(bus_to_idx)
        if not branches:
            raise RuntimeError("No enabled Line or Transformer branches found in the DSS model.")

        candidates = [i for i, branch in enumerate(branches) if branch["is_line"] and not branch["is_switch"]]
        if not candidates:
            raise RuntimeError("No enabled non-switch lines are available for N-1 scenarios.")
        if self.max_scenarios > 0 and len(candidates) > self.max_scenarios:
            rng = np.random.default_rng(self.seed)
            candidates = sorted(rng.choice(candidates, size=self.max_scenarios, replace=False).tolist())

        nominal_p, nominal_q = self._read_loads(bus_to_idx)
        total_base_load_kw = self._served_load_kw()
        if total_base_load_kw <= 0:
            raise RuntimeError("OpenDSS reported no served load in the base case.")
        base_node_features = self._make_node_features(bus_names, nominal_p, nominal_q)

        data_list = []
        for scenario_idx, branch_idx in enumerate(candidates):
            branch = branches[branch_idx]
            dss.Circuit.SetActiveElement(branch["name"])
            dss.CktElement.Enabled(False)
            dss.Solution.Solve()
            if not dss.Solution.Converged():
                dss.Circuit.SetActiveElement(branch["name"])
                dss.CktElement.Enabled(True)
                dss.Solution.Solve()
                continue

            x = base_node_features
            dns_mw = max(0.0, total_base_load_kw - self._served_load_kw()) / 1000.0
            scenario_edges, scenario_edge_attr = self._make_edges(branches, excluded_branch=branch_idx)
            data_list.append(
                Data(
                    x=x,
                    edge_index=scenario_edges,
                    edge_attr=scenario_edge_attr,
                    y=torch.tensor([dns_mw], dtype=torch.float32),
                    scenario_id=torch.tensor([scenario_idx], dtype=torch.long),
                )
            )

            dss.Circuit.SetActiveElement(branch["name"])
            dss.CktElement.Enabled(True)
            dss.Solution.Solve()
            if not dss.Solution.Converged():
                raise RuntimeError(f"OpenDSS could not restore base case after {branch['name']} outage.")

        if len(data_list) < 3:
            raise RuntimeError(
                f"Only {len(data_list)} converged N-1 scenarios; at least 3 are needed for train/validation/test splits."
            )
        torch.save(self.collate(data_list), self.processed_paths[0])

    def _prepare_compatible_master(self):
        generator_file = self.master_file.parent / "Generators.dss"
        source_text = generator_file.read_text()
        pattern = re.compile(
            r"(?im)^(\s*New\s+PVsystem\.[^\r\n!]*?)\s+Mode\s*=\s*7\b"
        )
        compatible_text, replacements = pattern.subn(r"\1 Model=1 PF=1", source_text)
        if not replacements:
            return self.master_file

        source_state = "|".join(
            f"{path.name}:{path.stat().st_size}:{path.stat().st_mtime_ns}"
            for path in sorted(self.master_file.parent.glob("*.dss"))
        )
        source_tag = hashlib.sha1(source_state.encode()).hexdigest()[:10]
        staged_dir = Path(self.processed_dir) / "opendss_compat" / source_tag
        staged_dir.mkdir(parents=True, exist_ok=True)
        for source_file in self.master_file.parent.glob("*.dss"):
            shutil.copy2(source_file, staged_dir / source_file.name)
        (staged_dir / generator_file.name).write_text(compatible_text)
        return staged_dir / self.master_file.name

    @staticmethod
    def _bus_key(bus_name):
        return bus_name.split(".", 1)[0].strip().lower()

    @classmethod
    def _read_branches(cls, bus_to_idx):
        branches = []
        for element_name in dss.Circuit.AllElementNames():
            lowered = element_name.lower()
            is_line = lowered.startswith("line.")
            is_transformer = lowered.startswith("transformer.")
            if not (is_line or is_transformer):
                continue
            dss.Circuit.SetActiveElement(element_name)
            if not dss.CktElement.Enabled():
                continue
            buses = dss.CktElement.BusNames()
            if len(buses) < 2:
                continue
            source = bus_to_idx.get(cls._bus_key(buses[0]))
            target = bus_to_idx.get(cls._bus_key(buses[1]))
            if source is None or target is None or source == target:
                continue

            is_switch = False
            rating = 0.0
            object_name = element_name.split(".", 1)[1]
            if is_line:
                dss.Lines.Name(object_name)
                is_switch = bool(dss.Lines.IsSwitch())
                rating = max(float(dss.Lines.NormAmps()), 0.0)
            else:
                dss.Transformers.Name(object_name)
                dss.Transformers.Wdg(1)
                rating = max(float(dss.Transformers.kVA()), 0.0)

            phases = float(dss.CktElement.NumPhases())
            feature = [float(is_line), float(is_transformer), phases, math.log1p(rating)]
            branches.append(
                {
                    "name": element_name,
                    "source": source,
                    "target": target,
                    "feature": feature,
                    "is_line": is_line,
                    "is_switch": is_switch,
                }
            )
        return branches

    @staticmethod
    def _read_loads(bus_to_idx):
        nominal_p = np.zeros(len(bus_to_idx), dtype=np.float32)
        nominal_q = np.zeros(len(bus_to_idx), dtype=np.float32)
        load_idx = dss.Loads.First()
        while load_idx:
            buses = dss.CktElement.BusNames()
            if buses:
                bus_idx = bus_to_idx.get(IEEE9500N1._bus_key(buses[0]))
                if bus_idx is not None:
                    nominal_p[bus_idx] += float(dss.Loads.kW())
                    nominal_q[bus_idx] += float(dss.Loads.kvar())
            load_idx = dss.Loads.Next()
        return nominal_p, nominal_q

    @staticmethod
    def _served_load_kw():
        served_kw = 0.0
        load_idx = dss.Loads.First()
        while load_idx:
            powers = dss.CktElement.Powers()
            num_conductors = dss.CktElement.NumConductors()
            terminal_powers = powers[: 2 * num_conductors : 2]
            served_kw += abs(float(sum(terminal_powers)))
            load_idx = dss.Loads.Next()
        return served_kw

    @staticmethod
    def _make_node_features(bus_names, nominal_p, nominal_q):
        p_scale = max(float(np.abs(nominal_p).sum()), 1.0)
        q_scale = max(float(np.abs(nominal_q).sum()), 1.0)
        features = np.zeros((len(bus_names), 3), dtype=np.float32)
        features[:, 0] = nominal_p / p_scale
        features[:, 1] = nominal_q / q_scale
        for bus_idx, bus_name in enumerate(bus_names):
            dss.Circuit.SetActiveBus(bus_name)
            magnitudes = dss.Bus.puVmagAngle()[::2]
            if magnitudes:
                features[bus_idx, 2] = float(np.mean(magnitudes))
        return torch.from_numpy(features)

    @staticmethod
    def _make_edges(branches, excluded_branch):
        sources = []
        targets = []
        features = []
        for branch_idx, branch in enumerate(branches):
            if branch_idx == excluded_branch:
                continue
            source, target = branch["source"], branch["target"]
            sources.extend((source, target))
            targets.extend((target, source))
            features.extend((branch["feature"], branch["feature"]))
        return (
            torch.tensor([sources, targets], dtype=torch.long),
            torch.tensor(features, dtype=torch.float32),
        )
