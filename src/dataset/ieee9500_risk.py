"""Regional labels for a declared IEEE 9500 single-line outage ensemble.

These samples describe consequences conditional on an N-1 outage. They do not
provide an equipment failure rate or a probability over a future time window.
The original DSS files are never compiled directly: all work happens in a copy.
"""

from pathlib import Path
import hashlib
import json
import math
import re
import shutil

import numpy as np
from opendssdirect import dss
import torch
from torch_geometric.data import Data

from .ieee9500 import IEEE9500N1


def _source_fingerprint(master_file):
    digest = hashlib.sha256()
    for path in sorted(master_file.parent.rglob("*")):
        if path.is_file():
            digest.update(str(path.relative_to(master_file.parent)).encode())
            digest.update(path.read_bytes())
    digest.update(master_file.name.encode())
    return digest.hexdigest()[:20]


def _compile(master_file, load_multiplier=1.0):
    dss.Basic.AllowChangeDir(False)
    dss.Basic.AllowEditor(False)
    dss.Basic.AllowForms(False)
    dss.Basic.AllowDOScmd(False)
    dss.Basic.ClearAll()
    dss.Basic.DataPath(str(master_file.parent))
    dss.Text.Command(f'compile "{master_file}"')
    # Exported DSS snapshots omit the original IEEE master iteration limits.
    dss.Solution.MaxIterations(100)
    dss.Solution.MaxControlIterations(100)
    dss.Solution.LoadMult(float(load_multiplier))
    try:
        dss.Solution.Solve()
    except dss.DSSException:
        return False
    return bool(dss.Solution.Converged())


def _connected_branches(bus_to_idx):
    branches = IEEE9500N1._read_branches(bus_to_idx)
    connected = []
    for branch in branches:
        dss.Circuit.SetActiveElement(branch["name"])
        phases = max(1, int(dss.CktElement.NumPhases()))
        # An open switch is enabled in DSS but must not connect the graph.
        if any(all(dss.CktElement.IsOpen(terminal, phase)
                   for phase in range(1, phases + 1))
               for terminal in (1, 2)):
            continue
        branch["norm_amps"] = 0.0
        if branch["is_line"]:
            dss.Lines.Name(branch["name"].split(".", 1)[1])
            branch["norm_amps"] = max(0.0, float(dss.Lines.NormAmps()))
        connected.append(branch)
    return connected


def _load_records(bus_to_idx):
    records = []
    current = dss.Loads.First()
    while current:
        buses = dss.CktElement.BusNames()
        if buses:
            bus = bus_to_idx.get(IEEE9500N1._bus_key(buses[0]))
            if bus is not None:
                conductors = int(dss.CktElement.NumConductors())
                records.append({"name": "Load." + dss.Loads.Name(),
                                "bus": bus, "nominal_kw": max(0.0, float(dss.Loads.kW())),
                                "served_kw": max(0.0, float(sum(
                                    dss.CktElement.Powers()[:2 * conductors:2])))})
        current = dss.Loads.Next()
    return records


def prepare_feeder(master_file, output_dir):
    """Stage and solve a DSS feeder; return its buses, coordinates, and topology.

    Coordinates are DSS X/Y values; their CRS is deliberately not guessed here.
    A caller must declare a CRS before interpreting these as county locations.
    """
    master_file = Path(master_file).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if not master_file.is_file():
        raise FileNotFoundError(f"DSS master file not found: {master_file}")
    if output_dir == master_file.parent or output_dir.is_relative_to(master_file.parent):
        raise ValueError("Choose an output directory outside the source DSS directory.")
    source_fingerprint = _source_fingerprint(master_file)
    staged_dir = output_dir / "opendss_compat" / source_fingerprint
    staged_master = staged_dir / master_file.name
    marker = staged_dir / ".regional_risk_staged_v2"
    if not marker.is_file():
        staged_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(master_file.parent, staged_dir, dirs_exist_ok=True)
        for path in staged_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() == ".dss":
                content = path.read_text(errors="replace")
                # The legacy IEEE model contains PV Mode=7, unsupported by
                # DSS C-API. Apply the same compatibility fix as IEEE9500N1.
                content = re.sub(r"(?im)^(\s*New\s+PVsystem\.[^\r\n!]*?)\s+Mode\s*=\s*7\b",
                                 r"\1 Model=1 PF=1", content)
                # Optional source-side output commands are unnecessary for a
                # simulation and may contain absolute write destinations.
                content = re.sub(r"(?im)^\s*(?:plot|show|export|save|dump|dos|cd)\b[^\r\n]*",
                                 "! disabled while generating regional risk samples", content)
                content = re.sub(r"(?im)^\s*set\s+datapath\s*=.*$",
                                 "! DataPath is controlled by the dataset builder", content)
                # Redirects inside this model retain their source directory
                # layout when a source happened to use an absolute path.
                content = content.replace(str(master_file.parent), str(staged_dir))
                def absolute_reference(match):
                    reference = next(value for value in match.groups()[1:] if value is not None)
                    referenced_path = Path(reference)
                    if not referenced_path.is_absolute():
                        referenced_path = path.parent / referenced_path
                    referenced_path = referenced_path.resolve()
                    if not referenced_path.is_relative_to(staged_dir):
                        raise ValueError(f"DSS reference is outside the staged source tree: {reference}")
                    return f'{match.group(1)}"{referenced_path}"'
                content = re.sub(r'''(?im)^(\s*(?:redirect|compile|buscoords|latlongcoords|uuids)\s+)(?:"([^"]+)"|'([^']+)'|([^\s!]+))''',
                                 absolute_reference, content)
                path.write_text(content)
        marker.write_text(source_fingerprint)
    if not _compile(staged_master):
        raise RuntimeError("OpenDSS did not converge for the initial operating case.")
    bus_names = list(dss.Circuit.AllBusNames())
    if not bus_names:
        raise RuntimeError("The DSS circuit has no buses.")
    bus_to_idx = {name.lower(): i for i, name in enumerate(bus_names)}
    coordinates = []
    for name in bus_names:
        dss.Circuit.SetActiveBus(name)
        coordinates.append({"bus": name.lower(), "x": float(dss.Bus.X()),
                            "y": float(dss.Bus.Y()),
                            "has_coordinates": bool(dss.Bus.Coorddefined())})
    branches = _connected_branches(bus_to_idx)
    nominal_p, nominal_q = IEEE9500N1._read_loads(bus_to_idx)
    if not branches or float(nominal_p.sum()) <= 0:
        raise RuntimeError("The DSS circuit needs connected branches and positive loads.")
    return {"master_file": staged_master, "source_master_file": master_file,
            "source_fingerprint": source_fingerprint, "bus_names": bus_names,
            "coordinates": coordinates, "branches": branches,
            "nominal_p": nominal_p, "nominal_q": nominal_q,
            "load_records": _load_records(bus_to_idx)}


def _bus_voltage_flags(bus_names, lower, upper):
    violation = np.zeros(len(bus_names), dtype=bool)
    deenergized = np.zeros(len(bus_names), dtype=bool)
    for i, name in enumerate(bus_names):
        # Disabling the last incident element can remove a bus from the active
        # DSS circuit. Preserve the fixed graph node and label it de-energized.
        if dss.Circuit.SetActiveBus(name) < 0:
            violation[i] = True
            deenergized[i] = True
            continue
        values = np.asarray(dss.Bus.puVmagAngle()[::2], dtype=float)
        nodes = np.asarray(dss.Bus.Nodes())
        if len(nodes) == len(values):
            values = values[nodes > 0]  # exclude an explicit neutral conductor
        if len(values) and not np.isfinite(values).all():
            raise RuntimeError("OpenDSS returned nonfinite bus voltage.")
        valid = len(values) > 0 and bool(np.isfinite(values).all())
        deenergized[i] = not valid or bool(np.all(values < 0.1))
        violation[i] = not valid or bool(np.any((values < lower) | (values > upper)))
    return violation, deenergized


def _served_by_bus(records, num_buses):
    served = np.zeros(num_buses, dtype=float)
    for record in records:
        if dss.Circuit.SetActiveElement(record["name"]) < 0:
            raise RuntimeError(f"Missing load while generating labels: {record['name']}")
        conductors = int(dss.CktElement.NumConductors())
        raw_value = float(sum(dss.CktElement.Powers()[:2 * conductors:2]))
        if not math.isfinite(raw_value):
            raise RuntimeError("OpenDSS returned nonfinite served load.")
        value = max(0.0, raw_value)
        served[record["bus"]] += value
    return served


def _overloads(branches, excluded, node_regions, num_regions):
    overloaded_buses = np.zeros(len(node_regions), dtype=bool)
    overloaded_counts = np.zeros(num_regions, dtype=float)
    rated_counts = np.zeros(num_regions, dtype=float)
    for i, branch in enumerate(branches):
        if i == excluded or not branch["is_line"] or branch["norm_amps"] <= 0:
            continue
        dss.Circuit.SetActiveElement(branch["name"])
        # Both terminals, phase conductors only; neutral rating is distinct.
        currents = np.asarray(dss.CktElement.CurrentsMagAng()[::2], dtype=float)
        conductors = int(dss.CktElement.NumConductors())
        phases = int(dss.CktElement.NumPhases())
        phase_currents = [currents[t * conductors:p + t * conductors]
                          for t in range(int(dss.CktElement.NumTerminals()))
                          for p in [phases]]
        current = np.concatenate(phase_currents) if phase_currents else np.array([])
        if not np.isfinite(current).all():
            raise RuntimeError("OpenDSS returned nonfinite line current.")
        overloaded = bool(len(current) and np.any(current > branch["norm_amps"]))
        regions = {int(node_regions[branch["source"]]), int(node_regions[branch["target"]])}
        for region in regions:
            rated_counts[region] += 1
            if overloaded:
                overloaded_counts[region] += 1
        if overloaded:
            overloaded_buses[[branch["source"], branch["target"]]] = True
    return overloaded_buses, np.divide(overloaded_counts, rated_counts,
                                        out=np.zeros_like(rated_counts), where=rated_counts > 0)


def _regional_labels(feeder, excluded, node_regions, num_regions, before,
                     voltage_min, voltage_max, dns_threshold, extent_threshold,
                     problem_type="any"):
    nominal_p = feeder["nominal_p"]
    region_load = np.bincount(node_regions, weights=nominal_p, minlength=num_regions)
    pre_region_load = np.bincount(node_regions, weights=before, minlength=num_regions)
    after = _served_by_bus(feeder["load_records"], len(node_regions))
    lost = np.maximum(0.0, before - after)
    voltage_bad, deenergized = _bus_voltage_flags(feeder["bus_names"], voltage_min, voltage_max)
    overload_bad, overload_fraction = _overloads(feeder["branches"], excluded, node_regions, num_regions)
    dns_bus_bad = (lost >= before * dns_threshold) & (before > 0)
    affected = (dns_bus_bad | deenergized | voltage_bad | overload_bad) & (nominal_p > 0)
    def fraction(weights, denominator):
        numerator = np.bincount(node_regions, weights=weights, minlength=num_regions)
        return np.divide(numerator, denominator, out=np.zeros(num_regions), where=denominator > 0)
    dns_fraction = fraction(lost, pre_region_load)
    voltage_fraction = fraction(nominal_p * voltage_bad, region_load)
    extent = fraction(nominal_p * affected, region_load)
    events = {"dns": dns_fraction >= dns_threshold,
              "voltage": voltage_fraction >= extent_threshold,
              "overload": overload_fraction > 0,
              "any": ((dns_fraction >= dns_threshold) | (extent >= extent_threshold)
                      | (overload_fraction > 0))}
    y = events[problem_type].astype(np.float32)
    return {"y": y, "extent": extent, "dns_fraction": dns_fraction,
            "voltage_fraction": voltage_fraction, "overload_fraction": overload_fraction,
            "affected": affected}


def generate_risk_dataset(master_file, output_dir, geography, num_scenarios=128,
                          seed=0, load_min=1.0, load_max=1.0, voltage_min=0.95,
                          voltage_max=1.05, dns_fraction_threshold=0.01,
                          min_affected_load_fraction=0.01, branch_probabilities=None,
                          problem_type="any"):
    """Return ``(graphs, metadata)`` for regional N-1 consequence prediction.

    ``geography`` supplies ``bus_to_region`` and a continuous zero-based
    ``regions`` list. Invalid solves are recorded, never assigned a safe label.
    A fresh compile before each scenario resets taps, controls, and switches.
    """
    if int(num_scenarios) != num_scenarios or num_scenarios < 0:
        raise ValueError("num_scenarios must be a nonnegative integer; 0 means all eligible lines.")
    if problem_type not in {"any", "dns", "voltage", "overload"}:
        raise ValueError("problem_type must be any, dns, voltage, or overload.")
    parameters = [load_min, load_max, voltage_min, voltage_max,
                  dns_fraction_threshold, min_affected_load_fraction]
    if not all(math.isfinite(float(value)) for value in parameters):
        raise ValueError("Scenario limits must be finite.")
    if not 0 < load_min <= load_max or not 0 < voltage_min < voltage_max:
        raise ValueError("Require positive ordered load and voltage limits.")
    if not 0 < dns_fraction_threshold <= 1 or not 0 < min_affected_load_fraction <= 1:
        raise ValueError("DNS and affected-load thresholds must be in (0, 1].")
    feeder = prepare_feeder(master_file, output_dir)
    regions = geography["regions"]
    if not regions:
        raise ValueError("At least one region is required.")
    mapping = {IEEE9500N1._bus_key(str(key)): int(value)
               for key, value in geography["bus_to_region"].items()}
    missing = [name for name in feeder["bus_names"] if name.lower() not in mapping]
    if missing:
        raise ValueError(f"Region mapping is missing {len(missing)} buses, including {missing[0]}.")
    node_regions = np.asarray([mapping[name.lower()] for name in feeder["bus_names"]], dtype=np.int64)
    num_regions = len(regions)
    if np.any(node_regions < 0) or np.any(node_regions >= num_regions):
        raise ValueError("Region indices must be zero based and within the regions list.")
    branches = feeder["branches"]
    candidates = [i for i, branch in enumerate(branches) if branch["is_line"] and not branch["is_switch"]]
    if not candidates:
        raise RuntimeError("No connected non-switch lines are eligible for N-1 outages.")
    candidate_names = [branches[i]["name"].lower() for i in candidates]
    distribution = None
    if branch_probabilities is not None:
        supplied = {str(key).lower(): float(value) for key, value in branch_probabilities.items()}
        if set(supplied) != set(candidate_names):
            raise ValueError("branch_probabilities must include exactly every eligible line using its Line.name.")
        distribution = np.asarray([supplied[name] for name in candidate_names], dtype=float)
        if not np.isfinite(distribution).all() or np.any(distribution < 0) or distribution.sum() <= 0:
            raise ValueError("Branch probabilities must be finite, nonnegative, with positive total.")
        distribution = distribution / distribution.sum()
        if num_scenarios == 0:
            raise ValueError("Weighted sampling requires a positive num_scenarios.")
    cache_config = {"version": 3, "source": feeder["source_fingerprint"], "geography": geography,
                    "num_scenarios": int(num_scenarios), "seed": int(seed),
                    "parameters": parameters, "branch_probabilities": branch_probabilities,
                    "problem_type": problem_type}
    cache_key = hashlib.sha256(json.dumps(cache_config, sort_keys=True).encode()).hexdigest()[:20]
    cache_dir = Path(output_dir).resolve() / "dataset" / cache_key
    graph_file, metadata_file = cache_dir / "graphs.pt", cache_dir / "metadata.json"
    if graph_file.is_file() and metadata_file.is_file():
        return torch.load(graph_file, map_location="cpu", weights_only=False), json.loads(metadata_file.read_text())
    rng = np.random.default_rng(int(seed))
    if distribution is None:
        count = len(candidates) if num_scenarios == 0 else min(int(num_scenarios), len(candidates))
        sampled = rng.choice(candidates, size=count, replace=False).tolist()
    else:
        sampled = rng.choice(candidates, size=int(num_scenarios), replace=True, p=distribution).tolist()
    nominal_p, nominal_q = feeder["nominal_p"], feeder["nominal_q"]
    region_load = np.bincount(node_regions, weights=nominal_p, minlength=num_regions)
    region_valid = region_load > 0
    if not region_valid.any():
        raise ValueError("At least one region must contain positive nominal load.")
    baseline_before = _served_by_bus(feeder["load_records"], len(node_regions))
    baseline_values = _regional_labels(feeder, None, node_regions, num_regions,
                                      baseline_before, voltage_min, voltage_max,
                                      dns_fraction_threshold, min_affected_load_fraction,
                                      problem_type)
    baseline = {name: values.tolist() for name, values in baseline_values.items() if name != "affected"}
    baseline.update(load_multiplier=1.0, converged=True)
    graphs, scenarios = [], []
    for scenario_id, excluded in enumerate(sampled):
        if scenario_id == 0 or (scenario_id + 1) % 16 == 0 or scenario_id + 1 == len(sampled):
            print(f"[IEEE9500 risk] scenario {scenario_id + 1}/{len(sampled)}, "
                  f"valid completed={len(graphs)}", flush=True)
        multiplier = float(rng.uniform(load_min, load_max)) if load_min != load_max else float(load_min)
        record = {"scenario_id": scenario_id, "branch_name": branches[excluded]["name"],
                  "load_multiplier": multiplier, "converged": False, "status": "baseline_nonconverged"}
        scenarios.append(record)
        if not _compile(feeder["master_file"], multiplier):
            continue
        if list(dss.Circuit.AllBusNames()) != feeder["bus_names"]:
            raise RuntimeError("Bus ordering changed during scenario compilation.")
        try:
            before = _served_by_bus(feeder["load_records"], len(node_regions))
        except (RuntimeError, dss.DSSException) as error:
            record.update(status="baseline_invalid_numeric_state", error=str(error))
            continue
        pre_region_load = np.bincount(node_regions, weights=before, minlength=num_regions)
        if np.any(pre_region_load[region_valid] <= 0):
            record["status"] = "baseline_region_without_served_load"
            continue
        x = IEEE9500N1._make_node_features(feeder["bus_names"], nominal_p, nominal_q).clone()
        x[:, :2] *= multiplier
        if not bool(torch.isfinite(x).all()):
            record["status"] = "baseline_invalid_numeric_state"
            continue
        outage_endpoint = torch.zeros((len(node_regions), 1), dtype=torch.float32)
        outage_endpoint[[branches[excluded]["source"], branches[excluded]["target"]]] = 1.0
        x = torch.cat((x, outage_endpoint), dim=1)
        dss.Circuit.SetActiveElement(branches[excluded]["name"])
        dss.CktElement.Enabled(False)
        try:
            dss.Solution.Solve()
        except dss.DSSException as error:
            record["status"] = "outage_solver_error"
            record["error"] = str(error)
            continue
        if not dss.Solution.Converged():
            record["status"] = "outage_nonconverged"
            continue
        try:
            labels = _regional_labels(feeder, excluded, node_regions, num_regions, before,
                                      voltage_min, voltage_max, dns_fraction_threshold,
                                      min_affected_load_fraction, problem_type)
        except (RuntimeError, dss.DSSException) as error:
            record.update(status="outage_invalid_numeric_state", error=str(error))
            continue
        y, extent = labels["y"], labels["extent"]
        dns_fraction, voltage_fraction = labels["dns_fraction"], labels["voltage_fraction"]
        overload_fraction, affected = labels["overload_fraction"], labels["affected"]
        edge_index, edge_attr = IEEE9500N1._make_edges(branches, excluded)
        def target(values):
            return torch.tensor(np.asarray(values)[None, :], dtype=torch.float32)
        graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                           region_id=torch.from_numpy(node_regions.copy()),
                           region_valid=torch.tensor(region_valid[None, :], dtype=torch.bool),
                           region_load_kw=target(region_load), node_affected=torch.from_numpy(affected),
                           y=target(y), extent=target(extent), dns_fraction=target(dns_fraction),
                           voltage_fraction=target(voltage_fraction), overload_fraction=target(overload_fraction),
                           scenario_id=torch.tensor([scenario_id], dtype=torch.long),
                           outage_group=branches[excluded]["name"].lower(),
                           load_multiplier=torch.tensor([multiplier], dtype=torch.float32)))
        record.update(converged=True, status="valid", y=y.tolist(), extent=extent.tolist(),
                      dns_fraction=dns_fraction.tolist(), voltage_fraction=voltage_fraction.tolist(),
                      overload_fraction=overload_fraction.tolist())
    metadata = {"source_master_file": str(feeder["source_master_file"]),
                "staged_master_file": str(feeder["master_file"]), "cache_key": cache_key,
                "eligible_candidate_count": len(candidates), "sampled_candidate_count": len(set(sampled)),
                "num_eligible_lines": len(candidates), "num_attempted": len(sampled),
                "num_converged": len(graphs), "baseline": baseline,
                "attempted_scenarios": len(sampled), "converged_scenarios": len(graphs),
                "invalid_scenarios": len(sampled) - len(graphs), "scenarios": scenarios,
                "regions": regions, "bus_names": feeder["bus_names"],
                "node_region": node_regions.tolist(), "region_nominal_load_kw": region_load.tolist(),
                "node_nominal_load_kw": nominal_p.tolist(),
                "region_valid": region_valid.tolist(),
                "problem_type": problem_type,
                "sampling": "uniform_without_replacement" if distribution is None else "weighted_with_replacement",
                "sampling_distribution": "uniform_without_replacement" if distribution is None else "weighted_with_replacement",
                "probability_scope": "Conditional on one sampled, connected non-switch line being out of service; no time horizon or failure frequency is inferred.",
                "thresholds": {"voltage_min": voltage_min, "voltage_max": voltage_max,
                               "dns_fraction": dns_fraction_threshold,
                               "affected_load_fraction": min_affected_load_fraction},
                "definitions": {
                    "input": "Pre-outage P/Q scaled by fixed feeder nominal load, pre-outage mean bus voltage, known outage endpoint indicator, and known post-outage topology; no post-outage voltage or served load is an input.",
                    "dns_fraction": "Sum of positive bus served-kW decreases divided by region pre-outage served kW.",
                    "voltage_fraction": "Nominal-load-weighted share of loaded buses with any non-neutral phase outside voltage thresholds, including de-energized buses.",
                    "overload_fraction": "Share of rated active lines incident to a region exceeding NormAmps on any phase at either terminal; a crossing line counts in both endpoint regions.",
                    "extent": "Union of loaded buses with per-bus DNS fraction at the DNS threshold, de-energization, voltage violations, or an overloaded incident line, weighted by nominal kW. This is a load share, not county land area.",
                    "y": {"any": "Problem if regional DNS fraction or union affected-load share reaches its threshold, or any rated incident line is overloaded.",
                          "dns": "Problem if regional lost served-kW fraction reaches the DNS threshold.",
                          "voltage": "Problem if nominal-load-weighted voltage violation share reaches the affected-load threshold.",
                          "overload": "Problem if any rated incident line is overloaded."}[problem_type],
                    "invalid": "Nonconvergent or invalid baseline/outage solves are unresolved, never treated as safe; include their count when reporting probability bounds."}}
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(graphs, graph_file)
    metadata_file.write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    return graphs, metadata
