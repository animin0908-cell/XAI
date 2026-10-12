"""Simulate IEEE 9500 contingencies and train the existing GNN for county risk."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import zipfile

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import numpy as np
import torch

from dataset.ieee9500_risk import prepare_feeder, generate_risk_dataset
from gnn.regional_risk import (RegionalRiskGNN, grouped_split, normalize_graphs,
                               train_regional_model, predict, fit_calibration,
                               probabilities, probability_metrics)
from utils.county_geography import build_geography, load_bus_coordinates


def extract_dss_archive(archive, output):
    """Extract only DSS assets into a content-addressed local copy."""
    archive = Path(archive).expanduser().resolve()
    destination = Path(output).resolve() / "dss_source" / hashlib.sha256(archive.read_bytes()).hexdigest()[:16]
    with zipfile.ZipFile(archive) as bundle:
        masters = [n for n in bundle.namelist() if n.lower().endswith("_base.dss")]
        if len(masters) != 1:
            raise ValueError("DSS archive must contain one *_base.dss master. Otherwise use --dss-master-file.")
        for name in bundle.namelist():
            path = destination / name
            if not path.resolve().is_relative_to(destination):
                raise ValueError("DSS archive contains a path outside its extraction directory.")
            if name.lower().endswith(".dss"):
                path.parent.mkdir(parents=True, exist_ok=True)
                if not path.exists():
                    path.write_bytes(bundle.read(name))
    return destination / masters[0]


def parse_args(argv=None):
    desktop = Path.home() / "Desktop"
    archive = desktop / "9500-Node-Model-Submission-Model-Files/Exported Models/ieee9500bal_dss.zip"
    coords = desktop / "ieee9500bal_csv.zip"
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--dss-master-file", type=Path)
    source.add_argument("--dss-archive", type=Path)
    parser.add_argument("--bus-coordinates", type=Path,
                        help="IEEE CSV zip or Busname,X,Y CSV; overrides DSS bus coordinates.")
    parser.add_argument("--county-geojson", type=Path)
    parser.add_argument("--bus-county-csv", type=Path,
                        help="Explicit bus,county[,geoid,longitude,latitude] mapping; no guessed counties.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/ieee9500_county_risk")
    parser.add_argument("--num-scenarios", type=int, default=128, help="0 = all eligible lines.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--load-min", type=float, default=1.0)
    parser.add_argument("--load-max", type=float, default=1.0)
    parser.add_argument("--voltage-min", type=float, default=0.95)
    parser.add_argument("--voltage-max", type=float, default=1.05)
    parser.add_argument("--dns-threshold", type=float, default=0.01)
    parser.add_argument("--affected-load-threshold", type=float, default=0.01)
    parser.add_argument("--problem-type", choices=["any", "dns", "voltage", "overload"], default="any",
                        help="Regional problem criterion; default combines DNS, voltage and line overload.")
    parser.add_argument("--branch-probabilities", type=Path,
                        help="JSON mapping every eligible Line.name to conditional outage weight.")
    parser.add_argument("--model-name", choices=["gin", "gat", "transformer"], default="gin")
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--prepare-only", action="store_true", help="Generate labeled graphs and simulation report without training.")
    args = parser.parse_args(argv)
    if args.dss_master_file is None and args.dss_archive is None:
        if not archive.is_file():
            parser.error("Supply --dss-master-file or --dss-archive (a CSV archive alone has no solved scenarios).")
        args.dss_archive = archive
    # The convenience coordinate default is only valid for this exact balanced model.
    if args.bus_coordinates is None and args.dss_archive == archive and coords.is_file():
        args.bus_coordinates = coords
    if args.county_geojson is None and args.bus_county_csv is None:
        args.county_geojson = ROOT / "data/geography/wa_counties_2025.geojson"
    if args.county_geojson is not None and not args.county_geojson.is_file():
        parser.error("County GeoJSON missing. Run .venv/bin/python src/prepare_counties.py first or supply --county-geojson/--bus-county-csv.")
    for name in ["batch_size", "num_epochs", "hidden_dim", "num_layers", "threads", "patience"]:
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    if not 0 <= args.dropout < 1 or not np.isfinite(args.lr) or args.lr <= 0:
        parser.error("Require dropout in [0,1) and positive finite learning rate.")
    return args


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def make_reports(graphs, metadata, geography, coordinates, output, model_output=None):
    attempted = metadata["attempted_scenarios"]
    unresolved = metadata["invalid_scenarios"]
    if not attempted:
        raise ValueError("No contingencies were generated.")
    regions = geography["regions"]
    num_regions = len(regions)
    observed_y = torch.cat([g.y for g in graphs]).numpy() if graphs else np.empty((0, num_regions))
    observed_extent = torch.cat([g.extent for g in graphs]).numpy() if graphs else np.empty((0, num_regions))
    warnings = []
    if unresolved:
        warnings.append(f"{unresolved} of {attempted} scenarios unresolved; probabilities have lower/upper bounds rather than assuming those scenarios safe.")
    if geography["coverage"]["unassigned_buses"]:
        warnings.append(f"{geography['coverage']['unassigned_buses']} buses have no county assignment and are reported as Unknown.")
    if attempted < 100:
        warnings.append("Small scenario sample: use this run to check the pipeline; probabilities are exploratory.")
    if metadata.get("model_skipped_reason"):
        warnings.append(metadata["model_skipped_reason"])
    baseline = metadata.get("baseline", {})
    county_rows = []
    for r, region in enumerate(regions):
        loaded = bool(metadata["region_valid"][r])
        county = region["kind"] == "county"
        known = loaded and county
        hits = int(observed_y[:, r].sum())
        row = {"county": region["name"], "geoid": region["geoid"], "is_county": county,
               "modeled_bus_count": region["bus_count"], "modeled_nominal_load_kw": metadata["region_nominal_load_kw"][r],
               "county_area_km2": region["area_km2"], "modeled_bounds_lonlat": region["modeled_bounds"],
               "modeled_bbox_area_km2": region["modeled_bbox_area_km2"],
               "attempted_scenarios": attempted, "resolved_scenarios": len(graphs), "unresolved_scenarios": unresolved,
               "simulation_probability": hits / attempted if known and not unresolved else None,
               "simulation_frequency_resolved": hits / len(graphs) if known and graphs else None,
               "risk_probability_lower_bound": hits / attempted if known else None,
               "risk_probability_upper_bound": (hits + unresolved) / attempted if known else None,
               "expected_affected_load_fraction_resolved": float(observed_extent[:, r].mean()) if loaded and graphs else None,
               "p95_affected_load_fraction_resolved": float(np.quantile(observed_extent[:, r], 0.95)) if loaded and graphs else None,
               "baseline_already_has_problem": bool(baseline.get("y", [False] * num_regions)[r]) if known else None,
               "baseline_affected_load_fraction": baseline.get("extent", [None] * num_regions)[r] if loaded else None,
               "scope": region["scope_note"]}
        if known and row["baseline_already_has_problem"]:
            warnings.append(f"{region['name']}: baseline already meets the selected problem criterion; contingency probabilities include pre-existing problems.")
        for threshold in [0.01, 0.05, 0.1, 0.25, 0.5]:
            extent_hits = int((observed_extent[:, r] >= threshold).sum())
            label = f"affected_load_ge_{int(threshold * 100)}pct"
            row[f"{label}_probability"] = extent_hits / attempted if known and not unresolved else None
            row[f"{label}_lower_bound"] = extent_hits / attempted if known else None
            row[f"{label}_upper_bound"] = (extent_hits + unresolved) / attempted if known else None
        if model_output is not None:
            status = model_output["calibration"][r]["status"]
            mean_score = float(model_output["probabilities"][:, r].mean())
            row.update(calibration_status=status,
                       gnn_probability_resolved=mean_score if known and status == "calibrated" else None,
                       gnn_uncalibrated_score=mean_score if known and status != "calibrated" else None,
                       gnn_expected_affected_load_fraction=float(model_output["extent"][:, r].mean()) if known else None,
                       test_brier_score=model_output["test_metrics"][r].get("brier_score") if known else None)
            if known and status != "calibrated":
                warnings.append(f"{region['name']}: insufficient positive/negative calibration labels; GNN output is an uncalibrated score.")
        county_rows.append(row)
    scenario_lookup = {s["scenario_id"]: s for s in metadata["scenarios"]}
    scenario_rows = []
    for i, graph in enumerate(graphs):
        scenario_id = int(graph.scenario_id.item())
        scenario = scenario_lookup[scenario_id]
        for r, region in enumerate(regions):
            row = {"scenario_id": scenario_id, "county": region["name"], "geoid": region["geoid"],
                   "outage_line": scenario["branch_name"], "load_multiplier": scenario["load_multiplier"],
                   "simulated_problem": int(graph.y[0, r]), "affected_load_fraction": float(graph.extent[0, r]),
                   "dns_fraction": float(graph.dns_fraction[0, r]), "voltage_fraction": float(graph.voltage_fraction[0, r]),
                   "overload_fraction": float(graph.overload_fraction[0, r])}
            if model_output is not None:
                row.update(gnn_score=float(model_output["probabilities"][i, r]),
                           calibration_status=model_output["calibration"][r]["status"],
                           gnn_affected_load_fraction=float(model_output["extent"][i, r]),
                           split=model_output["split_names"][i])
            scenario_rows.append(row)
    # Write unresolved scenarios too; they must not silently disappear from exported data.
    for scenario in metadata["scenarios"]:
        if not scenario["converged"]:
            for region in regions:
                scenario_rows.append({"scenario_id": scenario["scenario_id"], "county": region["name"],
                                      "outage_line": scenario["branch_name"], "load_multiplier": scenario["load_multiplier"],
                                      "status": scenario["status"]})
    exposure = torch.stack([g.node_affected for g in graphs]).float().mean(0).numpy() if graphs else None
    bus_rows, bus_features = [], []
    for i, bus in enumerate(metadata["bus_names"]):
        point = coordinates.get(bus.lower())
        loaded_bus = metadata.get("node_nominal_load_kw", [None] * len(metadata["bus_names"]))[i]
        row = {"bus": bus, "county": regions[metadata["node_region"][i]]["name"],
               "nominal_load_kw": loaded_bus,
               "problem_frequency_resolved": float(exposure[i]) if exposure is not None and loaded_bus and loaded_bus > 0 else None,
               "longitude": point[0] if point else None, "latitude": point[1] if point else None}
        bus_rows.append(row)
        if point:
            bus_features.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": list(point)}, "properties": row})
    write_csv(output / "county_risk.csv", county_rows)
    write_csv(output / "scenario_predictions.csv", scenario_rows)
    write_csv(output / "bus_problem_probability.csv", bus_rows)
    (output / "bus_problem_probability.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": bus_features}))
    features = [{"type": "Feature", "geometry": region["geometry"], "properties": row}
                for region, row in zip(regions, county_rows) if region.get("geometry")]
    (output / "county_risk.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    report = {"probability_scope": metadata["probability_scope"], "problem_type": metadata.get("problem_type", "any"),
              "sampling": metadata["sampling"], "eligible_lines": metadata["eligible_candidate_count"],
              "attempted_scenarios": attempted, "resolved_scenarios": len(graphs),
              "unresolved_scenarios": unresolved, "definitions": metadata["definitions"],
              "thresholds": metadata["thresholds"], "geography_coverage": geography["coverage"],
              "counties": county_rows, "warnings": warnings,
              "probability_bounds_scope": "Unresolved-solve bounds for this sampled ensemble; these are not confidence intervals for sampling uncertainty.",
              "model_report": model_output["report"] if model_output else None,
              "baseline": metadata.get("baseline")}
    (output / "risk_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    plot_risk_map(regions, bus_rows, output)
    return report


def plot_risk_map(regions, buses, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter

    located = [b for b in buses if b["longitude"] is not None and b["problem_frequency_resolved"] is not None]
    if not located:
        return
    fig, ax = plt.subplots(figsize=(10, 7))
    for region in regions:
        geometry = region.get("geometry")
        if not geometry:
            continue
        polygons = geometry["coordinates"] if geometry["type"] == "MultiPolygon" else [geometry["coordinates"]]
        for polygon in polygons:
            ring = np.asarray(polygon[0])
            ax.plot(ring[:, 0], ring[:, 1], color="#6b7280", linewidth=0.7)
    x = np.asarray([b["longitude"] for b in located])
    y = np.asarray([b["latitude"] for b in located])
    color = np.asarray([b["problem_frequency_resolved"] for b in located])
    points = ax.scatter(x, y, c=color, cmap="YlOrRd", vmin=0, vmax=1, s=7)
    margin_x, margin_y = max(float(np.ptp(x)) * 0.08, 0.01), max(float(np.ptp(y)) * 0.08, 0.01)
    ax.set_xlim(x.min() - margin_x, x.max() + margin_x)
    ax.set_ylim(y.min() - margin_y, y.max() + margin_y)
    for region in regions:
        members = [b for b in located if b["county"] == region["name"]]
        if members and region["kind"] == "county":
            ax.text(float(np.median([b["longitude"] for b in members])),
                    float(np.median([b["latitude"] for b in members])), region["name"], fontsize=10,
                    bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 3})
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.yaxis.set_major_locator(MaxNLocator(7))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.set_aspect(1 / np.cos(np.deg2rad(y.mean())))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Modeled IEEE 9500 feeder: load-bus problem frequency under N-1 outages")
    fig.colorbar(points, ax=ax, label="Simulated problem frequency (resolved scenarios)")
    fig.text(0.5, 0.01, "County boundaries: US Census 2025. Coverage is the modeled feeder, not the entire county.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(output / "county_risk_map.png", dpi=160)
    plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    master = args.dss_master_file or extract_dss_archive(args.dss_archive, output)
    print("Preparing feeder and county assignment...", flush=True)
    feeder = prepare_feeder(master, output)
    coordinate_override = load_bus_coordinates(args.bus_coordinates) if args.bus_coordinates else {}
    coordinates = {row["bus"]: coordinate_override.get(row["bus"], (row["x"], row["y"]) if row["has_coordinates"] else None)
                   for row in feeder["coordinates"]}
    geography = build_geography(coordinates, county_geojson=args.county_geojson, bus_county_csv=args.bus_county_csv)
    coordinates = geography.get("bus_coordinates", coordinates)
    if not geography["coverage"]["county_count"]:
        raise ValueError("No modeled buses mapped to a county; supply compatible coordinates and county boundaries.")
    (output / "geography.json").write_text(json.dumps(geography, ensure_ascii=False, indent=2))
    print("County coverage:", geography["coverage"], flush=True)
    branch_weights = json.loads(args.branch_probabilities.read_text()) if args.branch_probabilities else None
    graphs, metadata = generate_risk_dataset(master, output, geography, num_scenarios=args.num_scenarios,
                                             seed=args.seed, load_min=args.load_min, load_max=args.load_max,
                                             voltage_min=args.voltage_min, voltage_max=args.voltage_max,
                                             dns_fraction_threshold=args.dns_threshold,
                                             min_affected_load_fraction=args.affected_load_threshold,
                                             branch_probabilities=branch_weights, problem_type=args.problem_type)
    print(f"Simulation: {len(graphs)}/{metadata['attempted_scenarios']} scenarios resolved", flush=True)
    model_output = None
    if not args.prepare_only and len({g.outage_group for g in graphs}) < 12:
        metadata["model_skipped_reason"] = "GNN training skipped: fewer than 12 distinct resolved outage lines; simulation report is still available."
    if not args.prepare_only and not metadata.get("model_skipped_reason"):
        splits = grouped_split(graphs, args.seed)
        county_mask = torch.tensor([r["kind"] == "county" for r in geography["regions"]])
        for graph in graphs:
            graph.region_valid &= county_mask[None, :]
        mean, scale = normalize_graphs(graphs, splits["train"])
        device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
        device = torch.device(device_name)
        config = dict(input_dim=graphs[0].x.shape[1], edge_dim=graphs[0].edge_attr.shape[1],
                      num_regions=len(geography["regions"]), model_name=args.model_name,
                      hidden_dim=args.hidden_dim, num_layers=args.num_layers, dropout=args.dropout)
        model = RegionalRiskGNN(**config)
        history = train_regional_model(model, graphs, splits, device, args.num_epochs,
                                       args.batch_size, args.lr, args.patience)
        all_logits, all_extent = predict(model, graphs, list(range(len(graphs))), device, args.batch_size)
        labels = torch.cat([g.y for g in graphs])
        extents = torch.cat([g.extent for g in graphs])
        valid = torch.cat([g.region_valid for g in graphs])
        cal = splits["calibration"]
        calibration = fit_calibration(all_logits[cal], labels[cal], valid[cal])
        probs = probabilities(all_logits, calibration)
        test = splits["test"]
        metrics = probability_metrics(labels[test], probs[test], extents[test], all_extent[test], valid[test])
        training_class_counts = [{"county": region["name"],
                                  "positive": int(labels[splits["train"], r].sum()),
                                  "negative": int(len(splits["train"]) - labels[splits["train"], r].sum()),
                                  "valid": bool(valid[splits["train"], r].any())}
                                 for r, region in enumerate(geography["regions"])]
        split_names = [None] * len(graphs)
        for name, indices in splits.items():
            for i in indices:
                split_names[i] = name
        model_report = {"config": config, "split_counts": {name: len(indices) for name, indices in splits.items()},
                        "split_policy": "grouped_by_outage_line; separate tuning, calibration and held-out test sets",
                        "calibration": calibration, "test_metrics": metrics, "training_history": history,
                        "training_class_counts": training_class_counts,
                        "dataset_cache_key": metadata["cache_key"],
                        "summary_scope": "GNN ensemble summaries include training scenarios; held-out test metrics alone measure generalization."}
        torch.save({"state_dict": model.cpu().state_dict(), "config": config,
                    "normalization_mean": mean, "normalization_scale": scale,
                    "calibration": calibration, "county_names": [r["name"] for r in geography["regions"]],
                    "dataset_cache_key": metadata["cache_key"], "splits": splits}, output / "regional_gnn_best.pt")
        model_output = {"probabilities": probs.numpy(), "extent": all_extent.numpy(),
                        "calibration": calibration, "test_metrics": metrics, "split_names": split_names, "report": model_report}
    report = make_reports(graphs, metadata, geography, coordinates, output, model_output)
    (output / "run_config.json").write_text(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2))
    for county in report["counties"]:
        if county["is_county"]:
            print(county["county"], "simulation probability=", county["simulation_probability"],
                  "expected affected load share=", county["expected_affected_load_fraction_resolved"], flush=True)
    print(f"Reports and model saved to {output}", flush=True)
    for warning in report["warnings"]:
        print("Note:", warning, flush=True)
    return report


if __name__ == "__main__":
    main()
