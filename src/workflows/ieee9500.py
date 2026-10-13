"""Load the unchanged county-risk GNN and its matched cached IEEE9500 graphs."""

import json

import torch

from gnn.regional_risk import RegionalRiskGNN, probabilities
from workflows.common import ModelContext, Target, IEEE9500_NODE_FEATURES, IEEE9500_EDGE_FEATURES


def load_context(args, device):
    if args.task is not None:
        raise ValueError("--task selects IEEE24 tasks; IEEE9500 uses the saved county-risk model.")
    checkpoint = args.checkpoint or args.regional_run_dir / "regional_gnn_best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"IEEE9500 checkpoint missing: {checkpoint}. Train with src/train_region_risk.py first.")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    required = {"config", "state_dict", "normalization_mean", "normalization_scale", "calibration",
                "county_names", "dataset_cache_key", "splits"}
    if not required <= state.keys():
        raise ValueError("This checkpoint does not belong to the IEEE9500 county-risk workflow.")
    config = state["config"]
    name = config["model_name"]
    if args.model_name is not None and args.model_name != name:
        raise ValueError(f"Selected model {args.model_name} differs from checkpoint model {name}; supply its trained --checkpoint.")
    if config["input_dim"] != 4 or config["edge_dim"] != 4:
        raise ValueError("IEEE9500 feature descriptions require four node features and four edge features.")
    cache_dir = checkpoint.parent / "dataset" / state["dataset_cache_key"]
    graph_path = cache_dir / "graphs.pt"
    metadata_path = cache_dir / "metadata.json"
    if not graph_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Matched IEEE9500 dataset cache is missing: {cache_dir}")
    metadata = json.loads(metadata_path.read_text())
    if metadata["cache_key"] != state["dataset_cache_key"] or [r["name"] for r in metadata["regions"]] != state["county_names"]:
        raise ValueError("IEEE9500 dataset cache and checkpoint have different regions or cache keys.")
    graphs = torch.load(graph_path, map_location="cpu", weights_only=False)
    splits = state["splits"]
    if set(splits) != {"train", "tune", "calibration", "test"} or sorted(sum(splits.values(), [])) != list(range(len(graphs))):
        raise ValueError("IEEE9500 checkpoint splits do not match its cached graphs.")
    mean, scale = state["normalization_mean"], state["normalization_scale"]
    if mean.shape != (4,) or scale.shape != (4,) or not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("IEEE9500 checkpoint contains invalid normalization parameters.")
    for graph in graphs:
        if graph.x.shape[1] != 4 or graph.edge_attr.shape[1] != 4:
            raise ValueError("Cached graph feature dimensions do not match the IEEE9500 model.")
        graph.x = (graph.x.float() - mean) / scale
        graph.edge_attr = graph.edge_attr.float()
        if not torch.isfinite(graph.x).all() or not torch.isfinite(graph.edge_attr).all():
            raise ValueError("IEEE9500 inputs must be finite.")
    model = RegionalRiskGNN(**config).to(device)
    model.load_state_dict(state["state_dict"], strict=True)
    calibration = state["calibration"]
    if len(calibration) != config["num_regions"] or len(state["county_names"]) != config["num_regions"]:
        raise ValueError("IEEE9500 region count does not match checkpoint calibration.")
    targets = []
    for kind in ("risk", "extent"):
        for r, region in enumerate(metadata["regions"]):
            valid = region["kind"] == "county" and bool(metadata["region_valid"][r])
            calibrated = calibration[r]["status"] == "calibrated"
            status = calibration[r]["status"] if kind == "risk" else "model_prediction"
            unit = ("probability" if calibrated else "score") if kind == "risk" else "load_fraction"
            label = ("问题风险概率" if calibrated else "未校准问题风险分数") if kind == "risk" else "受影响负荷比例"
            targets.append(Target(f"{kind}_{r}", f"{region['name']} 的{label}", kind, unit,
                                  region=region["name"], status=status, valid=valid))

    def forward(batch):
        logits, extent = model(batch)
        return torch.cat((probabilities(logits, calibration), extent), dim=1)

    return ModelContext("ieee9500", name, "county_risk", model, graphs, splits, checkpoint,
                        IEEE9500_NODE_FEATURES, IEEE9500_EDGE_FEATURES, tuple(targets), forward,
                        device, mean, scale, metadata)
