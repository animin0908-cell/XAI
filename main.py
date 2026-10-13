"""Select a dataset and its model, then predict and explain with shared SHAP."""

import argparse
import csv
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_name", "--dataset", choices=["ieee24", "ieee9500"], default="ieee24")
    parser.add_argument("--model_name", "--model", choices=["gin", "gat", "transformer"],
                        help="IEEE24 defaults to Transformer; IEEE9500 reads its checkpoint architecture.")
    parser.add_argument("--list_models", action="store_true")
    parser.add_argument("--data_save_dir", type=Path, default=ROOT / "data")
    parser.add_argument("--model_save_dir", type=Path, default=ROOT / "model")
    parser.add_argument("--regional_run_dir", type=Path, default=ROOT / "outputs/ieee9500_county_risk")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--output", type=Path, help="Optional custom path for predictions.csv.")
    parser.add_argument("--task", choices=["binary_classification", "multi_classification", "regression"],
                        help="IEEE24 only; defaults to multi_classification.")
    parser.add_argument("--hidden_dim", type=int, default=16, help="IEEE24 hidden dimension; IEEE9500 uses its checkpoint config.")
    parser.add_argument("--num_layers", type=int, default=3, help="IEEE24 layers; IEEE9500 uses its checkpoint config.")
    parser.add_argument("--readout", choices=["mean", "sum", "max"], default="max", help="IEEE24 graph pooling.")
    parser.add_argument("--dropout", type=float, default=0.1, help="IEEE24 dropout; IEEE9500 uses its checkpoint config.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.00005)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train_ratio", type=float, default=0.85)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--predict_split", choices=["test", "all"], default="test")
    parser.add_argument("--predict_only", action="store_true", help="Require an existing IEEE24 checkpoint.")
    parser.add_argument("--sample_index", type=int, help="Predict and explain one cached graph by its index.")
    parser.add_argument("--explain_count", type=int, default=1, help="Number of predicted samples to explain.")
    parser.add_argument("--target", choices=["risk", "extent", "both"], default="risk", help="IEEE9500 output to explain.")
    parser.add_argument("--county", help="IEEE9500 county name; defaults to every county with modeled load.")
    parser.add_argument("--target_class", type=int, help="IEEE24 class to explain; defaults to the predicted class.")
    parser.add_argument("--audiences", nargs="+", choices=["admin", "asset_manager", "engineer", "executive", "domain_expert"],
                        default=["admin", "asset_manager"], help="Stakeholder sentences to export.")
    parser.add_argument("--shap_nsamples", type=int, default=256)
    parser.add_argument("--shap_batch_size", type=int, default=2)
    parser.add_argument("--skip_shap", action="store_true", help="Export predictions only.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    for name in ("batch_size", "num_epochs", "hidden_dim", "num_layers", "explain_count", "shap_batch_size", "threads"):
        if getattr(args, name) < 1:
            parser.error(f"--{name} must be positive.")
    if not (0 < args.train_ratio < 1 and 0 < args.val_ratio < 1 and args.train_ratio + args.val_ratio < 1):
        parser.error("Training and validation ratios must be positive and sum to less than 1.")
    if not 0 <= args.dropout < 1 or not math.isfinite(args.lr) or args.lr <= 0 or not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("Require dropout in [0,1), positive learning rate and nonnegative weight decay.")
    if args.sample_index is not None and args.sample_index < 0:
        parser.error("--sample_index must be nonnegative.")
    if not args.skip_shap and args.shap_nsamples < (16 if args.dataset_name == "ieee9500" else 14):
        parser.error("Too few SHAP samples for the dataset's feature groups.")
    if len(set(args.audiences)) != len(args.audiences):
        parser.error("--audiences must contain distinct roles.")
    return args


def select_targets(context, prediction, args):
    if context.dataset == "ieee24":
        if args.county is not None:
            raise ValueError("--county is only valid for IEEE9500.")
        if context.task == "regression":
            if args.target_class is not None:
                raise ValueError("--target_class is not valid for DNS regression.")
            return [0]
        index = int(prediction.argmax()) if args.target_class is None else args.target_class
        if not 0 <= index < len(context.targets):
            raise ValueError("--target_class is outside the model's class range.")
        return [index]
    if args.target_class is not None:
        raise ValueError("--target_class is only valid for IEEE24 classification.")
    selected = [i for i, target in enumerate(context.targets)
                if target.valid and (args.target == "both" or target.kind == args.target)
                and (args.county is None or target.region.casefold() == args.county.casefold())]
    if not selected:
        raise ValueError("No valid county output selected; check --county against the saved county names.")
    return selected


def main(args):
    if args.list_models:
        print("ieee24: gin / gat / transformer; original graph GNN; binary / multiclass / DNS")
        print("ieee9500: gin / gat / transformer; county GNN; risk / affected-load fraction")
        print("IEEE9500 requires a trained checkpoint and its matching dataset cache for the chosen model.")
        return
    import numpy as np
    import torch

    if not args.skip_shap:
        try:
            import shap  # noqa: F401
        except ImportError as error:
            raise ImportError("Install SHAP in this environment: pip install 'shap>=0.46,<1'.") from error
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device_name = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    device = torch.device(device_name)
    if args.dataset_name == "ieee24":
        from workflows.ieee24 import load_context
    else:
        from workflows.ieee9500 import load_context
    context = load_context(args, device)
    indices = list(range(len(context.graphs))) if args.predict_split == "all" else context.splits["test"]
    if args.sample_index is not None:
        if args.sample_index >= len(context.graphs):
            raise ValueError("--sample_index is outside the cached dataset.")
        indices = [args.sample_index]
    if not indices:
        raise ValueError("The selected prediction partition is empty.")
    output = args.output_dir or ROOT / "outputs" / context.dataset / f"{context.model_name}_{context.task}_shap"
    output.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output or output / "predictions.csv"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    partitions = {i: name for name, group in context.splits.items() for i in group}
    predictions, rows = {}, []
    for start in range(0, len(indices), args.batch_size):
        current = indices[start:start + args.batch_size]
        values = context.predict([context.graphs[i] for i in current])
        for index, scores in zip(current, values):
            # Validate the requested output before writing any result files.
            select_targets(context, scores, args)
            predictions[index] = scores
            graph = context.graphs[index]
            for column, target in enumerate(context.targets):
                if not target.valid:
                    continue
                if context.dataset == "ieee24":
                    truth = float(graph.y[0]) if context.task == "regression" else float(int(graph.y[0]) == column)
                else:
                    region_index = column % len(context.metadata["regions"])
                    truth = float((graph.y if target.kind == "risk" else graph.extent)[0, region_index])
                rows.append({"dataset": context.dataset, "model": context.model_name, "task": context.task,
                             "sample_index": index, "partition": partitions[index],
                             "scenario_id": int(graph.scenario_id.item()) if hasattr(graph, "scenario_id") else None,
                             "output_key": target.key, "target": target.name, "county": target.region,
                             "prediction": float(scores[column]), "true_value": truth, "unit": target.unit,
                             "calibration_status": target.status})
    with prediction_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Dataset: {context.dataset}; model: {context.model_name}; checkpoint: {context.checkpoint}", flush=True)
    print(f"Predictions: {prediction_path} ({len(indices)} graphs)", flush=True)
    if context.dataset == "ieee24":
        truth = torch.cat([context.graphs[i].y for i in indices])
        values = torch.stack([predictions[i] for i in indices])
        metric = (f"MAE={(values[:, 0] - truth).abs().mean().item():.6f}" if context.task == "regression"
                  else f"accuracy={(values.argmax(1) == truth).float().mean().item():.6f}")
        print(f"Prediction partition: {args.predict_split}; {metric}", flush=True)
    if args.skip_shap:
        return context
    from explainer.feature_shap import explain_features, write_explanations
    reference = context.reference()
    results = []
    for index in indices[:args.explain_count]:
        print(f"Explaining sample {index} with Kernel SHAP...", flush=True)
        targets = select_targets(context, predictions[index], args)
        results.extend(explain_features(context, index, targets, reference, args.shap_nsamples,
                                        args.shap_batch_size, args.seed))
    report = write_explanations(context, results, reference, output, args.shap_nsamples, args.seed, args.audiences)
    print(f"SHAP feature table, JSON and stakeholder sentences: {output}", flush=True)
    return report


if __name__ == "__main__":
    main(parse_args())
