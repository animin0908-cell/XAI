"""One grouped-feature Kernel SHAP engine for graph and county models."""

import csv
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from explainer.stakeholder_output import DEFAULT_AUDIENCES, render_stakeholders, stakeholder_text


SCOPE = (
    "每个 SHAP 特征组是一整列节点或边输入特征；图拓扑和县归属固定。"
    "被遮蔽的列替换为训练集列均值，停运端点标记替换为 0；由此合成的参考图可能不是实际运行状态。"
    "贡献解释模型输出相对该参考图的变化，不是单个母线或线路的贡献，也不代表因果控制效果。"
)

OUTPUT_UNITS = {"probability": "概率值", "score": "分数", "load_fraction": "负荷比例", "MW": "MW"}


def masked_predictions(context, graph, masks, reference, target_indices, batch_size):
    """Replace entire feature columns while retaining the sample's topology."""
    masks = np.asarray(masks, dtype=np.float32)
    width = graph.x.shape[1] + graph.edge_attr.shape[1]
    if masks.ndim != 2 or masks.shape[1] != width or not np.isfinite(masks).all():
        raise ValueError("Feature masks have an invalid shape or non-finite values.")
    node_reference, edge_reference = reference
    node_width = graph.x.shape[1]
    results = []
    for start in range(0, len(masks), batch_size):
        graphs = []
        for row in masks[start:start + batch_size]:
            masked = graph.clone()
            weights = torch.from_numpy(row)
            masked.x = node_reference + weights[:node_width] * (graph.x - node_reference)
            masked.edge_attr = edge_reference + weights[node_width:] * (graph.edge_attr - edge_reference)
            graphs.append(masked)
        results.append(context.predict(graphs)[:, target_indices].numpy())
    return np.concatenate(results, axis=0)


def contribution_sentence(feature, target, contribution):
    direction = "推动输出上升" if contribution > 1e-12 else "推动输出下降" if contribution < -1e-12 else "对输出的净贡献接近零"
    return (f"相对训练集参考图，特征“{feature.name}”（{feature.key}）对“{target.name}”"
            f"的 SHAP 贡献为 {contribution:+.6g} {OUTPUT_UNITS.get(target.unit, target.unit)}，{direction}。")


def explain_features(context, sample_index, target_indices, reference, nsamples=256,
                     batch_size=2, seed=0):
    import shap

    graph = context.graphs[sample_index]
    features = context.node_features + context.edge_features
    width = len(features)
    if len(set(target_indices)) != len(target_indices) or not target_indices:
        raise ValueError("Choose at least one distinct explanation target.")
    if nsamples < 2 * width:
        raise ValueError(f"--shap_nsamples must be at least {2 * width} for {width} feature groups.")

    def predict_masks(masks):
        return masked_predictions(context, graph, masks, reference, target_indices, batch_size)

    # Kernel SHAP sees binary group-presence masks, not flattened graph entities.
    explainer = shap.KernelExplainer(predict_masks, np.zeros((1, width)),
                                    feature_names=[f.key for f in features], link="identity")
    previous_state = np.random.get_state()
    try:
        np.random.seed(seed)
        values = explainer.shap_values(np.ones((1, width)), nsamples=nsamples,
                                      l1_reg=0.0, silent=True)
    finally:
        np.random.set_state(previous_state)
    if isinstance(values, list):
        values = np.stack(values, axis=-1)
    values = np.asarray(values).reshape(1, width, len(target_indices))[0]
    base = np.asarray(explainer.expected_value).reshape(-1)
    output = context.predict([graph])[0, target_indices].numpy()
    node_reference, edge_reference = reference
    feature_means = torch.cat((context.raw_nodes(graph.x).mean(0), graph.edge_attr.mean(0))).numpy()
    raw_reference = torch.cat((context.raw_nodes(node_reference), edge_reference)).numpy()
    results = []
    for j, index in enumerate(target_indices):
        target = context.targets[index]
        error = float(output[j] - base[j] - values[:, j].sum())
        if not np.isfinite(values[:, j]).all() or not np.isfinite(base[j]) or abs(error) > 1e-5 * max(1.0, abs(float(output[j]))):
            raise ValueError("SHAP contributions are non-finite or fail the additivity check.")
        rows = []
        order = np.argsort(-np.abs(values[:, j]), kind="stable")
        for rank, position in enumerate(order, start=1):
            feature = features[position]
            contribution = float(values[position, j])
            rows.append({**asdict(feature), "rank": rank,
                         "feature_mean": float(feature_means[position]),
                         "reference_value": float(raw_reference[position]),
                         "shap_value": contribution,
                         "sentence": contribution_sentence(feature, target, contribution)})
        results.append({"dataset": context.dataset, "model": context.model_name,
                        "task": context.task, "sample_index": sample_index,
                        "scenario_id": int(graph.scenario_id.item()) if hasattr(graph, "scenario_id") else None,
                        "partition": next((name for name, indices in context.splits.items() if sample_index in indices), None),
                        "outage_line": getattr(graph, "outage_group", None),
                        "target": asdict(target), "prediction": float(output[j]),
                        "base_value": float(base[j]), "shap_sum": float(values[:, j].sum()),
                        "additivity_error": error, "features": rows})
    return results


def write_explanations(context, results, reference, output_dir, nsamples, seed,
                       audiences=DEFAULT_AUDIENCES):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "method": "KernelSHAP", "link": "identity",
              "shap_nsamples": nsamples, "seed": seed, "checkpoint": str(context.checkpoint),
              "scope": SCOPE, "reference_policy": "training_column_mean; outage_endpoint=0",
              "reference_train_indices": context.splits["train"],
              "reference_model_input": {"node": reference[0].tolist(), "edge": reference[1].tolist()},
              "explanations": results}
    stakeholders = render_stakeholders(results, audiences)
    report["stakeholder_output"] = stakeholders
    (output_dir / "shap_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    rows, sentences = [], [SCOPE]
    for result in results:
        target = result["target"]
        sentences.append(f"\n{result['dataset']} / {result['model']} / 样本 {result['sample_index']} / {target['name']}")
        unit = OUTPUT_UNITS.get(target["unit"], target["unit"])
        sentences.append(f"参考输出 {result['base_value']:.6g} + SHAP 贡献合计 {result['shap_sum']:+.6g} = 预测 {result['prediction']:.6g} {unit}。")
        for feature in result["features"]:
            rows.append({"dataset": result["dataset"], "model": result["model"], "task": result["task"],
                         "sample_index": result["sample_index"], "scenario_id": result["scenario_id"],
                         "output_key": target["key"], "target": target["name"], "output_unit": target["unit"],
                         "calibration_status": target["status"], "base_value": result["base_value"],
                         "prediction": result["prediction"], "additivity_error": result["additivity_error"],
                         **feature})
            sentences.append(feature["sentence"])
    with (output_dir / "shap_features.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "feature_explanations.txt").write_text("\n".join(sentences) + "\n", encoding="utf-8")
    (output_dir / "explanations.txt").write_text(stakeholder_text(stakeholders), encoding="utf-8")
    (output_dir / "stakeholder_explanations.json").write_text(json.dumps(stakeholders, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return report
