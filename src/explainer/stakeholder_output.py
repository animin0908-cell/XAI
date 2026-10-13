"""Render evidence-linked SHAP results as stakeholder-specific sentences."""

import math


AUDIENCES = {
    "admin": "Admin",
    "asset_manager": "Asset Manager",
    "engineer": "Engineer",
    "executive": "Executive",
    "domain_expert": "Domain Expert",
}
DEFAULT_AUDIENCES = ("admin", "asset_manager")


def prediction_sentence(result):
    target = result["target"]
    value = result["prediction"]
    if target["unit"] == "score":
        return f"{target['name']}为 {value:.6g}，该分数未校准，不能按停电概率解读"
    if target["unit"] == "probability":
        status = "（已校准）" if target["status"] == "calibrated" else "（未校准）"
        return f"{target['name']}的模型概率输出为 {value * 100:.2f}%{status}"
    if target["unit"] == "load_fraction":
        return f"{target['name']}预测为所建模负荷的 {value * 100:.2f}%"
    return f"{target['name']}预测为 {value:.6g} {target['unit']}"


def contribution_text(feature, target):
    value = feature["shap_value"]
    if abs(value) <= 1e-12:
        return f"“{feature['name']}”（{feature['key']}）的净贡献接近零"
    direction = "提高" if value > 0 else "降低"
    if target["unit"] in ("probability", "load_fraction"):
        amount = f"约 {abs(value) * 100:.3g} 个百分点"
    elif target["unit"] == "score":
        amount = f"约 {abs(value):.3g} 分数单位"
    else:
        amount = f"约 {abs(value):.3g} {target['unit']}"
    return f"“{feature['name']}”（{feature['key']}）使该模型输出{direction}{amount}"


def dominant_features(result, count):
    return sorted(result["features"], key=lambda row: -abs(row["shap_value"]))[:count]


def driver_sentence(result, count):
    features = dominant_features(result, count)
    if not features or all(abs(f["shap_value"]) <= 1e-12 for f in features):
        return f"相对参考图，{result['target']['name']}的各输入特征净贡献接近零。"
    details = "；".join(contribution_text(feature, result["target"]) for feature in features)
    return f"对{result['target']['name']}，相对训练集参考图的主要特征贡献是：{details}。"


def role_sentence(audience, results):
    prediction = "；".join(prediction_sentence(result) for result in results) + "。"
    primary = next((r for r in results if r["target"]["kind"] == "extent"), results[0])
    if audience == "admin":
        drivers = " ".join(driver_sentence(result, 1) for result in results)
        return prediction + " " + drivers + " 可将这些输入特征列入工程复核清单。"
    if audience == "asset_manager":
        return (prediction + " " + driver_sentence(primary, 2)
                + " 可据此安排对相关输入数据和模型结果的工程复核；具体维护或运行调整需经工程评估。"
                + " 当前未提供停运成本和资产经济参数，本次不估算损失金额或维护 ROI。")
    if audience == "engineer":
        details = []
        for result in results:
            rows = dominant_features(result, 3)
            means = "、".join(f"{f['key']}={f['feature_mean']:.6g} {f['unit']}" for f in rows)
            details.append(f"{result['target']['name']}的参考输出为 {result['base_value']:.6g}，SHAP 贡献合计为 {result['shap_sum']:+.6g}；主要特征的整列均值为 {means}。")
            details.append(driver_sentence(result, 3))
        return (prediction + " " + " ".join(details)
                + " 本次归因为整列输入特征，图拓扑固定，未识别具体故障母线或关键子图。")
    if audience == "executive":
        return (prediction + " " + driver_sentence(primary, 1)
                + " 本次结果可作为工程复核的摘要；现有数据不足以给出投资收益或维护方案排序。")
    return (prediction + " " + driver_sentence(primary, 2)
            + " 请结合原始运行数据复核上述特征及其在图中的分布。"
            + " SHAP 比较的是合成参考图与当前输入，贡献方向不等于调整该特征后的因果效果；天气、设备年龄和历史故障记录未纳入本次模型。")


def render_stakeholders(results, audiences=DEFAULT_AUDIENCES):
    if not audiences or len(set(audiences)) != len(audiences) or any(a not in AUDIENCES for a in audiences):
        raise ValueError("Choose distinct supported stakeholder audiences.")
    groups = {}
    for result in results:
        if not all(math.isfinite(result[k]) for k in ("prediction", "base_value", "shap_sum")):
            raise ValueError("Stakeholder output requires finite model and SHAP values.")
        if any(not math.isfinite(f["shap_value"]) or not math.isfinite(f["feature_mean"]) for f in result["features"]):
            raise ValueError("Stakeholder output requires finite feature values.")
        key = (result["dataset"], result["model"], result["task"], result["sample_index"], result["target"]["region"])
        groups.setdefault(key, []).append(result)
    rendered = []
    for (dataset, model, task, sample, region), items in groups.items():
        evidence = [{"output_key": result["target"]["key"], "target": result["target"]["name"],
                     "prediction": result["prediction"], "unit": result["target"]["unit"],
                     "calibration_status": result["target"]["status"],
                     "features": [{"key": f["key"], "name": f["name"], "shap_value": f["shap_value"]}
                                  for f in result["features"]]} for result in items]
        scope = ("仅覆盖该县内所建模电网的假设单线路停运场景；受影响负荷比例包含失供电、电压或过载问题，不等于停电用户比例。"
                 if dataset == "ieee9500" else "解释所选 IEEE24 场景的类别概率输出或 DNS 回归输出；所选类别的概率不等于模型最终预测类别。")
        rendered.append({"dataset": dataset, "model": model, "task": task, "sample_index": sample,
                         "scenario_id": items[0].get("scenario_id"), "partition": items[0].get("partition"),
                         "outage_line": items[0].get("outage_line"), "county": region, "scope": scope,
                         "audiences": {a: {"label": f"For {AUDIENCES[a]}", "sentence": role_sentence(a, items),
                                           "evidence": evidence} for a in audiences}})
    return {"schema_version": 1, "audiences": list(audiences),
            "scope": "角色句子来自模型预测与 SHAP 特征归因，描述相对合成参考图的模型输出变化。",
            "missing_inputs": ["weather", "asset_age", "historical_failures", "outage_cost", "roi"],
            "groups": rendered}


def stakeholder_text(report):
    lines = [report["scope"]]
    for group in report["groups"]:
        heading = f"{group['dataset']} / {group['model']} / 样本 {group['sample_index']}"
        if group["county"]:
            heading += f" / {group['county']}"
        if group["partition"]:
            heading += f" / {group['partition']}"
        if group["outage_line"]:
            heading += f" / 假设 {group['outage_line']} 停运"
        lines.append("\n" + heading)
        lines.append(group["scope"])
        for audience in group["audiences"].values():
            lines.append(f"\n{audience['label']}: {audience['sentence']}")
    return "\n".join(lines) + "\n"
