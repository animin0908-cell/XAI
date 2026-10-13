# IEEE 9500 区域风险修改说明

## 1. 汇总

**输入：** IEEE 9500 电网模型、母线坐标、County 县界，以及负荷状态和假设停运的线路。

**目标：** 复用 XAI 的 GNN，预测每个县内模型电网出现问题的风险，以及受影响的负荷比例。

**概率含义：** 当前模型状态下，假设随机一条线路停运后的区域风险，覆盖范围限于模型中的电网。

## 2. 对数据集和 model 做了哪些修改

**数据集修改：**

- 将 DSS 电网模型转换为图：母线作为节点，线路和变压器连接作为边。
- 根据母线坐标和县界归县。本次 5,302 个母线全部归入 Adams County 和 Franklin County。
- 用 OpenDSS 模拟单条线路停运，生成“是否出问题”和“受影响负荷比例”的训练标签。默认问题包括失供电、电压越限和线路过载。
- 节点输入使用负荷 P/Q、停运前电压和停运线路端点标记；停运后的仿真结果只用于标签。原始桌面数据未修改。

主要新增代码：[数据生成](src/dataset/ieee9500_risk.py)、[母线归县](src/utils/county_geography.py)。

**Model 修改：**

- 原 `src/gnn/model.py` 未修改，复用其 GNN 消息传递结构，本次使用 GINE。
- 新增县级聚合，把原来的整网预测扩展为每个县分别预测。
- 新增两个输出：问题风险分数、受影响负荷比例。区域模型从头训练，没有直接使用原模型权重。
- 增加独立的调参、概率校准和测试流程；标签不足时明确标记为未校准。

主要新增代码：[区域模型](src/gnn/regional_risk.py)、[训练与导出入口](src/train_region_risk.py)。

## 3. 最终输出是什么

| 输出 | 内容 |
| --- | --- |
| [county_risk.csv](outputs/ieee9500_county_risk/county_risk.csv) | 每县风险分数、仿真概率上下界、受影响负荷比例；影响至少 1%、5%、10%、25%、50% 负荷的概率统计 |
| [scenario_predictions.csv](outputs/ieee9500_county_risk/scenario_predictions.csv) | 每个线路停运场景的区域结果 |
| [county_risk_map.png](outputs/ieee9500_county_risk/county_risk_map.png) | 母线问题频率和县界地图，另有 GeoJSON 地理结果 |
| [risk_report.json](outputs/ieee9500_county_risk/risk_report.json) | 训练、测试、校准状态及结果说明 |
| `regional_gnn_best.pt` | 已训练的区域模型及相关参数 |

本次完成 128 个场景和模型训练：119 个场景有效、9 个未收敛，28 项测试通过。结果保存在 `outputs/ieee9500_county_risk/`。

“受影响比例”按模型内的负荷计算，包含越限等问题，不等于停电面积或停电用户比例。影响阈值的概率来自仿真统计。

## 4. 现在有什么问题，以及 SHAP

- **缺少正常样本：** 两个县的基准状态已经达到默认问题判据，119 个有效场景全部标为“有问题”。因此当前 GNN 概率无法可靠校准，输出的是未校准分数。
- **缺少现实数据：** 没有真实故障记录、实际线路故障率和不同时段的运行数据，目前不能直接给出真实的未来停电概率。
- **部分场景未确定：** 9 个场景未收敛，不能当作安全；报告给出这部分不确定性造成的概率上下界。

**SHAP 已接入：** 根目录 `main.py` 可选择 IEEE9500 的县级风险或影响比例，使用统一的 Kernel SHAP 输出特征贡献和包含 feature 名称的中文句子。按节点/边整列特征分组，固定图拓扑，以训练集列均值作为遮蔽参考、端点标记以 0 为参考。合成参考图可能不对应实际运行状态；这是特征组解释，不是逐母线或逐线路解释。若要据此调整负荷或开关来控制输出，需要另加决策逻辑和仿真验证。SHAP 本身不能补齐数据或修正未校准概率。

## 5. 两套模型分离与统一运行

- **IEEE24：** [独立流程](src/workflows/ieee24.py) 加载 PowerGraph 数据和原始整图 GNN，支持分类或 DNS 回归，使用自己的缓存和权重。
- **IEEE9500：** [独立流程](src/workflows/ieee9500.py) 加载原有县级 GNN 及匹配的场景缓存，恢复保存的归一化、校准和测试划分。现有 GNN 结构和已训练权重保持不变。
- **共用 SHAP：** [特征解释器](src/explainer/feature_shap.py) 为两套模型导出相同格式的贡献表、JSON 和中文句子；解释目标及单位由各自模型提供。

```bash
.venv/bin/python main.py --dataset ieee9500 --model gin --target both
.venv/bin/python main.py --dataset ieee24 --model transformer
```

默认输出目录为 `outputs/<dataset>/<model>_<task>_shap/`，含 `predictions.csv`、`shap_features.csv`、`shap_report.json` 和 `explanations.txt`。IEEE24 当前仍需提供真实 PowerGraph 数据才能进行真实预测；IEEE9500 可直接使用已有权重和缓存。

## 6. 面向不同角色的句子

`explanations.txt` 现在按场景和县生成 **For Admin:** 与 **For Asset Manager:** 段落，合并该县的风险、影响比例和主要 SHAP feature。Admin 侧重预测摘要和主要影响因素；Asset Manager 侧重受影响负荷、相关特征和工程复核说明。

通过 `--audiences admin asset_manager engineer executive domain_expert` 可增加 Engineer、Executive、Domain Expert。句子由模板和实际数值生成；概率/比例贡献用百分点表达，未校准风险仍按分数呈现。现有数据未提供天气、设备年龄、历史故障或经济参数，所以不生成对应原因、成本或 ROI，也不把整列特征归因当作关键子图。

角色句子和对应的原始预测/贡献证据另存为 `stakeholder_explanations.json`，也包含在 `shap_report.json` 中。原先逐特征的详细句子保存在 `feature_explanations.txt`，便于审查。
