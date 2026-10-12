# Benchmarking GNN datasets for PowerGrids

Traditional methods for simulating cascading failures in power grids are computationally expensive and use historical blackout datasets that are scarce and incomplete. To address this limitation, we propose to use machine learning models, specifically Graph Neural Networks (GNNs), to instantly detect cascading failures from the pre-outage state of the system. The lack of publicly available GNN datasets for power grid applications has motivated us to develop a new graph dataset. This dataset, designed for graph-level tasks and ranging in size from small to medium, fills a gap in the OGB taxonomy for graph datasets. It is tailored to the society domain, where no public GNN graph property prediction datasets are currently available. Furthermore, we provide explainability masks. Currently, no real graph dataset for graph-level application can be used to benchmark explainability models.

With the InMemoryDatasets Class, we generate the GNN datasets for the UK, IEEE24, IEEE39, IEEE118 power grids. We use **InMemoryDataset** class of Pytorch Geometric.

## Installation

**Requirements**

- CPU or NVIDIA GPU, Linux, Python 3.7
- PyTorch >= 1.5.0, other packages

Load every additional packages:

```
pip install -r requirements.txt
```

## Prerequisites and data structure

To reproduce the results presented in the paper, download the following compressed data from [here](https://figshare.com/articles/dataset/PowerGraph/22820534) (~1.8GB, when uncompressed):

```bash
wget -O data.tar.gz "https://figshare.com/ndownloader/files/40571123"
tar -xf data.tar.gz
```

Each dataset folder contains the following files:

- `blist.mat`: branch list also called edge order or edge index
- `of_bi.mat`: binary classification labels
- `of_reg.mat`: regression labels
- `of_mc.mat`: multi-class labels
- `Bf.mat`: node feature matrix
- `Ef.mat`: edge feature matrix
- `exp.mat`: groundtruth explanation

## Dataset description

| Dataset    |     Name     | Description                    |
| ---------- | :----------: | ------------------------------ |
| BA-2motifs | `ba_2motifs` | Random BA graph with 2 motifs. |
| IEEE-24    |   `ieee24`   | IEEE-24 (Powergrid dataset)    |
| IEEE-39    |   `ieee39`   | IEEE-39 (Powergrid dataset)    |
| IEEE-118   |  `ieee118`   | IEEE-118 (Powergrid dataset)   |
| UK         |     `uk`     | UK (Powergrid dataset)         |

We have created a graph dataset that models cascading failure events, which are the main cause of blackouts in power grids. To generate a comprehensive dataset for different power grids, we used a physics-based cascading failure model called Cascades. This model simulates how failures propagate in the IEEE24, IEEE39, IEEE118 and UK power grids. The output of the model is the final demand not being served (DNS). Our dataset consists of a large set of power grid states, representing the operating conditions before an outage, and is linked to an initial triggering outage (one or more failed elements). Each power grid state is represented as a graph, with a graph-level label assigned based on the results of the physics-based model. The dataset is designed for various graph-level tasks, such as multi-class classification, binary classification, and regression. Bus and branches are the elements of a power grid, buses include loads and generators which represent the nodes of the graph, while branches include transmission lines and transformers which represent the edges of the graph. We provide three features per node: net active power, net apparent power and voltage magnitude. While the features per edge are four: active power flow, reactive power flow, line reactance and line rating.

## GNN Benchmarking

To test the datasets with different GNN architectures: GCN, GINe, GAT and Transformer, run,

```
python code/train_gnn.py
```

We have the main arguments to control namely

**--model_name**: transformer / gin / gat

**--datatype**: binary / multiclass / regression

**--dataset_name**: uk / ieee24 / ieee39

We do not use GCN model for the PowerGraph datasets since the graphs have edge attributes that carry meaningful information and should be used by the GNN model for prediction.

Make sure you have the dataset as per format. Models will be saved as per format (make sure you have the model folder)

```
.
├── src
├── dataset
│ ├── processed
│ ├── raw
| | ├── \*.mat
├──model
| ├──ieee24
| ├──ieee39
| ├──uk
```

Remove the for loop in train_gnn.py if running for a specific **--hidden_dim** and **num_layers**.

The models will be saved in **model** directory

### IEEE 9500 N-1 DNS regression

The OpenDSS IEEE 9500 files describe a feeder, not a labeled training set. XAI can generate graph-level training samples by disabling one enabled non-switch line at a time, solving the feeder with OpenDSS, and using the resulting demand not served (DNS, MW) as the regression target. Install the requirements first; the selected Python environment must include PyTorch, PyTorch Geometric, and OpenDSSDirect.py.

Run a single model configuration from the XAI repository root:

```bash
python src/train_gnn.py \
	--dataset_name ieee9500 \
	--task regression \
	--dss_master_file "../9500-Node-Model-Submission-Model-Files/Original DSS Models/Master-unbal-initial-config.dss" \
	--data_save_dir ./data \
	--max_scenarios 128 \
	--model_sweep False \
	--model_name gin \
	--hidden_dim 16 \
	--num_layers 3
```

The DSS master file's companion files are resolved relative to its directory. Generated samples are cached under `./data/ieee9500/processed`; use `--max_scenarios 0` to simulate every eligible line. Scenario generation can take time, and DNS is measured as the difference in solved load terminal power between the base case and the outage case.

### IEEE 9500 County 区域风险概率与影响范围

新增 `src/train_region_risk.py`，复用 `src/gnn/model.py` 中的 GINE、GAT 或 Transformer 消息传递结构，增加县级聚合、问题分类和受影响负荷比例预测。数据转换自动完成：DSS 潮流模型 → 母线/线路图 → 单线路停运场景 → 每个 County 的监督标签 → GNN 训练、概率校准和独立测试。

当前桌面的 `ieee9500bal_csv.zip` 用于读取母线经纬度；线路、变压器、负荷和开关电气模型来自同一版本的 `ieee9500bal_dss.zip`。CSV 本身没有故障结果标签，因此必须配套使用 DSS 或其他已求解场景。默认自动查找桌面 `9500-Node-Model-Submission-Model-Files/Exported Models/ieee9500bal_dss.zip`。使用其他模型时，请明确传入匹配的模型与坐标文件。

首次准备真实县界（[美国人口普查局 2025 县界](https://www.census.gov/geographies/mapping-files/2025/geo/carto-boundary-file.html)，默认 Washington，FIPS 53）：

```bash
.venv/bin/python src/prepare_counties.py
```

然后从项目根目录运行：

```bash
.venv/bin/python src/train_region_risk.py \
  --num-scenarios 128 \
  --model-name gin \
  --num-epochs 50 \
  --output-dir ./outputs/ieee9500_county_risk
```

默认保持当前负荷倍率 `1.0`。所谓概率是 **当前负荷下，假设一条已连接、非开关线路随机停运时，县内所建模电网出现指定问题的条件概率**。默认线路等概率、无放回抽样；不是未来一天的真实设备故障率，也不代表整个县所有电网或用户。`--num-scenarios 0` 枚举全部符合条件的线路。自定义 `--branch-probabilities weights.json` 可提供每条合格 `Line.name` 的非负权重，程序按归一化权重有放回抽样；这些权重仍是单线路停运条件下的分布。

默认 `--problem-type any` 使用以下问题判据：区域失供电比例达到 1%，或受影响负荷比例达到 1%，或区域关联的额定线路出现过载。电压越限默认 `0.95–1.05 pu`；过载按线路 `NormAmps` 检查。可选 `--problem-type dns` 仅看区域失供电、`voltage` 仅看负荷加权电压越限比例、`overload` 仅看线路过载。阈值可通过 `--dns-threshold`、`--affected-load-threshold`、`--voltage-min`、`--voltage-max` 修改，默认值是本实验设置。

输出到指定目录：

| 文件 | 内容 |
| --- | --- |
| `county_risk.csv` / `risk_report.json` | 每县仿真概率、未收敛场景带来的概率上下界、GNN 校准状态、受影响负荷比例；影响至少 1%、5%、10%、25%、50% 负荷的概率 |
| `scenario_predictions.csv` | 每个线路停运场景的区域标签、GNN 分数及影响比例，包含未确定场景 |
| `county_risk.geojson` | 县界和风险结果，可在 GIS 软件中打开 |
| `bus_problem_probability.csv` / `.geojson` | 有负荷母线出现问题的频率与坐标；无负荷母线不标记为“安全” |
| `county_risk_map.png` | 县界与所建模负荷母线风险图 |
| `regional_gnn_best.pt` | 按调参集损失选出的模型、归一化参数、县级校准参数和划分记录 |

影响范围以 **所建模负荷的比例** 衡量，并提供模型覆盖的坐标边界；县面积与母线坐标包围框仅作地理参考，不等于停电影响面积，也不推算未建模用户数量。跨县线路的过载会计入两个端点所属县，线路过载导致的负荷影响比例仅统计有负荷端点，不推断整条下游线路的停电。

`simulation_probability` 是所抽样仿真集合的频率。任意场景未收敛时，该字段为空，改为输出保守上下界；这些上下界只处理未确定场景，**不是抽样置信区间**。样本不足、单一类别或校准样本缺少正负例时，`gnn_probability_resolved` 为空，输出 `gnn_uncalibrated_score` 并说明原因。模型评估按停运线路分组，调参、校准与测试集彼此独立；全场景 GNN 均值包含训练场景，其泛化能力只由测试集的 Brier 分数、校准误差等衡量。

`baseline_already_has_problem` 明确标记不发生线路停运时就满足问题判据的县。这种情况下，综合问题标签可能全为 1，不能据此声称学出了可靠的风险区分能力。可选择具体研究事件（例如 `--problem-type dns`），或提供校验后的当前运行状态；程序不会自行修改电网来消除基准问题。

其他数据源与仅转换数据的示例：

```bash
.venv/bin/python src/train_region_risk.py \
  --dss-master-file /path/to/master.dss \
  --bus-coordinates /path/to/Buscoords.csv \
  --county-geojson /path/to/counties.geojson \
  --problem-type dns \
  --prepare-only
```

也可使用 `--bus-county-csv` 提供 `bus,county[,geoid,longitude,latitude]` 映射。无法归县的母线单列 `Unknown`，不会虚构县名。仿真仅在项目输出目录内的模型副本中运行，原始桌面模型保持不变；处理后的图按模型内容、县界、随机种子及仿真设置缓存。

验证命令：

```bash
.venv/bin/python -m unittest discover -s tests -v
```

## GNN Explainability Benchmarking

**Graph Classification Tasks**

| Non-generative Explainer | Paper                                                                               |
| :----------------------- | :---------------------------------------------------------------------------------- |
| Occlusion                | Visualizing and understanding convolutional networks                                |
| SA                       | Explainability Techniques for Graph Convolutional Networks.                         |
| Grad-CAM                 | Explainability Methods for Graph Convolutional Neural Networks.                     |
| Integrated Gradients     | Axiomatic Attribution for Deep Networks                                             |
| GNNExplainer             | GNNExplainer: Generating Explanations for Graph Neural Networks                     |
| SubgraphX                | On Explainability of Graph Neural Networks via Subgraph Exploration                 |
| PGM-Explainer            | PGM-Explainer: Probabilistic Graphical Model Explanations for Graph Neural Networks |

| Generative Explainer | Paper                                                                             |
| :------------------- | :-------------------------------------------------------------------------------- |
| PGExplainer          | Parameterized Explainer for Graph Neural Network                                  |
| RCExplainer          | Reinforced Causal Explainer for Graph Neural Networks                             |
| GSAT                 | Interpretable and Generalizable Graph Learning via Stochastic Attention Mechanism |
| GraphCFE             | CLEAR: Generative Counterfactual Explanations on Graphs                           |
| DiffExplainer        | D4Explainer (not published)                                                       |
| GflowExplainer       | DAG Matters! GFlowNets Enhanced Explainer For Graph Neural Networks `             |

**Run explainability methods**

```bash
python3 code/main.py --dataset_name [dataset-name] --model_name [gnn-model] --explainer_name [explainer-name]
```

We have the main arguments to control namely

**--model_name**: transformer / gin / gat

**--datatype**: multiclass

**--dataset_name**: uk_mc / ieee24_mc / ieee39_mc

**--explainer_name**: random / sa / ig / gradcam / occlusion / basic_gnnexplainer / gnnexplainer / subgraphx / pgmexplainer / pgexplainer / graphcfe / rcexplainer / gsat / diffexplainer / gflowexplainer

Default args:

**--explained_target**: 0 - Only Category A powergrids are explained (DNS>0 and Cascading failure)

## License

This work is licensed under a CC BY 4.0 license.
