"""Shared input and output descriptions for the two model workflows."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
from torch_geometric.data import Batch


@dataclass(frozen=True)
class Feature:
    key: str
    name: str
    kind: str
    unit: str


@dataclass(frozen=True)
class Target:
    key: str
    name: str
    kind: str
    unit: str
    region: str = ""
    status: str = ""
    valid: bool = True


@dataclass
class ModelContext:
    dataset: str
    model_name: str
    task: str
    model: torch.nn.Module
    graphs: list
    splits: dict
    checkpoint: Path
    node_features: tuple
    edge_features: tuple
    targets: tuple
    forward: Callable
    device: torch.device
    node_offset: torch.Tensor | None = None
    node_scale: torch.Tensor | None = None
    metadata: dict = field(default_factory=dict)

    @torch.inference_mode()
    def predict(self, graphs):
        self.model.eval()
        batch = Batch.from_data_list(graphs).to(self.device)
        result = self.forward(batch).detach().cpu()
        if result.shape != (len(graphs), len(self.targets)) or not torch.isfinite(result).all():
            raise ValueError("Model returned non-finite predictions or an unexpected output shape.")
        return result

    def reference(self):
        """Compute feature-column references using the training partition only."""
        indices = self.splits["train"]
        if not indices:
            raise ValueError("SHAP requires a nonempty training partition for its reference.")
        refs = []
        for key in ("x", "edge_attr"):
            width = getattr(self.graphs[indices[0]], key).shape[1]
            total = torch.zeros(width, dtype=torch.float64)
            count = 0
            for i in indices:
                values = getattr(self.graphs[i], key)
                total += values.double().sum(dim=0)
                count += values.shape[0]
            if not count:
                raise ValueError(f"Cannot compute SHAP reference for empty {key}.")
            refs.append((total / count).float())
        # Missing contemplated outages use the known no-marker reference.
        for j, feature in enumerate(self.node_features):
            if feature.key == "outage_endpoint":
                refs[0][j] = 0.0
        return tuple(refs)

    def raw_nodes(self, values):
        if self.node_scale is None:
            return values
        return values * self.node_scale + self.node_offset


IEEE24_NODE_FEATURES = (
    Feature("net_active_power", "节点净有功功率 Pnet", "node", "dataset_native"),
    Feature("net_apparent_power", "节点净视在功率 Snet", "node", "dataset_native"),
    Feature("voltage", "节点电压幅值 V", "node", "dataset_native"),
)
IEEE24_EDGE_FEATURES = (
    Feature("active_flow", "线路有功潮流 Pij", "edge", "dataset_native"),
    Feature("reactive_flow", "线路无功潮流 Qij", "edge", "dataset_native"),
    Feature("reactance", "线路电抗 Xij", "edge", "dataset_native"),
    Feature("rating", "线路额定容量", "edge", "dataset_native"),
)
IEEE9500_NODE_FEATURES = (
    Feature("load_p", "节点有功负荷 P（按全网基准负荷归一化）", "node", "normalized_load"),
    Feature("load_q", "节点无功负荷 Q（按全网基准负荷归一化）", "node", "normalized_load"),
    Feature("voltage", "节点停运前电压", "node", "pu"),
    Feature("outage_endpoint", "假设停运线路端点标记", "node", "binary"),
)
IEEE9500_EDGE_FEATURES = (
    Feature("is_line", "线路类型标记", "edge", "binary"),
    Feature("is_transformer", "变压器类型标记", "edge", "binary"),
    Feature("phases", "支路相数", "edge", "count"),
    Feature("log_rating", "支路额定值 log(1+rating)", "edge", "log_rating"),
)
