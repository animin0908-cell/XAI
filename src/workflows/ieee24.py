"""Original graph-level IEEE24 models, data and checkpoint lifecycle."""

from pathlib import Path

import torch
from torch.nn import functional as F
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader

import dataset.powergrid as powergrid
from gnn.model import get_gnnNets
from workflows.common import ModelContext, Target, IEEE24_NODE_FEATURES, IEEE24_EDGE_FEATURES


class IEEE24Dataset(powergrid.PowerGrid):
    @property
    def processed_dir(self):
        return str(Path(self.root) / self.name / "processed" / self.task)

    def __init__(self, root, task):
        self.name = "ieee24"
        self.task = task
        self.raw_path = str(root / self.name / "raw")
        powergrid.device = torch.device("cpu")
        InMemoryDataset.__init__(self, str(root))
        self.data, self.slices = torch.load(self.processed_paths[0], map_location="cpu", weights_only=False)


def load_context(args, device):
    name = args.model_name or "transformer"
    task = args.task or "multi_classification"
    root = args.data_save_dir.expanduser().resolve()
    if root.name.lower() == "ieee24":
        root = root.parent
    cache = root / "ieee24/processed" / task / "data.pt"
    if not cache.is_file():
        raw_dir = root / "ieee24/raw"
        names = ("Bf.mat", "blist.mat", "Ef.mat", "exp.mat", "of_bi.mat", "of_mc.mat", "of_reg.mat")
        missing = [n for n in names if not (raw_dir / n).is_file()]
        if missing:
            raise FileNotFoundError(f"IEEE24 raw data missing in {raw_dir}: {', '.join(missing)}. See README.md.")
    dataset = IEEE24Dataset(root, task)
    graphs = [dataset[i] for i in range(len(dataset))]
    if not graphs:
        raise ValueError("IEEE24 dataset is empty.")
    for graph in graphs:
        graph.x = graph.x.float()
        graph.edge_attr = graph.edge_attr.float()
        if graph.y.numel() != 1 or not torch.isfinite(graph.y).all():
            raise ValueError("IEEE24 requires one finite graph-level label per sample.")
        if task != "regression" and not torch.equal(graph.y, graph.y.round()):
            raise ValueError("IEEE24 classification labels must be integer class indices.")
        graph.y = graph.y.reshape(-1).float() if task == "regression" else graph.y.reshape(-1).long()
        if graph.x.shape[1] != 3 or graph.edge_attr.shape[1] != 4:
            raise ValueError("IEEE24 requires three node features and four edge features.")
        if not torch.isfinite(graph.x).all() or not torch.isfinite(graph.edge_attr).all() or not torch.isfinite(graph.y).all():
            raise ValueError("IEEE24 inputs and labels must be finite.")
    output_dim = {"regression": 1, "binary_classification": 2, "multi_classification": 4}[task]
    if task != "regression" and any(g.y.numel() != 1 or not 0 <= g.y.item() < output_dim for g in graphs):
        raise ValueError("IEEE24 labels do not match the selected classification task.")
    params = {key: getattr(args, key) for key in ("hidden_dim", "num_layers", "readout", "dropout")}
    params.update(model_name=name, edge_dim=4)
    model = get_gnnNets(3, output_dim, params, task).to(device)
    order = torch.randperm(len(graphs), generator=torch.Generator().manual_seed(args.seed)).tolist()
    n_train, n_val = int(len(graphs) * args.train_ratio), int(len(graphs) * args.val_ratio)
    splits = {"train": order[:n_train], "val": order[n_train:n_train + n_val], "test": order[n_train + n_val:]}
    if any(not indices for indices in splits.values()):
        raise ValueError("Dataset is too small for the requested train/validation/test ratios.")
    checkpoint = args.checkpoint or (args.model_save_dir / "ieee24/graph" / f"{name}_{task}_{args.num_layers}l_{args.hidden_dim}h_best.pth")
    if not checkpoint.is_file():
        if args.predict_only or args.checkpoint is not None:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        train = DataLoader([graphs[i] for i in splits["train"]], batch_size=args.batch_size, shuffle=True)
        val = DataLoader([graphs[i] for i in splits["val"]], batch_size=args.batch_size)
        best = float("inf")

        def loss_fn(pred, label):
            return F.mse_loss(pred.reshape(-1), label) if task == "regression" else F.nll_loss(pred, label)

        for epoch in range(args.num_epochs):
            model.train()
            for batch in train:
                batch = batch.to(device)
                optimizer.zero_grad()
                loss = loss_fn(model(data=batch), batch.y)
                loss.backward()
                torch.nn.utils.clip_grad_value_(model.parameters(), 2.0)
                optimizer.step()
            model.eval()
            total = 0.0
            with torch.inference_mode():
                for batch in val:
                    batch = batch.to(device)
                    total += loss_fn(model(data=batch), batch.y).item() * batch.num_graphs
            val_loss = total / n_val
            print(f"Epoch {epoch + 1}/{args.num_epochs}: validation loss={val_loss:.6f}", flush=True)
            if val_loss < best:
                best = val_loss
                torch.save({"net": model.state_dict(), "dataset_name": "ieee24", "model_params": params,
                            "task": task, "seed": args.seed, "splits": splits}, checkpoint)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    if "state_dict" in state or state.get("dataset_name", "ieee24") != "ieee24":
        raise ValueError("This checkpoint does not belong to the IEEE24 graph-level workflow.")
    if state.get("task", task) != task or state.get("model_params", params) != params:
        raise ValueError("Checkpoint task or architecture does not match the supplied IEEE24 model arguments.")
    if "splits" in state:
        saved = state["splits"]
        if state["seed"] != args.seed or set(saved) != set(splits) or sorted(sum(saved.values(), [])) != list(range(len(graphs))):
            raise ValueError("Checkpoint split does not match the dataset or --seed.")
        splits = saved
    else:
        print("Legacy checkpoint: use the original training seed and split ratios for held-out evaluation.", flush=True)
    model.load_state_dict(state.get("net", state), strict=True)

    if task == "regression":
        targets = (Target("dns_mw", "未供电负荷 DNS", "dns", "MW"),)
    elif task == "binary_classification":
        targets = tuple(Target(f"class_{i}", text, "class", "probability", status="uncalibrated")
                        for i, text in enumerate(("存在未供电负荷（DNS > 0）", "无未供电负荷（DNS = 0）")))
    else:
        names = ("A 类：失供电且发生级联故障", "B 类：失供电但无级联故障",
                 "C 类：无失供电但发生级联故障", "D 类：无失供电且无级联故障")
        targets = tuple(Target(f"class_{i}", text, "class", "probability", status="uncalibrated")
                        for i, text in enumerate(names))

    def forward(batch):
        output = model(data=batch)
        return output if task == "regression" else output.exp()

    return ModelContext("ieee24", name, task, model, graphs, splits, checkpoint,
                        IEEE24_NODE_FEATURES, IEEE24_EDGE_FEATURES, targets, forward, device)
