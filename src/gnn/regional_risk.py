"""County readout on the existing GINE/GAT/Transformer message-passing layers."""

import copy
import numpy as np
import torch
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool

from gnn.model import get_gnnNets


class RegionalRiskGNN(nn.Module):
    def __init__(self, input_dim, edge_dim, num_regions, model_name="gin",
                 hidden_dim=32, num_layers=3, dropout=0.1):
        super().__init__()
        self.num_regions = num_regions
        params = dict(model_name=model_name, edge_dim=edge_dim, hidden_dim=hidden_dim,
                      num_layers=num_layers, dropout=dropout, readout="mean")
        self.backbone = get_gnnNets(input_dim, 1, params, "regression")
        self.head = nn.Sequential(nn.Linear(hidden_dim * 3 + input_dim, hidden_dim),
                                  nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 2))

    def forward(self, data):
        emb = self.backbone.get_emb(data)
        groups = data.batch * self.num_regions + data.region_id
        size = data.num_graphs * self.num_regions
        regional_mean = global_mean_pool(emb, groups, size=size)
        regional_max = global_max_pool(emb, groups, size=size)
        global_mean = global_mean_pool(emb, data.batch).repeat_interleave(self.num_regions, dim=0)
        # Include load and contemplated outage location directly in the regional readout.
        # These are known inputs; no post-outage simulation measurements enter the model.
        regional_inputs = global_mean_pool(data.x, groups, size=size)
        regional_inputs[:, 3:] = global_add_pool(data.x[:, 3:], groups, size=size)
        raw = self.head(torch.cat([regional_mean, regional_max, global_mean, regional_inputs], dim=-1))
        raw = raw.reshape(data.num_graphs, self.num_regions, 2)
        return raw[:, :, 0], torch.sigmoid(raw[:, :, 1])


def grouped_split(graphs, seed=0):
    """Keep repeated simulations of the same failed line in one partition."""
    unique = sorted({g.outage_group for g in graphs})
    if len(unique) < 12:
        raise ValueError("At least 12 distinct outage lines are needed for train/tune/calibration/test splits.")
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n = len(unique)
    n_hold = max(2, int(n * 0.15))
    partitions = [unique[:n - 3 * n_hold], unique[n - 3 * n_hold:n - 2 * n_hold],
                  unique[n - 2 * n_hold:n - n_hold], unique[n - n_hold:]]
    return {name: [i for i, g in enumerate(graphs) if g.outage_group in set(groups)]
            for name, groups in zip(["train", "tune", "calibration", "test"], partitions)}


def normalize_graphs(graphs, train_indices):
    # Fixed node order permits sampling training graphs to avoid concatenating the full dataset.
    values = torch.cat([graphs[i].x for i in train_indices[:32]])
    mean = values.mean(0)
    scale = values.std(0).clamp_min(1e-4)
    # Outage markers must retain their location and 0/1 meaning for the summed readout.
    mean[3:] = 0
    scale[3:] = 1
    for graph in graphs:
        graph.x = (graph.x - mean) / scale
    return mean, scale


@torch.no_grad()
def predict(model, graphs, indices, device, batch_size):
    logits, extents = [], []
    model.eval()
    loader = DataLoader([graphs[i] for i in indices], batch_size=batch_size, shuffle=False)
    for batch in loader:
        raw, extent = model(batch.to(device))
        logits.append(raw.cpu())
        extents.append(extent.cpu())
    return torch.cat(logits), torch.cat(extents)


def train_regional_model(model, graphs, splits, device, num_epochs=50, batch_size=2,
                         lr=0.001, patience=10):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-5)
    loader = DataLoader([graphs[i] for i in splits["train"]], batch_size=batch_size, shuffle=True)
    best_loss = float("inf")
    best_state = None
    stale = 0
    history = []
    tune_y = torch.cat([graphs[i].y for i in splits["tune"]])
    tune_extent = torch.cat([graphs[i].extent for i in splits["tune"]])
    tune_mask = torch.cat([graphs[i].region_valid for i in splits["tune"]]).bool()
    if not tune_mask.any():
        raise ValueError("No county has modeled positive load; check the county assignment.")
    for epoch in range(num_epochs):
        model.train()
        losses = []
        for batch in loader:
            batch = batch.to(device)
            logits, extent = model(batch)
            mask = batch.region_valid.bool()
            loss = nn.functional.binary_cross_entropy_with_logits(logits[mask], batch.y[mask])
            loss = loss + nn.functional.mse_loss(extent[mask], batch.extent[mask])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(loss.item())
        tune_logits, predicted_extent = predict(model, graphs, splits["tune"], device, batch_size)
        tune_loss = (nn.functional.binary_cross_entropy_with_logits(tune_logits[tune_mask], tune_y[tune_mask])
                     + nn.functional.mse_loss(predicted_extent[tune_mask], tune_extent[tune_mask])).item()
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "tune_loss": tune_loss})
        print(f"Epoch {epoch + 1}/{num_epochs}: loss={np.mean(losses):.5f}, tune_loss={tune_loss:.5f}", flush=True)
        if tune_loss < best_loss:
            best_loss, best_state, stale = tune_loss, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return history


def fit_calibration(logits, labels, valid, min_each_class=5):
    """Fit independent Platt maps only on the reserved calibration partition."""
    from sklearn.linear_model import LogisticRegression

    calibration = []
    for r in range(logits.shape[1]):
        mask = valid[:, r].bool()
        x = logits[mask, r].numpy()
        y = labels[mask, r].numpy().astype(int)
        positives, negatives = int(y.sum()), int(len(y) - y.sum())
        status = {"positive_count": positives, "negative_count": negatives,
                  "status": "insufficient_labels", "slope": 1.0, "intercept": 0.0}
        if min(positives, negatives) >= min_each_class:
            fit = LogisticRegression(C=1.0, solver="lbfgs").fit(x[:, None], y)
            status.update(status="calibrated", slope=float(fit.coef_[0, 0]),
                          intercept=float(fit.intercept_[0]))
        calibration.append(status)
    return calibration


def probabilities(logits, calibration):
    slope = logits.new_tensor([v["slope"] for v in calibration])
    intercept = logits.new_tensor([v["intercept"] for v in calibration])
    return torch.sigmoid(logits * slope + intercept)


def probability_metrics(labels, probs, extents, predicted_extents, valid):
    from sklearn.metrics import roc_auc_score

    result = []
    for r in range(labels.shape[1]):
        mask = valid[:, r].bool()
        y, p = labels[mask, r].numpy(), probs[mask, r].numpy()
        if not len(y):
            result.append({"samples": 0})
            continue
        clipped = np.clip(p, 1e-7, 1 - 1e-7)
        bins = []
        ece = 0.0
        bin_ids = np.minimum((p * 10).astype(int), 9)
        for bin_id in range(10):
            select = bin_ids == bin_id
            if select.any():
                confidence, frequency = float(p[select].mean()), float(y[select].mean())
                ece += int(select.sum()) / len(y) * abs(confidence - frequency)
                bins.append({"mean_probability": confidence, "observed_frequency": frequency,
                             "count": int(select.sum())})
        result.append({"samples": len(y), "brier_score": float(np.mean((p - y) ** 2)),
                       "log_loss": float(-np.mean(y * np.log(clipped) + (1 - y) * np.log(1 - clipped))),
                       "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
                       "expected_calibration_error": ece, "reliability_bins": bins,
                       "extent_mae": float((extents[mask, r] - predicted_extents[mask, r]).abs().mean())})
    return result
