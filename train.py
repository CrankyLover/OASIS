from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from model import AnchorConditionedGNN


@dataclass
class QueryTimeGNNConfig:
    hidden_dim: int = 128
    dropout: float = 0.1
    doc_dim: int = 0
    num_layers: int = 2
    lr: float = 0.005
    weight_decay: float = 1e-4
    training_steps: int = 225
    seed: int = 42
    label_smoothing: float = 0.1
    supcon_weight: float = 0.1
    smooth_weight: float = 0.1
    smooth_mode: str = "unlabeled"
    global_loss: bool = False
    graph_feat_dim: int = 0
    use_edge_weight: bool = False


def compute_edge_weight(
    doc_embeddings: torch.Tensor,
    query_scores: torch.Tensor,
    edge_index: torch.Tensor,
    tau_s: float = 0.1,
    beta: float = 10.0,
    clip_c: float = 0.1,
    eta: float = 0.3,
) -> torch.Tensor:
    """Compute query-aware directed edge weights matching original propagation system."""
    src, dst = edge_index[0], edge_index[1]
    doc_sim = (doc_embeddings[src] * doc_embeddings[dst]).sum(dim=1)
    delta = (query_scores[dst] - query_scores[src]).clamp(-clip_c, clip_c)
    q_min, q_max = query_scores.min(), query_scores.max()
    g = (query_scores - q_min) / (q_max - q_min + 1e-12)
    gate = eta + (1 - eta) * g[dst]
    return torch.exp(doc_sim / tau_s) * torch.exp(beta * delta) * gate


def compute_graph_features(
    doc_embeddings: torch.Tensor,
    anchor_state: torch.Tensor,
    degree: torch.Tensor,
) -> torch.Tensor:
    """Compute [pos_proto_sim, neg_proto_sim, log_degree] per node."""
    pos_mask = (anchor_state == 1)
    neg_mask = (anchor_state == -1)

    pos_sim = torch.zeros(doc_embeddings.size(0), device=doc_embeddings.device)
    neg_sim = torch.zeros(doc_embeddings.size(0), device=doc_embeddings.device)

    if pos_mask.any():
        pos_proto = doc_embeddings[pos_mask].mean(dim=0)
        pos_proto = pos_proto / (pos_proto.norm() + 1e-12)
        pos_sim = doc_embeddings @ pos_proto

    if neg_mask.any():
        neg_proto = doc_embeddings[neg_mask].mean(dim=0)
        neg_proto = neg_proto / (neg_proto.norm() + 1e-12)
        neg_sim = doc_embeddings @ neg_proto

    log_deg = torch.log1p(degree)
    log_deg = log_deg / (log_deg.max() + 1e-12)

    return torch.stack([pos_sim, neg_sim, log_deg], dim=-1)


def build_anchor_state(
    num_nodes: int,
    oracle_labels: Dict[int, bool],
    exclude_nodes: Sequence[int] | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    excluded = set(int(node) for node in (exclude_nodes or []))
    anchor_state = torch.zeros(num_nodes, dtype=torch.long, device=device)
    for node_id, label in oracle_labels.items():
        node_i = int(node_id)
        if node_i in excluded:
            continue
        anchor_state[node_i] = 1 if bool(label) else -1
    return anchor_state


def _target_tensor(
    target_nodes: Sequence[int],
    oracle_labels: Dict[int, bool],
    device: torch.device | str,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    labels = [1.0 if bool(oracle_labels[int(node)]) else 0.0 for node in target_nodes]
    targets = torch.tensor(labels, dtype=torch.float32, device=device)
    if label_smoothing > 0:
        targets = targets * (1.0 - label_smoothing) + 0.5 * label_smoothing
    return targets


def _bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    raw_binary: torch.Tensor | None = None,
) -> torch.Tensor:
    ref = raw_binary if raw_binary is not None else targets
    positives = float(ref.sum().item())
    negatives = float(ref.numel() - positives)
    if positives <= 0.0 or negatives <= 0.0:
        return F.binary_cross_entropy_with_logits(logits, targets)
    pos_weight = torch.tensor(negatives / positives, dtype=torch.float32, device=targets.device)
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)


def _supcon_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    n = features.size(0)
    if n <= 1:
        return torch.tensor(0.0, device=features.device)

    features = F.normalize(features, dim=1)
    sim = features @ features.T / temperature

    label_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
    diag = ~torch.eye(n, dtype=torch.bool, device=features.device)
    pos_mask = label_mask & diag

    logits_max = sim.max(dim=1, keepdim=True).values.detach()
    logits = sim - logits_max
    exp_logits = torch.exp(logits) * diag.float()
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

    pos_count = pos_mask.float().sum(dim=1)
    mean_log_prob = (pos_mask.float() * log_prob).sum(dim=1) / (pos_count + 1e-12)

    valid = pos_count > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=features.device)

    return -mean_log_prob[valid].mean()


def _unlabeled_smoothness(
    logits: torch.Tensor, edge_index: torch.Tensor, oracle_index: torch.Tensor, num_nodes: int,
) -> torch.Tensor:
    oracle_mask = torch.zeros(num_nodes, dtype=torch.bool, device=logits.device)
    oracle_mask[oracle_index] = True
    src, dst = edge_index[0], edge_index[1]
    both_free = (~oracle_mask[src]) & (~oracle_mask[dst])
    frac = both_free.float().mean()
    if frac < 1e-8:
        return torch.tensor(0.0, device=logits.device)
    diff = (logits[src] - logits[dst]) ** 2
    return (both_free.float() * diff).mean() / frac


def train_query_time_gnn(
    data: Data,
    minibatches: Sequence[Sequence[int]],
    oracle_labels: Dict[int, bool],
    config: QueryTimeGNNConfig | None = None,
    device: torch.device | str | None = None,
    doc_embeddings: torch.Tensor | None = None,
    query_scores: torch.Tensor | None = None,
) -> Tuple[torch.nn.Module, List[float]]:
    if config is None:
        config = QueryTimeGNNConfig()
    if len(minibatches) <= 1:
        raise ValueError("at least two mini-batches are required")
    if not oracle_labels:
        raise ValueError("oracle_labels is empty")

    torch.manual_seed(config.seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    data = data.to(device)
    if doc_embeddings is not None:
        doc_embeddings = torch.as_tensor(doc_embeddings, dtype=torch.float32, device=device)
    if query_scores is not None:
        query_scores = torch.as_tensor(query_scores, dtype=torch.float32, device=device).reshape(-1)

    model = AnchorConditionedGNN(
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
        doc_dim=config.doc_dim,
        num_layers=config.num_layers,
        graph_feat_dim=config.graph_feat_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    clean_batches = [[int(n) for n in batch if int(n) in oracle_labels] for batch in minibatches]
    clean_batches = [b for b in clean_batches if b]
    if len(clean_batches) <= 1:
        raise ValueError("not enough non-empty mini-batches after label filtering")

    all_oracle_nodes = list(oracle_labels.keys())
    all_oracle_index = torch.tensor([int(n) for n in all_oracle_nodes], dtype=torch.long, device=device)
    all_oracle_targets = _target_tensor(
        [int(n) for n in all_oracle_nodes], oracle_labels, device, config.label_smoothing)
    all_oracle_binary = _target_tensor(
        [int(n) for n in all_oracle_nodes], oracle_labels, device) if config.label_smoothing > 0 else None
    all_oracle_labels_int = torch.tensor(
        [1 if oracle_labels[int(n)] else 0 for n in all_oracle_nodes],
        dtype=torch.long, device=device,
    )

    history: List[float] = []
    num_nodes = int(data.num_nodes)

    src, dst = data.edge_index[0], data.edge_index[1]
    degree = torch.zeros(num_nodes, dtype=torch.float, device=device)
    degree.scatter_add_(0, src, torch.ones_like(src, dtype=torch.float, device=device))

    edge_weight = (compute_edge_weight(doc_embeddings, query_scores, data.edge_index)
                   if config.use_edge_weight and doc_embeddings is not None and query_scores is not None
                   else None)

    for step in range(config.training_steps):
        target_nodes = clean_batches[step % len(clean_batches)]
        target_index = torch.tensor(target_nodes, dtype=torch.long, device=device)

        anchor_state = build_anchor_state(num_nodes, oracle_labels, target_nodes, device)
        target = _target_tensor(target_nodes, oracle_labels, device, config.label_smoothing)

        graph_feat = (compute_graph_features(doc_embeddings, anchor_state, degree)
                      if config.graph_feat_dim > 0 else None)

        model.train()
        optimizer.zero_grad()
        logits, fused = model(
            anchor_state, data.edge_index,
            doc_embeddings=doc_embeddings, query_scores=query_scores,
            graph_features=graph_feat, edge_weight=edge_weight,
            return_repr=True,
        )
        target_logits = logits[target_index]

        if config.global_loss:
            all_logits = logits[all_oracle_index]
            bce = _bce_loss(all_logits, all_oracle_targets, all_oracle_binary)
        else:
            raw_binary = _target_tensor(target_nodes, oracle_labels, device) if config.label_smoothing > 0 else None
            bce = _bce_loss(target_logits, target, raw_binary)
        loss = bce

        if config.supcon_weight > 0:
            all_repr = fused[all_oracle_index]
            loss = bce + config.supcon_weight * _supcon_loss(all_repr, all_oracle_labels_int)

        if config.smooth_weight > 0:
            sm = _unlabeled_smoothness(logits, data.edge_index, all_oracle_index, num_nodes)
            loss = loss + config.smooth_weight * sm

        loss.backward()
        optimizer.step()
        history.append(float(loss.detach().cpu().item()))

    return model, history
