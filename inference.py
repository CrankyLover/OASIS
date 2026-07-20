from typing import Dict, Sequence

import numpy as np
import torch
from torch_geometric.data import Data

from model import AnchorConditionedGNN
from train import build_anchor_state, compute_edge_weight, compute_graph_features


def _chunk_list(items: list[int], chunk_size: int) -> list[list[int]]:
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


@torch.no_grad()
def predict_all(
    model: AnchorConditionedGNN,
    data: Data,
    oracle_labels: Dict[int, bool],
    device: torch.device | str | None = None,
    doc_embeddings: torch.Tensor | np.ndarray | None = None,
    query_scores: torch.Tensor | np.ndarray | None = None,
    temperature: float = 2.0,
    unlabeled_temperature: float | None = None,
    return_logits: bool = False,
    oof_anchor_scores: bool = True,
    oracle_batches: Sequence[Sequence[int]] | None = None,
    oof_batch_size: int = 128,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Predict all nodes.

    For non-anchor nodes: use all oracle labels as anchors.
    For anchor nodes: if oof_anchor_scores=True, use out-of-fold prediction:
    when predicting an anchor batch, exclude that batch from anchor_state.
    This prevents anchor nodes from directly seeing their own labels during inference.
    """
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)
        model = model.to(device)

    data = data.to(device)
    num_nodes = int(data.num_nodes)

    if doc_embeddings is not None:
        doc_embeddings = torch.as_tensor(doc_embeddings, dtype=torch.float32, device=device)
    if query_scores is not None:
        query_scores = torch.as_tensor(query_scores, dtype=torch.float32, device=device).reshape(-1)

    model.eval()

    degree = None
    if model.graph_feat_dim > 0 and doc_embeddings is not None:
        src = data.edge_index[0]
        degree = torch.zeros(num_nodes, dtype=torch.float, device=device)
        degree.scatter_add_(0, src, torch.ones_like(src, dtype=torch.float, device=device))

    edge_weight = (compute_edge_weight(doc_embeddings, query_scores, data.edge_index)
                   if doc_embeddings is not None and query_scores is not None else None)

    # 1. Full-anchor prediction (used for all non-anchor nodes).
    full_anchor_state = build_anchor_state(num_nodes, oracle_labels, device=device)

    full_graph_feat = None
    if model.graph_feat_dim > 0 and doc_embeddings is not None:
        full_graph_feat = compute_graph_features(doc_embeddings, full_anchor_state, degree)

    full_logits = model(
        full_anchor_state, data.edge_index,
        doc_embeddings=doc_embeddings, query_scores=query_scores,
        graph_features=full_graph_feat, edge_weight=edge_weight,
    )

    final_logits = full_logits.clone()

    # 2. Replace anchor-node logits with out-of-fold logits.
    oracle_nodes = [int(n) for n in oracle_labels.keys()]

    if oof_anchor_scores and oracle_nodes:
        if oracle_batches is None:
            oracle_batches = _chunk_list(oracle_nodes, oof_batch_size)

        for batch in oracle_batches:
            batch = [int(n) for n in batch if int(n) in oracle_labels]
            if not batch:
                continue

            batch_index = torch.tensor(batch, dtype=torch.long, device=device)

            oof_anchor_state = build_anchor_state(
                num_nodes, oracle_labels, exclude_nodes=batch, device=device)

            oof_graph_feat = None
            if model.graph_feat_dim > 0 and doc_embeddings is not None:
                oof_graph_feat = compute_graph_features(doc_embeddings, oof_anchor_state, degree)

            oof_logits = model(
                oof_anchor_state, data.edge_index,
                doc_embeddings=doc_embeddings, query_scores=query_scores,
                graph_features=oof_graph_feat, edge_weight=edge_weight,
            )

            final_logits[batch_index] = oof_logits[batch_index]

    # 3. Temperature scaling.
    if unlabeled_temperature is not None:
        node_temp = torch.full((num_nodes,), float(temperature), dtype=torch.float32, device=device)
        oracle_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        if oracle_nodes:
            oracle_index = torch.tensor(oracle_nodes, dtype=torch.long, device=device)
            oracle_mask[oracle_index] = True

        if oof_anchor_scores:
            node_temp[:] = float(unlabeled_temperature)
        else:
            node_temp[~oracle_mask] = float(unlabeled_temperature)

        probabilities = torch.sigmoid(final_logits / node_temp)
    else:
        probabilities = torch.sigmoid(final_logits / temperature)

    probabilities = probabilities.detach().cpu().numpy().astype(np.float32)
    raw_logits = final_logits.detach().cpu().numpy().astype(np.float32)

    if return_logits:
        return probabilities, raw_logits
    return probabilities
