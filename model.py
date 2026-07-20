import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax as pyg_softmax
from typing import Optional


class MeanNeighborAggregator(MessagePassing):
    def __init__(self):
        super().__init__(aggr="add", flow="source_to_target")

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.propagate(edge_index=edge_index, x=x, edge_weight=edge_weight)
        if edge_weight is not None:
            norm = torch.zeros(x.size(0), dtype=torch.float, device=x.device)
            norm.scatter_add_(0, edge_index[1], edge_weight)
            out = out / (norm.view(-1, 1) + 1e-12)
        else:
            deg = torch.zeros(x.size(0), dtype=torch.float, device=x.device)
            deg.scatter_add_(0, edge_index[1], torch.ones(edge_index.size(1), device=x.device))
            out = out / (deg.view(-1, 1) + 1e-12)
        return out

    def message(self, x_j: torch.Tensor, edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        if edge_weight is not None:
            return x_j * edge_weight.view(-1, 1)
        return x_j


class AttentionAggregator(MessagePassing):
    """GAT-style attention aggregation: learns per-edge attention coefficients."""

    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__(aggr="add", flow="source_to_target")
        self.att_src = nn.Parameter(torch.empty(hidden_dim))
        self.att_dst = nn.Parameter(torch.empty(hidden_dim))
        nn.init.xavier_uniform_(self.att_src.view(1, -1))
        nn.init.xavier_uniform_(self.att_dst.view(1, -1))
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        alpha_src = (x * self.att_src).sum(dim=-1)
        alpha_dst = (x * self.att_dst).sum(dim=-1)
        e = F.leaky_relu(alpha_src[src] + alpha_dst[dst], negative_slope=0.2)
        alpha = pyg_softmax(e, dst, num_nodes=x.size(0))
        alpha = self.attn_dropout(alpha)
        return self.propagate(edge_index, x=x, edge_weight=alpha)

    def message(self, x_j: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        return x_j * edge_weight.view(-1, 1)


class AnchorConditionedGNN(nn.Module):
    """
    Anchor-conditioned GNN with query-gated document embedding + graph features.

    Node initial representation:
        x0 = state_embed(anchor_state)
           + query_score * doc_proj(doc_embedding)
           + graph_feat_proj([pos_proto_sim, neg_proto_sim, log_degree])

    GNN layers propagate with attention-weighted aggregation + LayerNorm + residual:
        x = LayerNorm(attn_aggregate(x, edge_index)) + x0

    Three temporal branches (early/mid/late) are fused for the final logit.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        doc_dim: int = 0,
        num_layers: int = 2,
        graph_feat_dim: int = 0,
    ):
        super().__init__()
        self.doc_dim = int(doc_dim)
        self.num_layers = max(2, int(num_layers))
        self.graph_feat_dim = int(graph_feat_dim)

        self.state_embedding = nn.Embedding(3, hidden_dim)
        if self.doc_dim > 0:
            self.doc_proj = nn.Linear(doc_dim, hidden_dim)
        if self.graph_feat_dim > 0:
            self.graph_feat_proj = nn.Linear(graph_feat_dim, hidden_dim)

        self.conv_layers = nn.ModuleList([
            AttentionAggregator(hidden_dim, dropout) for _ in range(self.num_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(self.num_layers)
        ])

        self.branch_early = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
        )
        self.branch_mid = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
        )
        self.branch_late = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        anchor_state: torch.Tensor,
        edge_index: torch.Tensor,
        doc_embeddings: Optional[torch.Tensor] = None,
        query_scores: Optional[torch.Tensor] = None,
        graph_features: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
        return_repr: bool = False,
        anchor_state_self: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        state_idx = torch.clamp(anchor_state.long() + 1, min=0, max=2)
        state_repr = self.state_embedding(state_idx)

        if anchor_state_self is not None:
            self_idx = torch.clamp(anchor_state_self.long() + 1, min=0, max=2)
            state_repr_self = self.state_embedding(self_idx)
        else:
            state_repr_self = state_repr

        x0 = state_repr
        x0_self = state_repr_self

        if self.doc_dim > 0 and doc_embeddings is not None:
            doc_repr = self.doc_proj(doc_embeddings.float())
            if query_scores is not None:
                gate = query_scores.float().reshape(-1, 1)
                doc_repr = gate * doc_repr
            x0 = x0 + doc_repr
            x0_self = x0_self + doc_repr

        if self.graph_feat_dim > 0 and graph_features is not None:
            gf = self.graph_feat_proj(graph_features.float())
            x0 = x0 + gf
            x0_self = x0_self + gf

        intermediates = [x0_self]
        x = x0
        for conv, ln in zip(self.conv_layers, self.layer_norms):
            agg = ln(conv(x, edge_index))
            x = agg + x0
            intermediates.append(agg + x0_self)

        h_early = self.branch_early(intermediates[0])
        h_mid = self.branch_mid(intermediates[self.num_layers // 2])
        h_late = self.branch_late(intermediates[-1])

        fused = torch.cat([h_early, h_mid, h_late], dim=-1)
        logits = self.fusion(fused).squeeze(-1)

        if return_repr:
            return logits, fused
        return logits
