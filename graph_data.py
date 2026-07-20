from typing import Dict, List, Optional

import torch
from torch_geometric.data import Data
from torch_geometric.utils import coalesce, remove_self_loops, to_undirected


def graph_to_edge_index(
    graph: Dict[int, List[int]],
    num_nodes: Optional[int] = None,
    make_undirected: bool = True,
) -> torch.Tensor:
    """Convert an adjacency-list graph into a PyG edge_index tensor."""
    if num_nodes is None:
        max_id = -1
        for src, neighbors in graph.items():
            max_id = max(max_id, int(src))
            for dst in neighbors:
                max_id = max(max_id, int(dst))
        num_nodes = max_id + 1

    rows = []
    cols = []
    for src in range(num_nodes):
        for dst in graph.get(src, []):
            src_i = int(src)
            dst_i = int(dst)
            if src_i == dst_i:
                continue
            rows.append(src_i)
            cols.append(dst_i)

    if not rows:
        raise ValueError("graph has no edges")

    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    edge_index, _ = remove_self_loops(edge_index)

    if make_undirected:
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)

    edge_index = coalesce(edge_index, num_nodes=num_nodes)
    return edge_index


def build_pyg_data(
    graph: Dict[int, List[int]],
    num_nodes: Optional[int] = None,
    make_undirected: bool = True,
) -> Data:
    """Build the minimal PyG Data object used by the anchor-only GNN."""
    edge_index = graph_to_edge_index(
        graph=graph,
        num_nodes=num_nodes,
        make_undirected=make_undirected,
    )
    if num_nodes is None:
        num_nodes = int(edge_index.max().item()) + 1
    return Data(edge_index=edge_index, num_nodes=int(num_nodes))

