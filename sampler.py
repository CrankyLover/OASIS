from collections import deque
from typing import Dict, Iterable, List, Sequence
from oracle import LLM_Oracle
from utils import llm_oracle_labeling

import numpy as np


def _normalize_neighbors(graph: Dict[int, List[int]], num_nodes: int) -> List[List[int]]:
    neighbor_sets = [set() for _ in range(num_nodes)]
    for src in range(num_nodes):
        for dst in graph.get(src, []):
            dst_i = int(dst)
            if dst_i == src or dst_i < 0 or dst_i >= num_nodes:
                continue
            neighbor_sets[src].add(dst_i)
            neighbor_sets[dst_i].add(src)
    return [sorted(list(items)) for items in neighbor_sets]


def _multi_source_assign(
    neighbors: Sequence[Sequence[int]],
    centers: Sequence[int],
) -> np.ndarray:
    owner = np.full(len(neighbors), -1, dtype=np.int64)
    queue: deque[int] = deque()

    for owner_id, center in enumerate(centers):
        center_i = int(center)
        if owner[center_i] != -1:
            continue
        owner[center_i] = owner_id
        queue.append(center_i)

    while queue:
        node = queue.popleft()
        for nbr in neighbors[node]:
            if owner[nbr] != -1:
                continue
            owner[nbr] = owner[node]
            queue.append(nbr)

    if np.any(owner < 0):
        fallback = 0 if len(centers) > 0 else -1
        owner[owner < 0] = fallback

    return owner


def _dist_to_sources(neighbors: Sequence[Sequence[int]], sources: Iterable[int]) -> np.ndarray:
    dist = np.full(len(neighbors), np.inf, dtype=np.float64)
    queue: deque[int] = deque()
    for source in sources:
        source_i = int(source)
        dist[source_i] = 0.0
        queue.append(source_i)

    while queue:
        node = queue.popleft()
        next_dist = dist[node] + 1.0
        for nbr in neighbors[node]:
            if dist[nbr] <= next_dist:
                continue
            dist[nbr] = next_dist
            queue.append(nbr)
    return dist


def _choose_farthest_centers(
    neighbors: Sequence[Sequence[int]],
    num_centers: int,
    rng: np.random.Generator,
) -> List[int]:
    num_nodes = len(neighbors)
    num_centers = min(max(1, int(num_centers)), num_nodes)

    first = int(rng.integers(0, num_nodes))
    centers = [first]

    while len(centers) < num_centers:
        dist = _dist_to_sources(neighbors, centers)
        finite = np.isfinite(dist)
        if not np.any(finite):
            remaining = [i for i in range(num_nodes) if i not in centers]
            centers.append(int(rng.choice(remaining)))
            continue

        max_dist = np.max(dist[finite])
        candidates = np.where(dist == max_dist)[0]
        candidates = [int(x) for x in candidates if int(x) not in centers]
        if not candidates:
            break
        centers.append(int(rng.choice(candidates)))

    return centers


def _batch_sizes(total_budget: int, num_batches: int) -> List[int]:
    base = total_budget // num_batches
    rem = total_budget % num_batches
    return [base + (1 if i < rem else 0) for i in range(num_batches)]


def _sample_one_batch(
    regions: List[List[int]],
    batch_size: int,
    rng: np.random.Generator,
) -> List[int]:
    nonempty = [idx for idx, region in enumerate(regions) if region]
    if batch_size <= 0 or not nonempty:
        return []

    selected: List[int] = []

    shuffled_regions = list(nonempty)
    rng.shuffle(shuffled_regions)
    for region_id in shuffled_regions:
        if len(selected) >= batch_size:
            break
        node_pos = int(rng.integers(0, len(regions[region_id])))
        selected.append(regions[region_id].pop(node_pos))

    while len(selected) < batch_size:
        nonempty = [idx for idx, region in enumerate(regions) if region]
        if not nonempty:
            break
        sizes = np.asarray([len(regions[idx]) for idx in nonempty], dtype=np.float64)
        probs = sizes / sizes.sum()
        region_id = int(rng.choice(nonempty, p=probs))
        node_pos = int(rng.integers(0, len(regions[region_id])))
        selected.append(regions[region_id].pop(node_pos))

    return selected


def graph_uniform_minibatches(
    graph: Dict[int, List[int]],
    num_nodes: int,
    total_budget: int,
    num_batches: int,
    seed: int = 42,
    num_regions: int | None = None,
) -> List[List[int]]:
    """
    Sample oracle mini-batches by graph coverage, without propagation scores.

    The graph is partitioned around farthest-point centers, then every mini-batch
    draws from as many regions as possible. Nodes are not repeated.
    """
    if total_budget <= 0:
        raise ValueError("total_budget should be positive")
    if num_batches <= 1:
        raise ValueError("num_batches should be greater than 1 for leave-one-out")
    if num_nodes <= 0:
        raise ValueError("num_nodes should be positive")

    rng = np.random.default_rng(seed)
    budget = min(int(total_budget), int(num_nodes))
    neighbors = _normalize_neighbors(graph, num_nodes)

    if num_regions is None:
        num_regions = min(max(num_batches * 2, 8), budget, num_nodes)

    centers = _choose_farthest_centers(neighbors, num_regions, rng)
    owner = _multi_source_assign(neighbors, centers)

    regions = [[] for _ in centers]
    for node_id, region_id in enumerate(owner):
        regions[int(region_id)].append(int(node_id))
    for region in regions:
        rng.shuffle(region)

    batches = []
    for size in _batch_sizes(budget, num_batches):
        batches.append(_sample_one_batch(regions, size, rng))

    return [batch for batch in batches if batch]


def merge_batch_labels(minibatches, document_contents, query_content, oracle, groundtruth):

    sample_ids = set()
    for batch in minibatches:
        for i in batch:
            sampled_ids.add(i)

    labels = llm_oracle_labeling(oracle, sampled_ids, document_contents, query_content)

    return labels


def chunk_oracle_batches(node_ids: Sequence[int], num_batches: int, seed: int = 42) -> List[List[int]]:
    """Split an oracle set into training mini-batches without changing membership."""
    ids = [int(x) for x in node_ids]
    if len(ids) < 2:
        raise ValueError("at least two oracle nodes are required")
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    num_batches = max(2, min(int(num_batches), len(ids)))
    return [part.astype(int).tolist() for part in np.array_split(np.asarray(ids), num_batches) if len(part) > 0]


def score_stratified_ranking(
    scores: np.ndarray,
    num_bins: int = 10,
    seed: int = 42,
) -> np.ndarray:
    """Create a score-stratified ranking by interleaving score bins.

    Nodes are sorted by score descending, split into equal-sized bins,
    then interleaved (1 from each bin, round-robin). This produces a
    ranking where every prefix covers all score ranges.

    Returns:
        ranking: node indices in stratified order, shape [num_nodes].
    """
    rng = np.random.default_rng(seed)
    order = np.argsort(-scores)

    n = len(scores)
    bins = []
    for k in range(num_bins):
        start = int(round(k * n / num_bins))
        end = int(round((k + 1) * n / num_bins))
        bin_nodes = order[start:end].copy()
        rng.shuffle(bin_nodes)
        bins.append(bin_nodes)

    ranking = []
    max_len = max(len(b) for b in bins)
    for i in range(max_len):
        for b in bins:
            if i < len(b):
                ranking.append(int(b[i]))

    return np.array(ranking, dtype=np.int64)


def score_stratified_minibatches(
    scores: np.ndarray,
    total_budget: int,
    num_batches: int,
    num_bins: int = 10,
    seed: int = 42,
) -> List[List[int]]:
    """Score-stratified bucket sampling: each batch covers all score ranges."""
    ranking = score_stratified_ranking(scores, num_bins=num_bins, seed=seed)
    budget = min(total_budget, len(ranking))
    selected = ranking[:budget]
    return chunk_oracle_batches(selected, num_batches, seed=seed)
