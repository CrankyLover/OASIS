import numpy as np

from typing import Dict, List, Set

# ------------------------------------------------------------------
# Uncertainty sampling helpers
# ------------------------------------------------------------------

def _gaussian_smooth_1d(values: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    if len(values) <= 2:
        return values.astype(np.float64)
    radius = max(1, int(3 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel = kernel / (kernel.sum() + 1e-12)
    return np.convolve(values, kernel, mode="same")


def _find_local_peaks(y: np.ndarray) -> list[int]:
    if len(y) == 0:
        return []
    if len(y) == 1:
        return [0]
    peaks = []
    if y[0] >= y[1]:
        peaks.append(0)
    for i in range(1, len(y) - 1):
        if y[i] >= y[i - 1] and y[i] >= y[i + 1]:
            peaks.append(i)
    if y[-1] >= y[-2]:
        peaks.append(len(y) - 1)
    return peaks


def _find_uncertain_interval_from_shape(
    smooth_hist: np.ndarray,
    bin_edges: np.ndarray,
    peak_height_ratio: float = 0.05,
    valley_depth_threshold: float = 0.80,
    expand_alpha: float = 0.35,
) -> tuple[float, float] | None:
    y = np.asarray(smooth_hist, dtype=np.float64)
    if len(y) < 5:
        return None

    max_height = float(y.max())
    if max_height <= 1e-12:
        return None

    raw_peaks = _find_local_peaks(y)
    peaks = [idx for idx in raw_peaks if y[idx] >= peak_height_ratio * max_height]
    peaks = sorted(set(peaks))

    if len(peaks) < 2:
        return None

    min_peak_gap = max(3, int(0.05 * len(y)))
    best = None

    for left_peak, right_peak in zip(peaks[:-1], peaks[1:]):
        if right_peak - left_peak < min_peak_gap:
            continue
        if right_peak - left_peak <= 2:
            continue

        inner = y[left_peak + 1:right_peak]
        if len(inner) == 0:
            continue

        valley = left_peak + 1 + int(np.argmin(inner))
        left_height = y[left_peak]
        right_height = y[right_peak]
        valley_height = y[valley]
        min_peak_height = min(left_height, right_height)

        if min_peak_height <= 1e-12:
            continue

        valley_depth = valley_height / (min_peak_height + 1e-12)
        if valley_depth > valley_depth_threshold:
            continue

        score = (1.0 - valley_depth) * min_peak_height
        if best is None or score > best["score"]:
            best = {
                "left_peak": left_peak,
                "right_peak": right_peak,
                "valley": valley,
                "valley_height": valley_height,
                "min_peak_height": min_peak_height,
                "score": score,
            }

    if best is None:
        return None

    valley = best["valley"]
    valley_height = best["valley_height"]
    min_peak_height = best["min_peak_height"]

    threshold = valley_height + expand_alpha * (min_peak_height - valley_height)

    left = valley
    while left > best["left_peak"] and y[left] <= threshold:
        left -= 1
    if y[left] > threshold and left < valley:
        left += 1

    right = valley
    while right < best["right_peak"] and y[right] <= threshold:
        right += 1
    if y[right] > threshold and right > valley:
        right -= 1

    low = float(bin_edges[left])
    high = float(bin_edges[right + 1])
    low = max(0.0, min(1.0, low))
    high = max(0.0, min(1.0, high))

    if high <= low:
        return None

    return low, high


def _score_stratified_sample(
    candidate_ids: np.ndarray,
    candidate_scores: np.ndarray,
    budget: int,
    low: float,
    high: float,
    num_bins: int = 10,
) -> Set[int]:
    candidate_ids = np.asarray(candidate_ids, dtype=np.int64)
    candidate_scores = np.asarray(candidate_scores, dtype=np.float64)
    n = len(candidate_ids)

    if budget <= 0 or n == 0:
        return set()
    if n <= budget:
        return set(int(i) for i in candidate_ids)

    if high <= low:
        low, high = 0.0, 1.0

    num_bins = int(min(max(1, num_bins), budget, n))
    bin_edges = np.linspace(low, high, num_bins + 1)
    scores = np.clip(candidate_scores, low, high)
    bin_ids = np.digitize(scores, bin_edges[1:-1], right=False)

    groups = []
    counts = []
    for b in range(num_bins):
        idx = np.where(bin_ids == b)[0]
        groups.append(idx)
        counts.append(len(idx))

    counts = np.asarray(counts, dtype=np.int64)
    if counts.sum() == 0:
        chosen = np.random.choice(n, size=budget, replace=False)
        return set(int(candidate_ids[i]) for i in chosen)

    raw_quota = counts / counts.sum() * budget
    quota = np.floor(raw_quota).astype(np.int64)
    quota = np.minimum(quota, counts)
    remaining = budget - int(quota.sum())

    fractional = raw_quota - np.floor(raw_quota)
    order = np.argsort(-fractional)

    while remaining > 0:
        updated = False
        for b in order:
            if remaining <= 0:
                break
            if quota[b] < counts[b]:
                quota[b] += 1
                remaining -= 1
                updated = True
        if not updated:
            break

    selected = []
    for b, idx in enumerate(groups):
        k = int(quota[b])
        if k <= 0 or len(idx) == 0:
            continue
        chosen = np.random.choice(idx, size=k, replace=False)
        selected.extend(chosen.tolist())

    if len(selected) < budget:
        selected_set = set(selected)
        rest = [i for i in range(n) if i not in selected_set]
        need = budget - len(selected)
        if len(rest) > 0:
            fill = np.random.choice(rest, size=min(need, len(rest)), replace=False)
            selected.extend(fill.tolist())

    return set(int(candidate_ids[i]) for i in selected[:budget])


def uncertain_sample(
    probs: np.ndarray,
    oracle: Dict[int, bool],
    num_nodes: int,
    budget: int,
    eta1: float
) -> Set[int]:
    """Distribution-shape-guided uncertainty sampling.

    1. Estimate score distribution over non-oracle nodes.
    2. If bimodal structure exists, find valley interval.
    3. If valley candidates < 8% of all candidates, stop (return empty).
    4. Otherwise sample from valley. If no bimodal structure, fallback to stratified.
    """
    # early_stop_threshold = 0.08

    if budget <= 0:
        return set()

    p = np.asarray(probs, dtype=np.float64).reshape(-1)
    if len(p) < num_nodes:
        raise ValueError(f"len(probs)={len(p)} < num_nodes={num_nodes}")

    p = p[:num_nodes]
    p = np.clip(p, 0.0, 1.0)

    oracle_set = {int(k) for k in oracle.keys()}
    valid_mask = np.isfinite(p)
    candidate_mask = np.zeros(num_nodes, dtype=bool)
    for i in range(num_nodes):
        if i not in oracle_set and valid_mask[i]:
            candidate_mask[i] = True

    candidate_ids = np.where(candidate_mask)[0]
    if len(candidate_ids) == 0:
        return set()

    budget = min(int(budget), len(candidate_ids))
    candidate_scores = p[candidate_ids]

    score_min = float(np.min(candidate_scores))
    score_max = float(np.max(candidate_scores))
    if abs(score_max - score_min) < 1e-12:
        return set()

    pad = 0.02 * (score_max - score_min)
    hist_low = max(0.0, score_min - pad)
    hist_high = min(1.0, score_max + pad)
    if hist_high <= hist_low:
        return set()

    hist_bins = min(100, max(30, int(np.sqrt(len(candidate_ids)))))
    hist, bin_edges = np.histogram(candidate_scores, bins=hist_bins, range=(hist_low, hist_high))
    smooth_hist = _gaussian_smooth_1d(hist.astype(np.float64), sigma=2.0)

    interval = _find_uncertain_interval_from_shape(
        smooth_hist=smooth_hist,
        bin_edges=bin_edges,
        peak_height_ratio=0.05,
        valley_depth_threshold=0.80,
        expand_alpha=0.35,
    )

    if interval is None:
        return _score_stratified_sample(
            candidate_ids=candidate_ids,
            candidate_scores=candidate_scores,
            budget=budget,
            low=0.0,
            high=1.0,
            num_bins=10,
        )

    low, high = interval
    uncertain_mask = (candidate_scores >= low) & (candidate_scores <= high)
    uncertain_ids = candidate_ids[uncertain_mask]
    uncertain_scores = candidate_scores[uncertain_mask]

    valley_ratio = len(uncertain_ids) / (len(candidate_ids) + 1e-12)
    print(f"    valley: {len(uncertain_ids)} uncertain / {len(candidate_ids)} candidates = {valley_ratio:.4f}")

    if valley_ratio < eta1:
        return set()

    if len(uncertain_ids) == 0:
        return set()
    if len(uncertain_ids) <= budget:
        return set(int(i) for i in uncertain_ids)

    return _score_stratified_sample(
        candidate_ids=uncertain_ids,
        candidate_scores=uncertain_scores,
        budget=budget,
        low=low,
        high=high,
        num_bins=8,
    )


def merge_oracle_newnodes_to_batches(
    oracle_annotated_labels: Dict[int, bool],
    new_nodes: Set[int],
    num_nodes: int,
    num_batches: int,
    seed: int = 42,
) -> List[List[int]]:
    rng = np.random.default_rng(seed)

    old_nodes = {int(i) for i in oracle_annotated_labels.keys() if 0 <= int(i) < num_nodes}
    valid_new_nodes = {int(i) for i in new_nodes if 0 <= int(i) < num_nodes}
    all_oracle_nodes = list(old_nodes | valid_new_nodes)

    if not all_oracle_nodes:
        return []

    rng.shuffle(all_oracle_nodes)
    num_batches = min(num_batches, len(all_oracle_nodes))
    batches: List[List[int]] = [[] for _ in range(num_batches)]

    for idx, node_id in enumerate(all_oracle_nodes):
        batches[idx % num_batches].append(node_id)

    return [batch for batch in batches if batch]


def compute_ranking_stability_metrics(
    probs: np.ndarray,
    prev_probs: np.ndarray,
    num_pairs: int = 200_000,
    seed: int = 42,
    eps: float = 1e-12,
) -> Dict[str, float]:

    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    prev_probs = np.asarray(prev_probs, dtype=np.float64).reshape(-1)

    if probs.shape != prev_probs.shape:
        raise ValueError(
            f"probs and prev_probs must have the same shape, "
            f"got {probs.shape} and {prev_probs.shape}."
        )

    n = len(probs)
    if n < 2:
        raise ValueError("At least two documents are required.")

    # Remove NaN / inf values.
    valid_mask = np.isfinite(probs) & np.isfinite(prev_probs)
    probs = probs[valid_mask]
    prev_probs = prev_probs[valid_mask]

    n = len(probs)
    if n < 2:
        raise ValueError("Fewer than two valid documents after filtering NaN/inf.")

    # ------------------------------------------------------------
    # 1. Score stability
    # ------------------------------------------------------------
    abs_diff = np.abs(probs - prev_probs)

    # Global mean absolute score change.
    score_delta = float(np.mean(abs_diff))

    # Uncertainty weight: max at p=0.5, close to 0 near p=0 or p=1.
    uncertain_weight = 4.0 * probs * (1.0 - probs)
    uncertain_weight = np.clip(uncertain_weight, 0.0, None)

    if uncertain_weight.sum() <= eps:
        uncertain_score_delta = 0.0
    else:
        uncertain_score_delta = float(
            np.sum(uncertain_weight * abs_diff) / np.sum(uncertain_weight)
        )

    # ------------------------------------------------------------
    # 2. Ranking stability: random pairwise rank flip rate
    # ------------------------------------------------------------
    rng = np.random.default_rng(seed)

    pair_i = rng.integers(0, n, size=num_pairs)
    pair_j = rng.integers(0, n, size=num_pairs)

    # Avoid self-pairs.
    non_self_mask = pair_i != pair_j
    pair_i = pair_i[non_self_mask]
    pair_j = pair_j[non_self_mask]

    curr_diff = probs[pair_i] - probs[pair_j]
    prev_diff = prev_probs[pair_i] - prev_probs[pair_j]

    # Ignore pairs that are nearly tied in either round.
    valid_pair_mask = (np.abs(curr_diff) > eps) & (np.abs(prev_diff) > eps)

    curr_diff = curr_diff[valid_pair_mask]
    prev_diff = prev_diff[valid_pair_mask]

    valid_pair_count = len(curr_diff)

    if valid_pair_count == 0:
        rank_flip_rate = 0.0
    else:
        rank_flip = curr_diff * prev_diff < 0
        rank_flip_rate = float(np.mean(rank_flip))

    metrics = {
        "score_delta": score_delta,
        "uncertain_score_delta": uncertain_score_delta,
        "rank_flip_rate": rank_flip_rate,
        "valid_pair_count": int(valid_pair_count),
    }

    print("Ranking stability metrics:")
    print(f"  score_delta             = {score_delta:.6f}")
    print(f"  uncertain_score_delta   = {uncertain_score_delta:.6f}")
    print(f"  rank_flip_rate          = {rank_flip_rate:.6f}")
    print(f"  valid_pair_count        = {valid_pair_count}")

    return metrics
