import numpy as np
from typing import Dict, Tuple, Any, Set, Optional, List

def _safe_normalize_hist(hist: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Normalize a histogram into a probability distribution.
    """
    hist = np.asarray(hist, dtype=np.float64)
    total = float(np.sum(hist))

    if total <= eps:
        return np.ones_like(hist, dtype=np.float64) / len(hist)

    return hist / total


def _js_similarity_from_hist(
    p_hist: np.ndarray,
    q_hist: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """
    Jensen-Shannon similarity between two histograms.

    Return value is in [0, 1].
        1.0 means identical distributions.
        0.0 means highly different distributions.

    We use log base 2, so JS divergence is in [0, 1].
    similarity = 1 - JS divergence.
    """
    p = _safe_normalize_hist(p_hist, eps)
    q = _safe_normalize_hist(q_hist, eps)

    m = 0.5 * (p + q)

    kl_pm = np.sum(p * np.log2((p + eps) / (m + eps)))
    kl_qm = np.sum(q * np.log2((q + eps) / (m + eps)))

    js_div = 0.5 * kl_pm + 0.5 * kl_qm

    js_sim = 1.0 - js_div
    js_sim = float(np.clip(js_sim, 0.0, 1.0))

    return js_sim


def _l1_similarity_from_hist(
    p_hist: np.ndarray,
    q_hist: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """
    L1 similarity between two histograms.

    For two probability distributions, L1 distance is in [0, 2].
    So we map it to similarity:
        similarity = 1 - L1 / 2

    Return value is in [0, 1].
    """
    p = _safe_normalize_hist(p_hist, eps)
    q = _safe_normalize_hist(q_hist, eps)

    l1_dist = float(np.sum(np.abs(p - q)))
    l1_sim = 1.0 - l1_dist / 2.0

    return float(np.clip(l1_sim, 0.0, 1.0))


def _evaluate_reconstructed_distribution_quality(
    propagation_scores: np.ndarray,
    groundtruth: np.ndarray,
    bins: np.ndarray,
    recon_hist_pos: np.ndarray,
    recon_hist_neg: np.ndarray,
    smooth: bool = True,
) -> Dict[str, float]:

    propagation_scores = np.asarray(propagation_scores, dtype=np.float64).reshape(-1)
    groundtruth = np.asarray(groundtruth).reshape(-1)

    valid_mask = ~np.isnan(propagation_scores)
    valid_mask &= (groundtruth == 0) | (groundtruth == 1)

    scores = propagation_scores[valid_mask]
    labels = groundtruth[valid_mask].astype(np.int64)

    full_pos_scores = scores[labels == 1]
    full_neg_scores = scores[labels == 0]

    if len(full_pos_scores) == 0 or len(full_neg_scores) == 0:
        return {
            "pos_js_similarity": np.nan,
            "neg_js_similarity": np.nan,
            "mean_js_similarity": np.nan,
            "pos_l1_similarity": np.nan,
            "neg_l1_similarity": np.nan,
            "mean_l1_similarity": np.nan,
            "overall_similarity": np.nan,
        }

    full_hist_pos, _ = np.histogram(full_pos_scores, bins=bins)
    full_hist_neg, _ = np.histogram(full_neg_scores, bins=bins)

    full_hist_pos = full_hist_pos.astype(np.float64)
    full_hist_neg = full_hist_neg.astype(np.float64)

    # 为了和重建分布的形状处理方式一致，这里也可以做同样的平滑。
    if smooth:
        full_hist_pos = _smooth_distr(full_hist_pos, window_size=5)
        full_hist_neg = _smooth_distr(full_hist_neg, window_size=5)

    pos_js_sim = _js_similarity_from_hist(recon_hist_pos, full_hist_pos)
    neg_js_sim = _js_similarity_from_hist(recon_hist_neg, full_hist_neg)
    mean_js_sim = float(np.nanmean([pos_js_sim, neg_js_sim]))

    pos_l1_sim = _l1_similarity_from_hist(recon_hist_pos, full_hist_pos)
    neg_l1_sim = _l1_similarity_from_hist(recon_hist_neg, full_hist_neg)
    mean_l1_sim = float(np.nanmean([pos_l1_sim, neg_l1_sim]))

    return {
        "pos_js_similarity": pos_js_sim,
        "neg_js_similarity": neg_js_sim,
        "mean_js_similarity": mean_js_sim,
        "pos_l1_similarity": pos_l1_sim,
        "neg_l1_similarity": neg_l1_sim,
        "mean_l1_similarity": mean_l1_sim,
        "overall_similarity": mean_js_sim,
    }


# ======================================================================
# Basic utilities
# ======================================================================

def _moving_average(signal: np.ndarray, window_size: int) -> np.ndarray:
    window = np.ones(window_size, dtype=np.float64) / window_size
    return np.convolve(signal, window, mode="same")


def _smooth_distr(hist: np.ndarray, window_size: int = 5) -> np.ndarray:
    return _moving_average(hist, window_size)


def _area_under_curve(x: np.ndarray, y: np.ndarray, l: float, r: float) -> float:
    if l == r:
        return 0.0

    mask = (x >= l) & (x <= r)
    if np.sum(mask) < 2:
        return 0.0

    return float(np.trapezoid(y[mask], x=x[mask]))


def _AT(
    x: np.ndarray,
    y_pos: np.ndarray,
    y_neg: np.ndarray,
    l: float,
    r: float,
    l_s: float,
    r_s: float,
) -> float:
    """
    Estimated accuracy after directly classifying scores outside [l, r]
    and sending scores inside [l, r] to oracle/LLM.

    Left side <= l is predicted as negative.
    Right side >= r is predicted as positive.
    Middle region [l, r] is verified by oracle/LLM.
    """
    fn = _area_under_curve(x, y_pos, l_s, l)
    fp = _area_under_curve(x, y_neg, r, r_s)
    tp = _area_under_curve(x, y_pos, l, r_s)
    tn = _area_under_curve(x, y_neg, l_s, r)

    denom = tp + fp + tn + fn
    if denom == 0:
        return 1.0

    return (tp + tn) / denom


def _unfiltered_ratio(
    x: np.ndarray,
    y_pos: np.ndarray,
    y_neg: np.ndarray,
    l: float,
    r: float,
    l_s: float,
    r_s: float,
) -> float:
    """
    Ratio of the uncertain region [l, r].
    Smaller value means fewer documents need oracle/LLM verification.
    """
    gap_area = (
        _area_under_curve(x, y_pos, l, r)
        + _area_under_curve(x, y_neg, l, r)
    )

    total_area = (
        _area_under_curve(x, y_pos, l_s, r_s)
        + _area_under_curve(x, y_neg, l_s, r_s)
    )

    if total_area == 0:
        return 1.0

    return gap_area / total_area


# ======================================================================
# AT threshold selection
# ======================================================================

def _select_sim_filter_AT(
    steps: np.ndarray,
    x: np.ndarray,
    y_pos: np.ndarray,
    y_neg: np.ndarray,
    l_s: float,
    r_s: float,
    target_acc: float,
) -> Tuple[float, float, float]:
    """
    Select an uncertain interval [l, r] such that the estimated AT accuracy
    is at least target_acc, while the uncertain ratio is minimized.
    """
    l_0 = steps[0]
    for l in steps:
        if _AT(x, y_pos, y_neg, l, steps[-1], l_s, r_s) >= target_acc:
            l_0 = l

    r_0 = steps[-1]
    for r in steps[::-1]:
        # 修正点：
        # 原代码这里把 l_s 误传成了 l_0。
        # 这里应该始终使用全局 score range 的左边界 l_s。
        if _AT(x, y_pos, y_neg, steps[0], r, l_s, r_s) >= target_acc:
            r_0 = r

    l_1 = steps[0]
    for l in steps:
        if l < r_0 and _AT(x, y_pos, y_neg, l, r_0, l_s, r_s) >= target_acc:
            l_1 = l

    r_1 = steps[-1]
    for r in steps[::-1]:
        if l_0 < r and _AT(x, y_pos, y_neg, l_0, r, l_s, r_s) >= target_acc:
            r_1 = r

    l_0_idx = np.where(steps == l_0)[0].item()
    r_0_idx = np.where(steps == r_0)[0].item()
    l_1_idx = np.where(steps == l_1)[0].item()
    r_1_idx = np.where(steps == r_1)[0].item()

    l_idx, r_idx = l_1_idx, r_0_idx
    best_l, best_r = l_0_idx, r_1_idx
    min_unf = 1.0

    while l_idx != l_0_idx or r_idx != r_1_idx:
        if l_idx + 1 == r_idx:
            r_idx += 1
        elif _AT(
            x,
            y_pos,
            y_neg,
            steps[l_idx + 1],
            steps[r_idx],
            l_s,
            r_s,
        ) >= target_acc:
            l_idx += 1
        else:
            r_idx += 1

        if r_idx >= len(steps):
            break

        unf = _unfiltered_ratio(
            x,
            y_pos,
            y_neg,
            steps[l_idx],
            steps[r_idx],
            l_s,
            r_s,
        )

        if unf < min_unf:
            min_unf = unf
            best_l, best_r = l_idx, r_idx

    return float(steps[best_l]), float(steps[best_r]), float(min_unf)


def _get_sample_label(
    idx: int,
    propagation_scores: np.ndarray,
    groundtruth: np.ndarray,
    eps: float = 1e-12,
) -> int:
    """
    For oracle/sample points, propagation_prob[idx] should be exactly 0 or 1.

    0 means negative oracle sample.
    1 means positive oracle sample.

    If numerical noise exists, fall back to groundtruth.
    """
    s = propagation_scores[idx]

    if abs(s - 0.0) <= eps:
        return 0

    if abs(s - 1.0) <= eps:
        return 1

    return int(groundtruth[idx])


def _allocate_incremental_stratified_quotas(
    pop_counts: np.ndarray,
    current_counts: np.ndarray,
    available_counts: np.ndarray,
    add_budget: int,
) -> np.ndarray:
    """
    Allocate exactly add_budget samples to bins.

    Goal:
        After adding samples, current_counts + quotas should be as close as
        possible to the population distribution pop_counts.

    This avoids repeated per-bin ceil(), which over-samples low-density bins.
    """
    pop_counts = np.asarray(pop_counts, dtype=np.int64)
    current_counts = np.asarray(current_counts, dtype=np.int64)
    available_counts = np.asarray(available_counts, dtype=np.int64)

    num_bins = len(pop_counts)
    quotas = np.zeros(num_bins, dtype=np.int64)

    add_budget = int(add_budget)
    if add_budget <= 0:
        return quotas

    add_budget = min(add_budget, int(available_counts.sum()))
    if add_budget <= 0:
        return quotas

    pop_total = int(pop_counts.sum())
    if pop_total <= 0:
        return quotas

    # Target total sample count after this round.
    target_total = int(current_counts.sum()) + add_budget

    # Ideal number of cumulative samples in each bin.
    ideal_counts = pop_counts.astype(np.float64) / float(pop_total) * float(target_total)

    # Greedily add samples to the most under-represented bin.
    # add_budget is usually small, e.g., 50, so this is cheap.
    for _ in range(add_budget):
        can_add = np.where(available_counts > quotas)[0]
        if len(can_add) == 0:
            break

        # Deficit after considering the samples already allocated in this round.
        deficits = ideal_counts[can_add] - (current_counts[can_add] + quotas[can_add])

        max_deficit = np.max(deficits)
        best_bins = can_add[np.where(deficits == max_deficit)[0]]

        # Tie break: choose the bin with larger remaining capacity.
        if len(best_bins) > 1:
            remaining_caps = available_counts[best_bins] - quotas[best_bins]
            best_bins = best_bins[np.where(remaining_caps == np.max(remaining_caps))[0]]

        bid = int(best_bins[0])
        quotas[bid] += 1

    return quotas


def _incremental_stratified_sample_fixed_budget(
    propagation_scores: np.ndarray,
    existing_sample_set: Set[int],
    last_sample_set: Set[int],
    sample_ratio: float,
    rng: np.random.RandomState,
    num_sample_bins: int = 64,
) -> Tuple[Set[int], Dict[str, Any]]:
    """
    Fixed-budget incremental stratified sampling.

    sample_ratio means the global incremental ratio in this round.
    Example:
        n = 10000, sample_ratio = 0.005 -> add about 50 samples.

    The method avoids per-bin ceil() and keeps the cumulative sample
    distribution close to the global score distribution.
    """
    n = len(propagation_scores)
    all_indices = np.arange(n, dtype=np.int64)

    target_add_size = int(round(n * sample_ratio))
    if target_add_size <= 0:
        return set(), {
            "target_add_size": 0,
            "actual_add_size": 0,
            "reason": "zero_budget",
        }

    existing_mask = np.zeros(n, dtype=bool)
    if len(existing_sample_set) > 0:
        existing_arr = np.asarray(sorted(existing_sample_set), dtype=np.int64)
        existing_arr = existing_arr[(existing_arr >= 0) & (existing_arr < n)]
        existing_mask[existing_arr] = True

    last_mask = np.zeros(n, dtype=bool)
    if len(last_sample_set) > 0:
        last_arr = np.asarray(sorted(last_sample_set), dtype=np.int64)
        last_arr = last_arr[(last_arr >= 0) & (last_arr < n)]
        last_mask[last_arr] = True

    # Eligible population excludes existing oracle/sample ids.
    # Current reconstruction samples are part of the eligible population.
    eligible_mask = ~existing_mask

    # Available candidates exclude both existing oracle/sample ids and last samples.
    available_mask = eligible_mask & (~last_mask)

    available_indices = all_indices[available_mask]
    if len(available_indices) == 0:
        return set(), {
            "target_add_size": target_add_size,
            "actual_add_size": 0,
            "reason": "no_available_candidates",
        }

    target_add_size = min(target_add_size, len(available_indices))

    score_min = float(np.min(propagation_scores))
    score_max = float(np.max(propagation_scores))

    # Degenerate case: all scores are identical. Fall back to uniform sampling.
    if score_min >= score_max:
        chosen = rng.choice(
            available_indices,
            size=target_add_size,
            replace=False,
        )
        return set(int(x) for x in chosen), {
            "target_add_size": target_add_size,
            "actual_add_size": len(chosen),
            "reason": "uniform_fallback",
        }

    bucket_edges = np.linspace(score_min, score_max, num_sample_bins + 1)
    bucket_ids = np.digitize(propagation_scores, bucket_edges, right=False) - 1
    bucket_ids = np.clip(bucket_ids, 0, num_sample_bins - 1)

    pop_counts = np.bincount(
        bucket_ids[eligible_mask],
        minlength=num_sample_bins,
    )

    current_counts = np.bincount(
        bucket_ids[last_mask],
        minlength=num_sample_bins,
    )

    available_counts = np.bincount(
        bucket_ids[available_mask],
        minlength=num_sample_bins,
    )

    quotas = _allocate_incremental_stratified_quotas(
        pop_counts=pop_counts,
        current_counts=current_counts,
        available_counts=available_counts,
        add_budget=target_add_size,
    )

    new_sample_id_set: Set[int] = set()

    for bid in range(num_sample_bins):
        q = int(quotas[bid])
        if q <= 0:
            continue

        bucket_candidates = all_indices[
            (bucket_ids == bid) & available_mask
        ]

        if len(bucket_candidates) == 0:
            continue

        q = min(q, len(bucket_candidates))

        chosen = rng.choice(
            bucket_candidates,
            size=q,
            replace=False,
        )

        for idx in chosen:
            new_sample_id_set.add(int(idx))

    debug_info = {
        "target_add_size": target_add_size,
        "actual_add_size": len(new_sample_id_set),
        "bucket_edges": bucket_edges,
        "bucket_ids": bucket_ids,
        "pop_counts": pop_counts,
        "current_counts": current_counts,
        "available_counts": available_counts,
        "quotas": quotas,
        "reason": "fixed_budget_stratified",
    }

    return new_sample_id_set, debug_info


def _choose_threshold_from_sample_score(
    propagation_scores: np.ndarray,
    last_sample_idx: Dict[int, bool],
    sample_ratio: float,
    oracle_nodes: Set[int],
    oracle,
    document_contents: List[str],
    query_content: str,
    num_bins: int,
    target_acc: float,
    seed: int,
) -> Tuple[float, float, Dict[str, Any], Dict[int, bool]] | None:

    if num_bins < 4:
        raise ValueError(f"num_bins should be at least 4, got {num_bins}")
    n = len(propagation_scores)
    if n == 0:
        return None
    rng = np.random.RandomState(seed)

    if last_sample_idx is None:
        last_sample_idx = {}

    last_sample_idx = {
        int(idx): bool(label)
        for idx, label in last_sample_idx.items()
        if 0 <= int(idx) < n
    }

    last_sample_set: Set[int] = set(last_sample_idx.keys())

    new_sample_id_set, sampling_debug = _incremental_stratified_sample_fixed_budget(
        propagation_scores=propagation_scores,
        existing_sample_set=oracle_nodes,
        last_sample_set=last_sample_set,
        sample_ratio=sample_ratio,
        rng=rng,
        num_sample_bins=64,
    )

    reconstruct_sample_id_set: Set[int] = set(last_sample_set)
    reconstruct_sample_id_set.update(new_sample_id_set)

    if len(reconstruct_sample_id_set) == 0:
        return None

    sample_indices = np.asarray(sorted(reconstruct_sample_id_set), dtype=np.int64)
    sample_values = propagation_scores[sample_indices].astype(np.float64)
    new_oracle_labels = {}
    if len(new_sample_id_set) > 0:
        new_oracle_labels = llm_oracle_labeling(oracle, new_sample_id_set, document_contents, query_content)

    all_sample_labels = {}
    all_sample_labels.update(last_sample_idx)
    all_sample_labels.update(new_oracle_labels)
    sample_labels = np.asarray([all_sample_labels[idx] for idx in sample_indices], dtype=np.int64)

    pos_scores = sample_values[sample_labels == 1]
    neg_scores = sample_values[sample_labels == 0]

    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return None

    domain_min = min(
        float(np.min(propagation_scores)),
        float(np.min(sample_values)),
        0.0,
    )

    domain_max = max(
        float(np.max(propagation_scores)),
        float(np.max(sample_values)),
        1.0,
    )


    if domain_min == domain_max:
        return None

    bins = np.linspace(
        domain_min,
        domain_max,
        num_bins + 1
    )

    hist_pos, _ = np.histogram(
        pos_scores,
        bins=bins
    )

    hist_neg, _ = np.histogram(
        neg_scores,
        bins=bins
    )


    rand_pos = rng.choice(
        [0.0, 0.1, 0.2],
        size=hist_pos.shape,
        p=[0.6, 0.3, 0.1],
    )

    rand_neg = rng.choice(
        [0.0, 0.1, 0.2],
        size=hist_neg.shape,
        p=[0.6, 0.3, 0.1],
    )


    hist_pos = (
        hist_pos.astype(np.float64)
        + rand_pos
    )

    hist_neg = (
        hist_neg.astype(np.float64)
        + rand_neg
    )


    hist_pos = _smooth_distr(
        hist_pos,
        window_size=5
    )

    hist_neg = _smooth_distr(
        hist_neg,
        window_size=5
    )
    # fit_quality = _evaluate_reconstructed_distribution_quality(
    #     propagation_scores=propagation_scores,
    #     groundtruth=groundtruth,
    #     bins=bins,
    #     recon_hist_pos=hist_pos,
    #     recon_hist_neg=hist_neg,
    #     smooth=True,
    # )
    bins_center = (bins[:-1] + bins[1:]) / 2.0
    steps = bins[::2]

    l, r, _ = _select_sim_filter_AT(
        steps=steps,
        x=bins_center,
        y_pos=hist_pos,
        y_neg=hist_neg,
        l_s=bins[0],
        r_s=bins[-1],
        target_acc=target_acc,
    )

    fig_var = {}

    fig_var["bin_center"] = bins_center
    fig_var["hist_pos"] = hist_pos
    fig_var["hist_neg"] = hist_neg
    fig_var["bins"] = bins

    # ============================================================
    # 保存当前累计所有sample的id和LLM label
    # key: document id
    # value: oracle label
    # ============================================================

    fig_var["sample_labels"] = all_sample_labels

    # fig_var["fit_quality"] = fit_quality

    l_idx = np.where(steps == l)[0][0]
    r_idx = np.where(steps == r)[0][0]


    le_idx = max(
        0,
        l_idx * 2 - 1
    )

    re_idx = min(
        len(bins)-1,
        r_idx * 2 + 1
    )

    tau_neg = float(bins[le_idx])
    tau_pos = float(bins[re_idx])

    if tau_neg > tau_pos:
        tau_neg, tau_pos = tau_pos, tau_neg

    return (
        tau_neg,
        tau_pos,
        fig_var,
        all_sample_labels,
    )


# ======================================================================
# Main function
# ======================================================================

def progressive_calibration(
    propagation_prob,
    oracle,
    last_sample_idx: Optional[Dict[int, bool]],
    oracle_nodes: Set[int],
    document_contents: List[str],
    query_content: str,
    alpha: float = 0.8,
    last_sample_number: float = 0,
    sample_number: float = 0.02,
    num_bins: int = 64,
    seed: int = 42,
):

    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha should be in (0,1], got {alpha}")

    scores = np.asarray(
        propagation_prob,
        dtype=np.float64
    ).reshape(-1)

    n = len(scores)

    threshold = _choose_threshold_from_sample_score(
        propagation_scores=scores,
        last_sample_idx=last_sample_idx,
        sample_ratio=sample_number-last_sample_number,
        oracle_nodes=oracle_nodes,
        oracle=oracle,
        document_contents=document_contents,
        query_content=query_content,
        num_bins=num_bins,
        target_acc=alpha,
        seed=seed,
    )

    if threshold is None:
        return (0.0, 1.0, {}, last_sample_idx)

    l, r, fig_var, all_sample_labels = threshold
    return (l, r, fig_var, all_sample_labels)


def LLM_final_judge(l, r, probs, oracle_nodes, last_sample_idx, oracle, document_contents, query_content) -> Tuple[Dict[int, bool], int]:
    final_result = {}
    llm_docs = set(oracle_nodes.keys())
    llm_docs.update(last_sample_idx.keys())
    ambiguous_doc = set()

    for i in range(len(document_contents)):

        if i in oracle_nodes.keys():
            final_result[i] = oracle_nodes[i]
            continue

        if i in last_sample_idx.keys():
            final_result[i] = last_sample_idx[i]
            continue

        if probs[i] <= l:
            final_result[i] = False

        elif probs[i] >= r:
            final_result[i] = True

        else:
            ambiguous_doc.add(i)

    if len(ambiguous_doc) > 0:

        ambiguous_labels = llm_oracle_labeling(oracle, ambiguous_doc, document_contents, query_content)
        final_result.update(ambiguous_labels)
        llm_docs.update(ambiguous_doc)

    llm_invocation_times = len(llm_docs)

    return final_result, llm_invocation_times
