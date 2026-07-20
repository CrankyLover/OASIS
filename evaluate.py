from typing import Dict, Optional

import numpy as np


def auc_like_score(probabilities, groundtruth, oracle: Optional[Dict[int, bool]] = None) -> float:
    """Project-local distribution separation score, equivalent to AUC.

    If oracle is provided, nodes whose ids appear in oracle.keys()
    are excluded from AUC computation.
    """
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    gt = np.asarray(groundtruth, dtype=np.int64).reshape(-1)

    if p.shape[0] != gt.shape[0]:
        raise ValueError(
            f"probabilities and groundtruth should have the same length, "
            f"got {p.shape[0]} and {gt.shape[0]}"
        )

    valid = ~np.isnan(p)

    if oracle is not None:
        oracle_mask = np.zeros_like(valid, dtype=bool)
        for node_id in oracle.keys():
            node_i = int(node_id)
            if 0 <= node_i < len(oracle_mask):
                oracle_mask[node_i] = True
        valid = valid & (~oracle_mask)

    p = p[valid]
    gt = gt[valid]

    pos = p[gt == 1]
    neg = p[gt == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")

    pos_sorted = np.sort(pos)
    neg_sorted = np.sort(neg)
    pos_ratio = np.searchsorted(neg_sorted, pos, side="left") / len(neg_sorted)
    neg_ratio = (len(pos_sorted) - np.searchsorted(pos_sorted, neg, side="right")) / len(pos_sorted)
    return float(np.mean(np.concatenate([pos_ratio, neg_ratio])))


def binary_error_summary(probabilities, groundtruth, threshold: float = 0.5) -> Dict[str, float]:
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    gt = np.asarray(groundtruth, dtype=np.int64).reshape(-1)
    pred = p >= threshold

    fp = int(np.sum((pred == 1) & (gt == 0)))
    fn = int(np.sum((pred == 0) & (gt == 1)))
    tp = int(np.sum((pred == 1) & (gt == 1)))
    tn = int(np.sum((pred == 0) & (gt == 0)))

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "high_conf_fp": int(np.sum((p >= 0.8) & (gt == 0))),
        "low_conf_fn": int(np.sum((p <= 0.2) & (gt == 1))),
    }


def hard_case_summary(
    probabilities,
    groundtruth,
    raw_scores: Optional[np.ndarray] = None,
    low_quantile: float = 0.2,
    high_quantile: float = 0.8,
) -> Dict[str, float]:
    """Measure hard positive and hard negative behavior against raw similarity."""
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    gt = np.asarray(groundtruth, dtype=np.int64).reshape(-1)
    summary: Dict[str, float] = {}

    if raw_scores is None:
        return summary

    raw = np.asarray(raw_scores, dtype=np.float64).reshape(-1)
    low_cut = float(np.quantile(raw, low_quantile))
    high_cut = float(np.quantile(raw, high_quantile))

    hard_pos = (gt == 1) & (raw <= low_cut)
    hard_neg = (gt == 0) & (raw >= high_cut)

    summary["hard_pos_count"] = int(np.sum(hard_pos))
    summary["hard_neg_count"] = int(np.sum(hard_neg))
    summary["hard_pos_mean_prob"] = float(np.mean(p[hard_pos])) if np.any(hard_pos) else float("nan")
    summary["hard_neg_mean_prob"] = float(np.mean(p[hard_neg])) if np.any(hard_neg) else float("nan")
    summary["hard_pos_recall_at_05"] = float(np.mean(p[hard_pos] >= 0.5)) if np.any(hard_pos) else float("nan")
    summary["hard_neg_false_positive_at_05"] = float(np.mean(p[hard_neg] >= 0.5)) if np.any(hard_neg) else float("nan")
    return summary


def evaluate_probabilities(probabilities, groundtruth, raw_scores=None) -> Dict[str, float]:
    summary = {
        "auc": auc_like_score(probabilities, groundtruth),
        "pos_mean": float(np.mean(np.asarray(probabilities)[np.asarray(groundtruth) == 1])),
        "neg_mean": float(np.mean(np.asarray(probabilities)[np.asarray(groundtruth) == 0])),
    }
    summary.update(binary_error_summary(probabilities, groundtruth))
    summary.update(hard_case_summary(probabilities, groundtruth, raw_scores))
    return summary
