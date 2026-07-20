"""Query-time GNN evaluation with distribution-shape-guided active learning.

Round-based labeling:
  Round 0: graph-coverage sample (400 nodes) → train → predict
  Round 1..N: uncertain_sample from prediction valley (100/round) → retrain → predict
  Early stop: when valley contains < 8% of unlabeled candidates, or budget exhausted.

After all queries: threshold selection via progressive sampling at user-defined accuracy targets.
"""

import argparse
import os
import pickle
import sys
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from scipy.stats import wasserstein_distance

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import load_pickle, cosine_scores, normalized_docs, llm_oracle_labeling
from oracle import LLM_Oracle
from nsg_index import GraphIndex
from evaluate import auc_like_score, evaluate_probabilities
from graph_data import build_pyg_data
from inference import predict_all
from sampler import graph_uniform_minibatches, merge_batch_labels
from train import QueryTimeGNNConfig, train_query_time_gnn
from active_learning import merge_oracle_newnodes_to_batches, compute_ranking_stability_metrics, uncertain_sample
from threshold_calibration import progressive_calibration, LLM_final_judge


def main():
    parser = argparse.ArgumentParser(description="Query-Conditioned GNN evaluation with active learning")

    parser.add_argument("--doc-path", default="pubmed.pkl")
    parser.add_argument("--query-path", default="question.pkl")

    parser.add_argument("--target-accuracy", type=float, default=0.8)
    parser.add_argument("--encode-model", type=str, default="Qwen3-Embedding-8B")
    parser.add_argument("--queries", type=int, nargs="*", default=None)
    parser.add_argument("--budget", type=int, default=1000)
    parser.add_argument("--mini-batches", type=int, default=10)
    parser.add_argument("--first-sample", type=int, default=400)
    parser.add_argument("--round-budget", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--training-steps", type=int, default=225)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--supcon-weight", type=float, default=0.2)
    parser.add_argument("--smooth-weight", type=float, default=0.1)
    parser.add_argument("--eta1", type=float, default=0.08)
    parser.add_argument("--eta2", type=float, default=0.08)
    parser.add_argument("--eta3", type=float, default=0.09)
    parser.add_argument("--smooth-mode", default="unlabeled")
    parser.add_argument("--global-loss", action="store_true")
    parser.add_argument("--edge-weight", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--unlabeled-temp", type=float, default=None)
    parser.add_argument("--knn-k", type=int, default=200)
    parser.add_argument("--nsg-R", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    # offline phase

    document_contents = load_pickle(args.doc_path)
    document_embeddings = encode_documents(args.doc_path, args.encode_model)

    device = torch.device(args.device) if args.device else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    doc_dim = int(np.asarray(document_embeddings).shape[1])

    print(f"Config: budget={args.budget}, first_sample={args.first_sample}, round_budget={args.round_budget}, "
          f"hidden={args.hidden_dim}, layers={args.num_layers}, "
          f"global_loss={args.global_loss}, smooth=({args.smooth_weight}, {args.smooth_mode}), T={args.temperature}")

    print("Building NSG graph...")
    graph = GraphIndex(document_embeddings, metric="cosine", knn_k=args.knn_k, nsg_R=args.nsg_R, candidate_L=100)
    graph.construct_graph_index()
    data = build_pyg_data(graph.graph, num_nodes=graph.num_nodes, make_undirected=True)
    doc_tensor = torch.as_tensor(normalized_docs(document_embeddings), dtype=torch.float32)

    config = QueryTimeGNNConfig(
        hidden_dim=args.hidden_dim, doc_dim=doc_dim,
        num_layers=args.num_layers, lr=args.lr,
        training_steps=args.training_steps, seed=args.seed,
        label_smoothing=args.label_smoothing,
        supcon_weight=args.supcon_weight,
        smooth_weight=args.smooth_weight,
        smooth_mode=args.smooth_mode,
        global_loss=args.global_loss,
        graph_feat_dim=3,
        use_edge_weight=args.edge_weight,
    )

    api_key = os.environ["OPENROUTER_API_KEY"]
    oracle = LLM_Oracle(api_key=api_key, max_workers=10)

    # online phase

    query_content = load_pickle(args.query_path)
    query_embedding = encode_query(args.query_path, args.encode_model)

    # Round 0: graph-coverage initial sample
    batches = graph_uniform_minibatches(graph.graph, graph.num_nodes, args.first_sample, args.mini_batches, seed=args.seed)
    oracle_annotated_labels = merge_batch_labels(batches, document_contents, query_content, oracle, gt)
    pos = sum(1 for v in oracle_annotated_labels.values() if v)
    print(f"Oracle (Round 0): total={len(oracle_annotated_labels)}, pos={pos}, neg={len(oracle_annotated_labels)-pos}")

    raw_scores = cosine_scores(document_embeddings, query_embedding)
    raw_min, raw_max = float(raw_scores.min()), float(raw_scores.max())
    qs = (raw_scores - raw_min) / (raw_max - raw_min + 1e-12)
    qs_tensor = torch.as_tensor(qs, dtype=torch.float32)

    # Train + predict Round 0
    model, history = train_query_time_gnn(data, batches, oracle_annotated_labels, config, device, doc_embeddings=doc_tensor, query_scores=qs_tensor)
    probs = predict_all(model, data, oracle_annotated_labels, device, doc_embeddings=doc_tensor, query_scores=qs_tensor, temperature=args.temperature, unlabeled_temperature=args.unlabeled_temp)

    print(f"  [Round 0] ")
    print(f"  Loss: first={history[0]:.4f}, last={history[-1]:.4f}")

    # Round 1..N: uncertain sampling loop
    round_id = 1
    round_budget = args.round_budget
    old_probs = probs.copy()

    while len(oracle_annotated_labels) < args.budget:
        remaining_budget = args.budget - len(oracle_annotated_labels)
        round_budget = min(args.round_budget, remaining_budget)
        prev_probs = np.asarray(probs, dtype=np.float64).copy()

        new_nodes = uncertain_sample(probs, oracle_annotated_labels, graph.num_nodes, budget=round_budget, eta1=args.eta1)

        if len(new_nodes) == 0:
            print(f"  [Round {round_id}] no uncertain nodes found, stop.")
            break

        new_batches = merge_oracle_newnodes_to_batches(oracle_annotated_labels, new_nodes, graph.num_nodes, args.mini_batches, seed=args.seed)

        new_labels = llm_oracle_labeling(orcale, new_nodes, document_contents, query_content)
        oracle_annotated_labels.update(new_labels)

        pos = sum(1 for v in oracle_annotated_labels.values() if v)
        print(f"  [Round {round_id}] +{len(new_nodes)} nodes → Oracle: total={len(oracle_annotated_labels)}, pos={pos}, neg={len(oracle_annotated_labels)-pos}")

        model, history = train_query_time_gnn(data, new_batches, oracle_annotated_labels, config, device, doc_embeddings=doc_tensor, query_scores=qs_tensor)
        probs = predict_all(model, data, oracle_annotated_labels, device, doc_embeddings=doc_tensor, query_scores=qs_tensor, temperature=args.temperature, unlabeled_temperature=args.unlabeled_temp)

        metrics = compute_ranking_stability_metrics(probs=probs, prev_probs=prev_probs, num_pairs=200_000, seed=args.seed)

        if metrics["score_delta"] < args.eta2 and metrics["rank_flip_rate"] < args.eta3:
            print("Converged, stop iterations")
            break
            
        round_id += 1

    # Final evaluation
    oracle_nodes = set(oracle_annotated_labels.keys())

    # --- Threshold selection (second phase) ---
    print(f"\n{'=' * 60}")
    print("Threshold selection")
    print(f"{'=' * 60}")

    alpha_value = args.target_accuracy
    total_final_results, total_llm_cost = [], []
    all_fig_vars = []
    total_score = []

    sd_total_llm_cost  =[]
    sd_total_final_result = []

    sample_list = [0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]
    last_sample_idx = {}
    last_l = 0
    last_r = 1
    # iteratively add samples
    for sample_idx, sample_ratio in enumerate(sample_list):
        if sample_idx == 0:
            l, r, fig_var, samps = progressive_calibration(probs, oracle, last_sample_idx, oracle_nodes, document_contents, query_content, alpha=args.target_accuracy, last_sample_number=0, sample_number=sample_ratio)
        else:
            l, r, fig_var, samps = progressive_calibration(probs, oracle, last_sample_idx, oracle_nodes, document_contents, query_content, alpha=args.target_accuracy, last_sample_number=sample_list[sample_idx - 1], sample_number=sample_ratio)

        last_sample_idx = fig_var["sample_labels"]
        if l == last_l and r == last_r:
            break
        last_l = l
        last_r = r
    
    final_result, llm_cost = LLM_final_judge(l, r, probs, oracle_annotated_labels, last_sample_idx, oracle, document_contents, query_content)

    print(f"The llm invocation times is {llm_cost} under {len(document_contents)} documents.")


if __name__ == "__main__":
    main()
