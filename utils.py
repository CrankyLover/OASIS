import pickle
from oracle import LLM_Oracle
import numpy as np
from typing import Dict, Tuple, List, Set, Optional


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def cosine_scores(docs, query):
    docs = np.asarray(docs, dtype=np.float32)
    query = np.asarray(query, dtype=np.float32).reshape(-1)
    docs = docs / np.maximum(np.linalg.norm(docs, axis=1, keepdims=True), 1e-12)
    query = query / max(float(np.linalg.norm(query)), 1e-12)
    return docs @ query


def normalized_docs(docs):
    docs = np.asarray(docs, dtype=np.float32)
    return docs / np.maximum(np.linalg.norm(docs, axis=1, keepdims=True), 1e-12)


def llm_oracle_labeling(
    oracle,
    new_nodes: Set[int],
    document_contents: List[str],
    query_content: str
) -> Dict[int, bool]:

    node_ids = list(new_nodes)
    docs = [document_contents[i] for i in node_ids]
    local_labels = oracle.LLM_call(docs, query_content)
    new_labels = {node_ids[idx]: label for idx, label in local_labels.items()}

    return new_labels
