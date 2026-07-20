import numpy as np
from typing import List, Dict, Optional, Tuple


class GraphIndex:
    """
    Graph-based index for document embeddings.

    This class is designed as a lightweight NSG-style graph index.
    Each document is represented as a vertex, and graph edges connect
    semantically similar documents.

    The full NSG construction can be implemented incrementally:
    1. Build an initial kNN graph.
    2. Select a navigating node.
    3. Generate candidate neighbors.
    4. Apply NSG-style neighbor pruning.
    5. Ensure graph connectivity.
    """

    def __init__(
        self,
        document_embeddings: List[np.ndarray],
        metric: str = "cosine",
        knn_k: int = 200,
        nsg_R: int = 50,
        candidate_L: int = 100,
        normalize: bool = True,
        random_seed: int = 42,
    ):
       
        self.metric = metric
        self.knn_k = knn_k
        self.nsg_R = nsg_R
        self.candidate_L = candidate_L
        self.normalize = normalize
        self.random_seed = random_seed

        self.embeddings: np.ndarray = self._prepare_embeddings(document_embeddings)
        self.num_nodes: int = self.embeddings.shape[0]
        self.dim: int = self.embeddings.shape[1]

        # Initial kNN graph:
        # self.knn_graph[i] = List of neighbor ids of node i
        self.knn_graph: Optional[Dict[int, List[int]]] = None

        # Final NSG-style graph:
        # self.graph[i] = List of neighbor ids of node i
        self.graph: Dict[int, List[int]] = {}

        self.edge_positive_weights: Dict[Tuple[int, int], float] = {}
        self.edge_negative_weights: Dict[Tuple[int, int], float] = {}
        self.edge_risk_weights: Dict[Tuple[int, int], float] = {}
        self.transfrom_matrix: Dict[int, List[Tuple[int, float]]] = {}
        self.query_scores: np.ndarray

        # Navigating node, usually a medoid or approximate center
        self.entry_point: Optional[int] = None

        np.random.seed(self.random_seed)

    def construct_graph_index(self) -> None:

        print("start building NSG index")
        self.knn_graph = self._build_initial_knn_graph()
        print("finish building initial knn graph")
        self.entry_point = self._select_entry_point()
        print("finish selecting entry point")
        self.graph = self._build_nsg_graph()
        print("finish building graph")
        self._ensure_connectivity()
        self.symmetrize_graph()
        print("finish building NSG index")

    # ------------------------------------------------------------------
    # Embedding preprocessing
    # ------------------------------------------------------------------

    def _prepare_embeddings(self, document_embeddings: List[np.ndarray]) -> np.ndarray:
        embeddings = np.asarray(document_embeddings, dtype=np.float32)

        if embeddings.ndim != 2:
            raise ValueError(
                f"document_embeddings should be a 2-D array-like object, "
                f"but got shape {embeddings.shape}"
            )

        if self.normalize:
            embeddings = self._l2_normalize(embeddings)

        return embeddings

    def _l2_normalize(self, x: np.ndarray, eps: float = 1e-12) -> np.ndarray:

        norm = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.maximum(norm, eps)


    def _similarity(self, i: int, j: int) -> float:
  
        if i < 0 or i >= self.num_nodes:
            raise IndexError(f"Node id i={i} is out of range.")
        if j < 0 or j >= self.num_nodes:
            raise IndexError(f"Node id j={j} is out of range.")

        vi = self.embeddings[i]
        vj = self.embeddings[j]

        if self.metric == "cosine":
            if self.normalize:
                return float(np.dot(vi, vj))

            denom = np.linalg.norm(vi) * np.linalg.norm(vj)
            if denom <= 1e-12:
                return 0.0
            return float(np.dot(vi, vj) / denom)

        elif self.metric == "l2":
            # Return negative distance so that larger score means more similar.
            return -float(np.linalg.norm(vi - vj))

        else:
            raise ValueError(f"Unsupported metric: {self.metric}")


    def _batch_similarity(self, query_vec: np.ndarray, candidate_ids: np.ndarray) -> np.ndarray:

        query_vec = np.asarray(query_vec, dtype=np.float32).reshape(-1)

        if query_vec.shape[0] != self.dim:
            raise ValueError(
                f"query_vec has dim {query_vec.shape[0]}, "
                f"but expected dim {self.dim}."
            )

        candidate_ids = np.asarray(candidate_ids, dtype=np.int64)

        if candidate_ids.ndim != 1:
            raise ValueError("candidate_ids should be a 1-D array.")

        if len(candidate_ids) == 0:
            return np.asarray([], dtype=np.float32)

        if np.any(candidate_ids < 0) or np.any(candidate_ids >= self.num_nodes):
            raise IndexError("candidate_ids contains node id out of range.")

        candidate_vecs = self.embeddings[candidate_ids]

        if self.metric == "cosine":
            if self.normalize:
                q = query_vec
                q_norm = np.linalg.norm(q)
                if q_norm > 1e-12:
                    q = q / q_norm
                return candidate_vecs @ q

            q_norm = np.linalg.norm(query_vec)
            c_norm = np.linalg.norm(candidate_vecs, axis=1)
            denom = np.maximum(q_norm * c_norm, 1e-12)
            return (candidate_vecs @ query_vec) / denom

        elif self.metric == "l2":
            # Return negative L2 distance so that larger score means more similar.
            diff = candidate_vecs - query_vec[None, :]
            return -np.linalg.norm(diff, axis=1)

        else:
            raise ValueError(f"Unsupported metric: {self.metric}")


    def _build_initial_knn_graph(self) -> Dict[int, List[int]]:

        if self.knn_k <= 0:
            raise ValueError("knn_k should be positive.")

        if self.knn_k >= self.num_nodes:
            raise ValueError(
                f"knn_k={self.knn_k} should be smaller than num_nodes={self.num_nodes}."
            )

        knn_graph: Dict[int, List[int]] = {}

        if self.metric == "cosine":
            # Since embeddings are normalized by default, cosine similarity
            # is just matrix multiplication.
            sim_matrix = self.embeddings @ self.embeddings.T

            # Remove self-loop by setting diagonal to -inf.
            np.fill_diagonal(sim_matrix, -np.inf)

            for i in range(self.num_nodes):
                # Get top-k candidate ids without fully sorting all nodes.
                candidate_ids = np.argpartition(
                    -sim_matrix[i],
                    kth=self.knn_k
                )[:self.knn_k]

                # Sort these k candidates by descending similarity.
                candidate_ids = candidate_ids[
                    np.argsort(-sim_matrix[i][candidate_ids])
                ]

                knn_graph[i] = candidate_ids.astype(int).tolist()

            return knn_graph

        elif self.metric == "l2":
            # Exact L2 construction in batches to avoid storing a full
            # [n, n, dim] tensor.
            batch_size = 512

            for start in range(0, self.num_nodes, batch_size):
                end = min(start + batch_size, self.num_nodes)
                batch = self.embeddings[start:end]

                # dist_sq shape: [batch_size, num_nodes]
                # dist_sq = ||x||^2 + ||y||^2 - 2xy
                batch_norm = np.sum(batch * batch, axis=1, keepdims=True)
                all_norm = np.sum(self.embeddings * self.embeddings, axis=1, keepdims=True).T
                dist_sq = batch_norm + all_norm - 2.0 * (batch @ self.embeddings.T)
                dist_sq = np.maximum(dist_sq, 0.0)

                for local_i, global_i in enumerate(range(start, end)):
                    dist_sq[local_i, global_i] = np.inf

                    candidate_ids = np.argpartition(
                        dist_sq[local_i],
                        kth=self.knn_k
                    )[:self.knn_k]

                    candidate_ids = candidate_ids[
                        np.argsort(dist_sq[local_i][candidate_ids])
                    ]

                    knn_graph[global_i] = candidate_ids.astype(int).tolist()

            return knn_graph

        else:
            raise ValueError(f"Unsupported metric: {self.metric}")


    def _select_entry_point(self) -> int:
     
        centroid = np.mean(self.embeddings, axis=0)

        if self.metric == "cosine":
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm > 1e-12:
                centroid = centroid / centroid_norm

            scores = self.embeddings @ centroid
            return int(np.argmax(scores))

        elif self.metric == "l2":
            diff = self.embeddings - centroid[None, :]
            dist = np.linalg.norm(diff, axis=1)
            return int(np.argmin(dist))

        else:
            raise ValueError(f"Unsupported metric: {self.metric}")


    def _build_nsg_graph(self) -> Dict[int, List[int]]:
       
        if self.knn_graph is None:
            raise RuntimeError(
                "self.knn_graph is None. Please call _build_initial_knn_graph() first."
            )

        if self.nsg_R <= 0:
            raise ValueError("nsg_R should be positive.")

        graph: Dict[int, List[int]] = {}

        for node_id in range(self.num_nodes):
            candidates = self._get_candidate_neighbors(node_id)
            selected_neighbors = self._prune_neighbors(node_id, candidates)
            graph[node_id] = selected_neighbors

        return graph
    

    def _get_candidate_neighbors(self, node_id: int) -> List[int]:
      
        if self.knn_graph is None:
            raise RuntimeError("self.knn_graph is None.")

        if node_id < 0 or node_id >= self.num_nodes:
            raise IndexError(f"node_id={node_id} is out of range.")

        candidate_set = set()

        # 1. Direct kNN neighbors.
        for nbr in self.knn_graph.get(node_id, []):
            if nbr != node_id:
                candidate_set.add(nbr)

        # 2. Second-order neighbors.
        for nbr in self.knn_graph.get(node_id, []):
            for nbr2 in self.knn_graph.get(nbr, []):
                if nbr2 != node_id:
                    candidate_set.add(nbr2)

        if not candidate_set:
            return []

        candidates = np.asarray(list(candidate_set), dtype=np.int64)

        # Rank candidates by similarity to node_id.
        query_vec = self.embeddings[node_id]
        scores = self._batch_similarity(query_vec, candidates)

        if len(candidates) > self.candidate_L:
            top_ids = np.argpartition(
                -scores,
                kth=self.candidate_L - 1
            )[:self.candidate_L]

            candidates = candidates[top_ids]
            scores = scores[top_ids]

        # Sort selected candidates by descending similarity.
        order = np.argsort(-scores)
        candidates = candidates[order]

        return candidates.astype(int).tolist()
    

    def _prune_neighbors(self, node_id: int, candidates: List[int]) -> List[int]:

        if node_id < 0 or node_id >= self.num_nodes:
            raise IndexError(f"node_id={node_id} is out of range.")

        # Remove self-loop and duplicates while preserving order.
        seen = set()
        unique_candidates = []
        for c in candidates:
            if c == node_id:
                continue
            if c not in seen:
                seen.add(c)
                unique_candidates.append(c)

        if not unique_candidates:
            return []

        # Ensure candidates are sorted by similarity to node_id.
        cand_arr = np.asarray(unique_candidates, dtype=np.int64)
        scores = self._batch_similarity(self.embeddings[node_id], cand_arr)
        order = np.argsort(-scores)
        sorted_candidates = cand_arr[order].astype(int).tolist()

        selected: List[int] = []

        def distance_from_similarity(sim: float) -> float:

            if self.metric == "cosine":
                # For normalized cosine, distance in [0, 2].
                return 1.0 - sim
            elif self.metric == "l2":
                # self._similarity for l2 returns negative L2 distance.
                return -sim
            else:
                raise ValueError(f"Unsupported metric: {self.metric}")

        for cand in sorted_candidates:
            if len(selected) >= self.nsg_R:
                break

            dist_node_cand = distance_from_similarity(
                self._similarity(node_id, cand)
            )

            occluded = False

            for sel in selected:
                dist_sel_cand = distance_from_similarity(
                    self._similarity(sel, cand)
                )

                # If candidate is closer to an already selected neighbor
                # than to the current node, it is covered by that neighbor.
                if dist_sel_cand < dist_node_cand:
                    occluded = True
                    break

            if not occluded:
                selected.append(cand)

        # Fallback: if pruning is too aggressive, keep nearest candidates.
        if len(selected) == 0 and len(sorted_candidates) > 0:
            selected.append(sorted_candidates[0])

        return selected


    def _ensure_connectivity(self) -> None:

        if self.entry_point is None:
            raise RuntimeError("self.entry_point is None.")

        if not self.graph:
            raise RuntimeError("self.graph is empty.")

        def dfs_from_entry() -> set:
            visited = set()
            stack = [self.entry_point]

            while stack:
                u = stack.pop()
                if u in visited:
                    continue
                visited.add(u)

                for v in self.graph.get(u, []):
                    if v not in visited:
                        stack.append(v)

            return visited

        def add_edge_with_pruning(src: int, dst: int) -> None:

            if src == dst:
                return

            neighbors = self.graph.get(src, [])

            if dst not in neighbors:
                neighbors = neighbors + [dst]

            if len(neighbors) > self.nsg_R:
                neighbors = self._prune_neighbors(src, neighbors)

                # It is possible that pruning removes dst again.
                # For connectivity repair, force keeping dst if needed.
                if dst not in neighbors:
                    if len(neighbors) >= self.nsg_R:
                        # Replace the weakest neighbor with dst.
                        scores = np.asarray(
                            [self._similarity(src, x) for x in neighbors],
                            dtype=np.float32
                        )
                        weakest_pos = int(np.argmin(scores))
                        neighbors[weakest_pos] = dst
                    else:
                        neighbors.append(dst)

            self.graph[src] = neighbors

        visited = dfs_from_entry()

        # Repeat until all nodes are reachable.
        # In practice, this usually finishes in a few iterations.
        while len(visited) < self.num_nodes:
            unreachable = [i for i in range(self.num_nodes) if i not in visited]
            reachable = np.asarray(list(visited), dtype=np.int64)

            for u in unreachable:
                # Find the nearest currently reachable node.
                scores = self._batch_similarity(self.embeddings[u], reachable)
                nearest_reachable = int(reachable[int(np.argmax(scores))])

                # Add bidirectional edges.
                # nearest_reachable -> u makes u reachable from entry.
                # u -> nearest_reachable improves local navigability.
                add_edge_with_pruning(nearest_reachable, u)
                add_edge_with_pruning(u, nearest_reachable)

                visited.add(u)

            visited = dfs_from_entry()

    def symmetrize_graph(self) -> None:

        new_graph = {i: set(neighbors) for i, neighbors in self.graph.items()}

        for i, neighbors in self.graph.items():
            new_graph.setdefault(i, set())
            for j in neighbors:
                new_graph.setdefault(j, set())
                new_graph[i].add(j)
                new_graph[j].add(i)

        self.graph = {i: sorted(list(neighbors)) for i, neighbors in new_graph.items()}


    def get_neighbors(self, node_id: int) -> List[int]:
        """
        Return neighbors of a given node in the final graph.
        """
        if node_id not in self.graph:
            raise ValueError(f"Node {node_id} is not in the graph.")
        return self.graph[node_id]

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        search_L: Optional[int] = None,
    ) -> List[Tuple[int, float]]:

        raise NotImplementedError

    def save(self, path: str) -> None:

        raise NotImplementedError

    @classmethod
    def load(cls, path: str) -> "GraphIndex":

        raise NotImplementedError
        