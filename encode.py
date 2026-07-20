from vllm import LLM, PoolingParams
from sklearn.preprocessing import normalize
from pathlib import Path
from typing import List
from utils import load_pickle

def encode_documents(doc_path: str, model: str) -> List[np.ndarray]:
    # load the embedding model and get the document embeddings
    document_list = load_pickle(doc_path)

    llm = LLM(model=model, runner="pooling", hf_overrides={"is_matryoshka": True})

    embeddings = [] 
    for doc in document_list:
        (output,) = llm.embed(doc, pooling_params=PoolingParams(normalize=True, dimensions=1024))
        embeds = output.outputs.embedding
        embeds_normalized = normalize([embeds])[0]
        embeddings.append(embeds_normalized)

    return embeddings


def encode_query(query_path: str, model: str) -> np.ndarray:
    # load the embedding model and get the document embeddings
    query = load_pickle(query_path)

    llm = LLM(model=model, runner="pooling", hf_overrides={"is_matryoshka": True})

    (output,) = llm.embed(query, pooling_params=PoolingParams(normalize=True, dimensions=1024))
    embeds = output.outputs.embedding
    embeds_normalized = normalize([embeds])[0]

    return embeds_normalized
