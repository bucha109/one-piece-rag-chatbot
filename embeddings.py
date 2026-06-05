import os
from pathlib import Path
import umap
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import numpy as np
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

from chunking import load_and_chunk, Chunk
from config import LLM_MODEL, EMBED_MODEL, RAW_DIR, CHROMA_PATH, EMBED_DIMS, COLLECTION_NAME


# Setup clients first, note we want chroma to be persistent 
load_dotenv(find_dotenv(usecwd=True)) 
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    print("Warning: OPENROUTER_API_KEY not found in environment")
    client = None
else:
    ai_client = OpenAI( api_key=api_key, base_url="https://openrouter.ai/api/v1")
    print("OpenRouter client created successfully!")


chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

openai_ef = OpenAIEmbeddingFunction(
    api_key=ai_client.api_key,
    model_name=EMBED_MODEL,
    dimensions=EMBED_DIMS,
    api_base="https://openrouter.ai/api/v1"
)

# See if the collection exists, and if it doesn't, then load the data into it after you create it
try:
    collection = chroma_client.get_collection(name=COLLECTION_NAME,
                 embedding_function = openai_ef)
except:
    collection = chroma_client.create_collection(name=COLLECTION_NAME,
                embedding_function=openai_ef)



def index_chunks(collection: chromadb.Collection, chunks: list[Chunk]) -> None:
    """
    Add only new chunks to ChromaDB. Existing chunk IDs are skipped, so rerunning
    the notebook does not overwrite existing chunks or recompute embeddings for them.
    """
    
    ids = [f"{c.source}__{c.chunk_id}" for c in chunks] # example: if c = {"source": "paper.pdf", "chunk_idx": 3} -> "paper.pdf__3"

    existing = collection.get(ids=ids, include=[])  # check which ids exist in collection; include=[] returns only IDs, no documents or embeddings
    existing_ids = set(existing["ids"])             # set for O(1) membership lookup

    # filter only new chunks
    new_chunks = [
        (chunk_id, chunk)
        for chunk_id, chunk in zip(ids, chunks)
        if chunk_id not in existing_ids
    ]

    if not new_chunks:
        print("[index] No new chunks to index.")
        return
    
    # add only new chunks to collection, existing remaining changes -> embeddings are automatically created by ChromaDB using the collection's embedding function when we call add() with new documents.
    collection.add(
        ids=[chunk_id for chunk_id, _ in new_chunks],
        documents=[chunk.text for _, chunk in new_chunks],
        metadatas=[{"source": chunk.source, "section": chunk.section,
         "subsection": chunk.subsection, "chunk_id": chunk.chunk_id} for _, chunk in new_chunks],
    )

    print(f"[index] {len(new_chunks)} new chunks indexed into ChromaDB.")



def retrieve(
    collection: chromadb.Collection,
    query: str,
    top_k: int,
) -> list[dict]:
    
    """
    Return the top_k most semantically similar chunks for a given query.
 
    Returns:
        List of dicts: [{"text": str, "source": str}, ...]
    """
    
    results = collection.query(query_texts=[query], n_results=top_k) 
    #note: query must be passed as a list. query -> embedding function, then similarity search against chunk embeddings, 
    # returning top_k results with their documents and metadata (source). 
    return [
        {"text": doc, "source": meta["source"], "section": meta["section"], "subsection": meta["subsection"], "chunk_id": meta["chunk_id"]}
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]) 
        # ["documents"][0] ->first query's results (list of txt chunks), 
        # ["metadatas"][0] ->first query's metadatas (list of dicts with "source" key))
    ]



def wrap_text(text: str, width: int = 50) -> str:
    words = text.split()
    lines, current = [], []
    for word in words:
        if sum(len(w) for w in current) + len(current) + len(word) > width:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return "<br>".join(lines)

def visualize_embeddings_3d(
    collection: chromadb.Collection,
    n_clusters: int = 10,
) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Fetch stored embeddings, cluster them, and render an interactive 3-D UMAP scatter plot.

    Parameters
  
    collection : chromadb.Collection
        A populated ChromaDB collection returned by main().
    n_clusters : int
        Number of k-means clusters. Default 10 (one per source document) is a reasonable
        starting point; increase to expose finer-grained topic separation.

    Returns
  
    labels : np.ndarray, shape (n_chunks,)
    sources : list[str]
    documents : list[str]
    """


    # ── 1. Pull embeddings from ChromaDB 
    
    result     = collection.get(include=["embeddings", "documents", "metadatas"])
    embeddings = np.array(result["embeddings"])        # (n_chunks, 3072)
    sources    = [m["source"] for m in result["metadatas"]]
    documents  = result["documents"]
    n          = embeddings.shape[0]

    print(f"[viz] {n} vectors, {embeddings.shape[1]} dims")

    # ── 2. K-means on raw embeddings 
    
    print(f"[viz] fitting k-means with {n_clusters} clusters…")
    km     = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = km.fit_predict(embeddings)                # (n_chunks,)

    # ── 3. UMAP → dimensional reduction 
    
    print("[viz] running UMAP 3072 → 3D…")
    coords = umap.UMAP(
        n_components=3,  # 3 dim
        n_neighbors=15, # each point looks at 15 nearest neighbors to understand local structure
        min_dist=0.1, # min distance between points
        random_state=42,
    ).fit_transform(embeddings) # (n_chunks, 3)

    # ── 4. Plotly interactive 3-D scatter 
    
    hover_text = [
        f"<b>Cluster {lbl}</b> | {src}<br>{wrap_text(doc[:150].replace(chr(10), ' '))}…"
        for lbl, src, doc in zip(labels, sources, documents)
    ]

    fig = go.Figure(data=[go.Scatter3d(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        mode="markers",
        marker=dict(
            size=4,
            color=labels.tolist(),
            colorscale="Turbo",
            opacity=0.85,
            colorbar=dict(title="Cluster", thickness=14),
        ),
        text=hover_text,
        hoverinfo="text",
    )])

    fig.update_layout(
        title=f"Embedding Space — UMAP 3D, {n_clusters} k-means clusters ({n} chunks)",
        scene=dict(xaxis_title="Dim 1", yaxis_title="Dim 2", zaxis_title="Dim 3"),
        margin=dict(l=0, r=0, b=0, t=50),
        height=700,
    )

    return fig



if __name__ == '__main__':
    chunks = load_and_chunk(RAW_DIR)
    index_chunks(collection, chunks)