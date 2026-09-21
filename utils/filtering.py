# utils/filtering.py
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch

def filter_dialogues(df: pd.DataFrame, top_percent: float = 0.4, min_lines: int = 10) -> pd.DataFrame:
    """
    For each character, keep only the top_percent of lines closest to the centroid.
    Uses SentenceTransformer embeddings.
    """
    print("\nApplying centroid‑based filtering to select representative lines...")
    embed_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    print("Encoding dialogues for filtering...")
    embeddings = embed_model.encode(
        df["english_dialogue"].tolist(),
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    df = df.copy()
    df["emb_idx"] = np.arange(len(df))
    filtered_rows = []
    for character in tqdm(df["character_name"].unique(), desc="Filtering characters"):
        char_df = df[df["character_name"] == character]
        if len(char_df) == 0:
            continue
        indices = char_df["emb_idx"].values
        char_embs = embeddings[indices]
        centroid = np.mean(char_embs, axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        sims = cosine_similarity(char_embs, centroid.reshape(1, -1)).flatten()
        char_df = char_df.copy()
        char_df["style_score"] = sims
        keep_n = max(int(len(char_df) * top_percent), min_lines)
        keep_n = min(keep_n, len(char_df))
        char_df = char_df.sort_values("style_score", ascending=False).head(keep_n)
        filtered_rows.append(char_df)
    filtered_df = pd.concat(filtered_rows).reset_index(drop=True)
    filtered_df = filtered_df.drop(columns=["emb_idx", "style_score"], errors='ignore')
    print(f"Filtered from {len(df)} to {len(filtered_df)} lines.")
    return filtered_df

def extract_embeddings_batch(
    texts,
    tokenizer,
    model,
    layer_idx: int,
    max_len: int,
    batch_size: int = 16
) -> torch.Tensor:
    """Extract last‑token hidden states at a given layer for a list of texts."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len
        ).to(model.device)
        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )
            hidden = outputs.hidden_states[layer_idx]  # [B, T, D]
            attention_mask = inputs['attention_mask']
            lengths = attention_mask.sum(dim=1) - 1
            pooled = hidden[torch.arange(hidden.size(0)), lengths]  # last token
            pooled = pooled / (pooled.norm(dim=-1, keepdim=True) + 1e-8)
            all_embeddings.append(pooled)
    return torch.cat(all_embeddings, dim=0)