# osi/build_vectors.py
import torch
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm
from typing import Dict, List, Union

def build_character_embeddings(
    df,
    tokenizer,
    model,
    layer_idx: int,
    max_len: int,
    batch_size: int = 16,
    normalize: bool = True
) -> Dict[str, torch.Tensor]:
    char_embeddings = {}
    print(f"\nBuilding character embeddings from {len(df)} lines...")
    for char in tqdm(df["character_name"].unique(), desc="Characters"):
        lines = df[df["character_name"] == char]["english_dialogue"].tolist()
        clean_lines = [str(t) for t in lines if pd.notna(t) and len(str(t).strip()) > 0]
        if len(clean_lines) == 0:
            continue
        embs = extract_embeddings_batch(
            clean_lines, tokenizer, model, layer_idx, max_len, batch_size
        )
        char_emb = embs.mean(0)
        if normalize:
            char_emb = char_emb / (char_emb.norm() + 1e-8)
        char_embeddings[char] = char_emb
        print(f"  ✓ {char}: {len(clean_lines)} lines")
    print(f"\n✓ Total characters: {len(char_embeddings)}\n")
    return char_embeddings

def compute_style_axes_pca(
    char_embeddings: Dict[str, torch.Tensor],
    n_components: int = 1
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Compute the first n_components principal components (style axes)."""
    print("\nComputing style axes with PCA...")
    all_embs = torch.stack(list(char_embeddings.values()))
    global_mean = all_embs.mean(0, keepdim=True)
    centered = all_embs - global_mean
    emb_np = centered.cpu().numpy()
    pca = PCA(n_components=n_components)
    pca.fit(emb_np)
    explained = pca.explained_variance_ratio_
    print(f"Explained variance: {explained}")
    axes = []
    for i in range(n_components):
        axis = torch.tensor(pca.components_[i], dtype=all_embs.dtype, device=all_embs.device)
        axis = axis / (axis.norm() + 1e-8)
        axes.append(axis)
    if n_components == 1:
        return axes[0]
    return axes

def compute_contrastive_vectors(
    char_embeddings: Dict[str, torch.Tensor],
    normalize: bool = True
) -> Dict[str, torch.Tensor]:
    """Return c - c_avg for each character."""
    all_embs = torch.stack(list(char_embeddings.values()))
    avg = all_embs.mean(0)
    contrastive = {}
    for name, emb in char_embeddings.items():
        vec = emb - avg
        if normalize:
            vec = vec / (vec.norm() + 1e-8)
        contrastive[name] = vec
    return contrastive

def compute_cosine_similarity_gap(char_embeddings: Dict[str, torch.Tensor]) -> float:
    """Mean same‑character similarity minus mean different‑character similarity."""
    names = list(char_embeddings.keys())
    embs = torch.stack([char_embeddings[n] for n in names])
    sim_matrix = embs @ embs.T
    same_sum, same_count, diff_sum, diff_count = 0.0, 0, 0.0, 0
    for i in range(len(names)):
        for j in range(len(names)):
            if i == j:
                continue
            if names[i] == names[j]:
                same_sum += sim_matrix[i, j].item()
                same_count += 1
            else:
                diff_sum += sim_matrix[i, j].item()
                diff_count += 1
    same_mean = same_sum / same_count if same_count else 0
    diff_mean = diff_sum / diff_count if diff_count else 0
    gap = same_mean - diff_mean
    print(f"Cosine similarity gap (same‑char - diff‑char): {gap:.4f}")
    return gap