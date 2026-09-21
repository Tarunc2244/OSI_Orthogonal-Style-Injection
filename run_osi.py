# run_osi.py
import os
import argparse
import pandas as pd
import torch
from tqdm import tqdm

import config as cfg
from utils.language_utils import LANGUAGE_CONFIGS
from utils.model_utils import load_model, detect_languages_with_references
from utils.filtering import filter_dialogues
from src.osi.build_vectors import (
    build_character_embeddings,
    compute_style_axes_pca,
    compute_contrastive_vectors,
    compute_cosine_similarity_gap,
)
from src.osi.osi_core import translate_with_osi


def run_osi(
    input_csv: str,
    output_dir: str,
    language: str,
    has_reference: bool,
    tokenizer=None,
    model=None,
    alpha: float = cfg.ALPHA,
    multi_axis: bool = cfg.USE_MULTI_AXIS,
    n_axes: int = cfg.N_STYLE_AXES,
    contrastive: bool = cfg.USE_CONTRASTIVE,
    apply_filtering: bool = cfg.APPLY_FILTERING,
    top_percent: float = cfg.TOP_PERCENT,
    min_lines: int = cfg.MIN_LINES_PER_CHARACTER,
    max_len: int = cfg.MAX_INPUT_LEN,
    max_new_tokens: int = cfg.MAX_NEW_TOKENS,
    batch_size: int = cfg.BATCH_SIZE_EMBED,
    condition_layer: int = cfg.CONDITION_LAYER_IDX,
):
    os.makedirs(output_dir, exist_ok=True)
    out_csv = os.path.join(output_dir, f"qwen_{language}_osi.csv")

    lang_name = LANGUAGE_CONFIGS[language]['name']
    target_col = LANGUAGE_CONFIGS[language]['column_suffix']

    print(f"\n{'='*60}")
    print(f"Processing: {lang_name} (α={alpha}, multi‑axis={multi_axis}, contrastive={contrastive})")
    print(f"{'='*60}")

    # Load and prepare data
    df_full = pd.read_csv(input_csv)
    df_full.columns = df_full.columns.str.strip().str.lower()
    rename_map = {}
    if "character_name" not in df_full.columns and "character" in df_full.columns:
        rename_map["character"] = "character_name"
    if "english_dialogue" not in df_full.columns and "english" in df_full.columns:
        rename_map["english"] = "english_dialogue"
    df_full = df_full.rename(columns=rename_map)

    required_cols = ["character_name", "english_dialogue"]
    for col in required_cols:
        if col not in df_full.columns:
            raise KeyError(f"Column '{col}' not found. Found: {df_full.columns.tolist()}")

    if has_reference:
        if target_col not in df_full.columns:
            raise KeyError(f"Column '{target_col}' not found")
        df_full = df_full.dropna(subset=["english_dialogue", target_col])
    else:
        df_full = df_full.dropna(subset=["english_dialogue"])

    print(f"✓ Loaded {len(df_full)} total dialogue pairs for translation")

    # Character vector construction
    if apply_filtering:
        df_char_build = filter_dialogues(df_full, top_percent=top_percent, min_lines=min_lines)
        print(f"Using {len(df_char_build)} lines to build character embeddings.")
    else:
        df_char_build = df_full
        print("Using all lines to build character embeddings (no filtering).")

    if tokenizer is None or model is None:
        tokenizer, model = load_model(cfg.MODEL_NAME)

    normalize_char = not contrastive
    char_embeddings = build_character_embeddings(
        df_char_build, tokenizer, model,
        layer_idx=condition_layer,
        max_len=max_len,
        batch_size=batch_size,
        normalize=normalize_char
    )

    if multi_axis:
        style_axes = compute_style_axes_pca(char_embeddings, n_components=n_axes)
    else:
        style_axes = compute_style_axes_pca(char_embeddings, n_components=1)

    if contrastive:
        char_vectors = compute_contrastive_vectors(char_embeddings, normalize=True)
    else:
        char_vectors = {name: emb / (emb.norm() + 1e-8) for name, emb in char_embeddings.items()}

    print("\n--- Validation on character vectors ---")
    compute_cosine_similarity_gap(char_vectors)

    # Translate all dialogues
    print(f"\nGenerating translations for {len(df_full)} dialogues...")
    results = []
    for idx, row in tqdm(df_full.iterrows(), total=len(df_full), desc="Translating"):
        text = row["english_dialogue"]
        char = row["character_name"]
        ref = row[target_col] if has_reference else ""

        vec = char_vectors.get(char)
        if vec is None:
            hyp = translate_with_osi(
                text, None, None, tokenizer, model,
                language=language, alpha=alpha,
                max_new_tokens=max_new_tokens,
                condition_layer=condition_layer
            )
        else:
            hyp = translate_with_osi(
                text, vec, style_axes, tokenizer, model,
                language=language, alpha=alpha,
                max_new_tokens=max_new_tokens,
                condition_layer=condition_layer
            )

        results.append({
            'character': char,
            'source': text,
            'hypothesis': hyp,
            'reference': ref if has_reference else ''
        })

    results_df = pd.DataFrame(results)
    if not has_reference:
        results_df = results_df[['character', 'source', 'hypothesis']]
    results_df.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f"\n✓ Saved {len(results)} translations to {out_csv}")
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OSI for a single language")
    parser.add_argument("--input_csv", type=str, default=cfg.DATASETS["Titanic"])
    parser.add_argument("--output_dir", type=str, default="outputs_qwen_T_Hsem")
    parser.add_argument("--language", type=str, default="bengali")
    parser.add_argument("--has_reference", action="store_true", default=True)
    parser.add_argument("--alpha", type=float, default=cfg.ALPHA)
    parser.add_argument("--multi_axis", action="store_true", default=cfg.USE_MULTI_AXIS)
    parser.add_argument("--n_axes", type=int, default=cfg.N_STYLE_AXES)
    parser.add_argument("--contrastive", action="store_true", default=cfg.USE_CONTRASTIVE)
    parser.add_argument("--apply_filtering", action="store_true", default=cfg.APPLY_FILTERING)
    parser.add_argument("--top_percent", type=float, default=cfg.TOP_PERCENT)
    parser.add_argument("--min_lines", type=int, default=cfg.MIN_LINES_PER_CHARACTER)
    parser.add_argument("--max_len", type=int, default=cfg.MAX_INPUT_LEN)
    parser.add_argument("--max_new_tokens", type=int, default=cfg.MAX_NEW_TOKENS)
    parser.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE_EMBED)
    parser.add_argument("--condition_layer", type=int, default=cfg.CONDITION_LAYER_IDX)
    parser.add_argument("--model_name", type=str, default=cfg.MODEL_NAME)

    args = parser.parse_args()

    # Update config if needed (or just pass args directly)
    cfg.MODEL_NAME = args.model_name
    cfg.CONDITION_LAYER_IDX = args.condition_layer
    cfg.ALPHA = args.alpha
    cfg.USE_MULTI_AXIS = args.multi_axis
    cfg.N_STYLE_AXES = args.n_axes
    cfg.USE_CONTRASTIVE = args.contrastive
    cfg.APPLY_FILTERING = args.apply_filtering
    cfg.TOP_PERCENT = args.top_percent
    cfg.MIN_LINES_PER_CHARACTER = args.min_lines
    cfg.MAX_INPUT_LEN = args.max_len
    cfg.MAX_NEW_TOKENS = args.max_new_tokens
    cfg.BATCH_SIZE_EMBED = args.batch_size

    run_osi(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        language=args.language,
        has_reference=args.has_reference,
        tokenizer=None,
        model=None,
        alpha=args.alpha,
        multi_axis=args.multi_axis,
        n_axes=args.n_axes,
        contrastive=args.contrastive,
        apply_filtering=args.apply_filtering,
        top_percent=args.top_percent,
        min_lines=args.min_lines,
        max_len=args.max_len,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        condition_layer=args.condition_layer,
    )