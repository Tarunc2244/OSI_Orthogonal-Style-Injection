# run_osi_all.py
import os
import argparse
from utils.language_utils import LANGUAGE_CONFIGS
from utils.model_utils import load_model, detect_languages_with_references
from run_osi import run_osi
import config as cfg

def main():
    parser = argparse.ArgumentParser(description="Run OSI for all languages")
    parser.add_argument("--input_csv", type=str, default=cfg.DATASETS["Titanic"])
    parser.add_argument("--output_dir", type=str, default="outputs_qwen_all")
    parser.add_argument("--model_name", type=str, default=cfg.MODEL_NAME)
    # ... add other args as needed, or rely on defaults from config
    args = parser.parse_args()

    # Detect which languages have references
    with_refs, without_refs = detect_languages_with_references(args.input_csv, LANGUAGE_CONFIGS)
    print(f"Languages with references: {with_refs}")
    print(f"Languages without references: {without_refs}")

    # Load model once
    tokenizer, model = load_model(args.model_name)

    for lang in LANGUAGE_CONFIGS.keys():
        has_ref = lang in with_refs
        print(f"\n>>> Processing {lang} (has_reference={has_ref})")
        run_osi(
            input_csv=args.input_csv,
            output_dir=args.output_dir,
            language=lang,
            has_reference=has_ref,
            tokenizer=tokenizer,
            model=model,
            alpha=cfg.ALPHA,
            multi_axis=cfg.USE_MULTI_AXIS,
            n_axes=cfg.N_STYLE_AXES,
            contrastive=cfg.USE_CONTRASTIVE,
            apply_filtering=cfg.APPLY_FILTERING,
            top_percent=cfg.TOP_PERCENT,
            min_lines=cfg.MIN_LINES_PER_CHARACTER,
            max_len=cfg.MAX_INPUT_LEN,
            max_new_tokens=cfg.MAX_NEW_TOKENS,
            batch_size=cfg.BATCH_SIZE_EMBED,
            condition_layer=cfg.CONDITION_LAYER_IDX,
        )

if __name__ == "__main__":
    main()