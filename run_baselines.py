#!/usr/bin/env python3
"""
Run all baselines (vanilla, instruction prompt, few‑shot, additive steering) for given datasets and models.
Allows command‑line customization; defaults to the three movie datasets and both Qwen/Llama models.
"""

import os
import argparse
from utils.language_utils import LANGUAGE_CONFIGS
from utils.model_utils import load_model, detect_languages_with_references
import config as cfg

from src.baselines.zero‑shot import run_vanilla
from src.baselines.instruction_prompt import run_instruction_prompt
from src.baselines.fewshot import run_fewshot
from src.baselines.additive_steering import run_additive_steering


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run all baselines for multiple datasets and models."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        help="Path to input CSV. Can be repeated. If none, uses default list."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/baselines",
        help="Base output directory (default: outputs/baselines)"
    )
    parser.add_argument(
        "--model",
        action="append",
        nargs=2,
        metavar=("MODEL_NAME", "PREFIX"),
        help="Model name and prefix (e.g., --model Qwen/Qwen2.5-7B-Instruct qwen). Can be repeated."
    )
    parser.add_argument(
        "--lang",
        action="append",
        choices=LANGUAGE_CONFIGS.keys(),
        help="Language(s) to process. If none, all languages in config are processed."
    )
    parser.add_argument(
        "--skip_vanilla",
        action="store_true",
        help="Skip vanilla baseline"
    )
    parser.add_argument(
        "--skip_instruction",
        action="store_true",
        help="Skip instruction prompt baseline"
    )
    parser.add_argument(
        "--skip_fewshot",
        action="store_true",
        help="Skip few‑shot baseline"
    )
    parser.add_argument(
        "--skip_additive",
        action="store_true",
        help="Skip additive steering baseline"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine datasets
    if args.dataset:
        datasets = {os.path.basename(p).replace('.csv', ''): p for p in args.dataset}
    else:
        datasets = cfg.DATASETS

    # Determine models
    if args.model:
        models = [(name, prefix) for name, prefix in args.model]
    else:
        models = [
            ("Qwen/Qwen2.5-7B-Instruct", "qwen"),
            ("meta-llama/Meta-Llama-3.1-8B-Instruct", "llama"),
        ]

    # Determine languages
    languages = args.lang if args.lang else list(LANGUAGE_CONFIGS.keys())

    # Baselines to run
    run_vanilla_flag = not args.skip_vanilla
    run_instruction_flag = not args.skip_instruction
    run_fewshot_flag = not args.skip_fewshot
    run_additive_flag = not args.skip_additive

    base_out = args.output_dir

    for dataset_name, csv_path in datasets.items():
        print(f"\n{'#'*60}")
        print(f"# DATASET: {dataset_name}")
        print(f"{'#'*60}")

        # Detect which languages have references in this CSV (once per dataset)
        with_refs, without_refs = detect_languages_with_references(csv_path, LANGUAGE_CONFIGS)
        print(f"Languages with references: {with_refs}")
        print(f"Languages without references: {without_refs}")

        for model_name, model_prefix in models:
            print(f"\n{'='*60}")
            print(f"Running baselines for {model_prefix} on {dataset_name}")
            print(f"{'='*60}")

            tokenizer, model = load_model(model_name)

            for lang in languages:
                has_ref = lang in with_refs
                print(f"\n>>> Processing {lang} (has_reference={has_ref})")

                out_subdir = os.path.join(base_out, model_prefix, dataset_name)
                os.makedirs(out_subdir, exist_ok=True)

                if run_vanilla_flag:
                    run_vanilla(csv_path, out_subdir, lang, has_ref, tokenizer, model, model_prefix)

                if run_instruction_flag:
                    run_instruction_prompt(csv_path, out_subdir, lang, has_ref, tokenizer, model, model_prefix)

                if run_fewshot_flag:
                    run_fewshot(csv_path, out_subdir, lang, has_ref, tokenizer, model, model_prefix)

                if run_additive_flag:
                    run_additive_steering(
                        csv_path, out_subdir, lang, has_ref, tokenizer, model, model_prefix,
                        apply_filtering=cfg.APPLY_FILTERING,
                        top_percent=cfg.TOP_PERCENT,
                        min_lines=cfg.MIN_LINES_PER_CHARACTER,
                        alpha=cfg.ALPHA,
                    )


if __name__ == "__main__":
    main()