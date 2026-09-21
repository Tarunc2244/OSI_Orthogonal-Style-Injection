# src/baselines/fewshot.py
import torch
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils.language_utils import LANGUAGE_CONFIGS, post_prune_generated_text
import config as cfg

def build_fewshot_prompt(text: str, character: str, df: pd.DataFrame, language: str, num_shots: int):
    """Build a few‑shot prompt with examples from the same character (no reference leakage)."""
    lang_name = LANGUAGE_CONFIGS[language]['name']

    char_df = df[df["character_name"] == character].copy()
    char_df = char_df[char_df["english_dialogue"] != text]
    available = len(char_df)
    shots = min(num_shots, available)

    prompt = f"You are translating dialogue into {lang_name}. Maintain the speaking style of the character.\n\n"
    prompt += "Here are examples of how this character speaks in English:\n\n"

    if shots > 0:
        examples = char_df.sample(n=shots, random_state=42)
        for _, row in examples.iterrows():
            prompt += f'English: "{row["english_dialogue"]}"\n\n'
            # No translations – prevents reference leakage

    prompt += (
        f"Now translate the following English sentence into {lang_name} with the character's speaking style shown above.\n\n"
    )
    prompt += f'English: "{text}"\n'
    prompt += f"{lang_name}: "
    return prompt

def translate_fewshot(text: str, character: str, df: pd.DataFrame, tokenizer, model, language: str) -> str:
    """Few‑shot translation."""
    prompt = build_fewshot_prompt(text, character, df, language, cfg.FEWSHOT_NUM_SHOTS)
    # Original system message (professional translator)
    messages = [
        {"role": "system", "content": f"You are a professional English to {LANGUAGE_CONFIGS[language]['name']} dialogue translator."},
        {"role": "user", "content": prompt}
    ]
    full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=cfg.MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    raw = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return post_prune_generated_text(raw, language)

def run_fewshot(
    input_csv: str,
    output_dir: str,
    language: str,
    has_reference: bool,
    tokenizer,
    model,
    model_prefix: str = "llama",
):
    os.makedirs(output_dir, exist_ok=True)
    out_csv = os.path.join(output_dir, f"{model_prefix}_{language}_fewshot.csv")
    target_col = LANGUAGE_CONFIGS[language]['column_suffix']

    df = pd.read_csv(input_csv)
    if has_reference:
        df = df.dropna(subset=["english_dialogue", target_col])
    else:
        df = df.dropna(subset=["english_dialogue"])

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"FewShot {language}"):
        text = row["english_dialogue"]
        char = row["character_name"]
        ref = row[target_col] if has_reference else ""
        hyp = translate_fewshot(text, char, df, tokenizer, model, language)
        results.append({
            'character': char,
            'source': text,
            'hypothesis': hyp,
            'reference': ref if has_reference else ''
        })

    pd.DataFrame(results).to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f"✓ Saved {out_csv}")