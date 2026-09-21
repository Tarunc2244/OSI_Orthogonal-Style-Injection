# src/baselines/instruction_prompt.py
import torch
import os
import pandas as pd
from tqdm import tqdm
from utils.language_utils import LANGUAGE_CONFIGS, post_prune_generated_text
import config as cfg

def translate_instruction_prompt(text: str, character: str, tokenizer, model, language: str) -> str:
    """
    Instruction prompt that asks to preserve style, tone, and register,
    but does NOT mention the character's name.
    """
    lang_name = LANGUAGE_CONFIGS[language]['name']
    
    # Generic style‑preservation instruction (no character name)
    system_msg = (
        "You are a professional translator of movie dialogues. "
        f"Translate the given English dialogue to {lang_name}, "
        "preserving the original style, tone, and register as much as possible."
    )
    user_msg = f"English: {text}\n{lang_name}:"

    # Build messages according to model's chat template
    model_name_lower = cfg.MODEL_NAME.lower()
    if "llama" in model_name_lower or "qwen" in model_name_lower:
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
    else:  # gemma and others may not have system role – combine into user
        messages = [
            {"role": "user", "content": f"{system_msg}\n\n{user_msg}"}
        ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=cfg.MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    raw = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return post_prune_generated_text(raw, language)

def run_instruction_prompt(
    input_csv: str,
    output_dir: str,
    language: str,
    has_reference: bool,
    tokenizer,
    model,
    model_prefix: str = "llama",
):
    os.makedirs(output_dir, exist_ok=True)
    out_csv = os.path.join(output_dir, f"{model_prefix}_{language}_instruction_prompt.csv")
    target_col = LANGUAGE_CONFIGS[language]['column_suffix']

    df = pd.read_csv(input_csv)
    if has_reference:
        df = df.dropna(subset=["english_dialogue", target_col])
    else:
        df = df.dropna(subset=["english_dialogue"])

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"InstrPrompt {language}"):
        text = row["english_dialogue"]
        char = row["character_name"]  # unused in the new prompt, but kept for compatibility
        ref = row[target_col] if has_reference else ""
        hyp = translate_instruction_prompt(text, char, tokenizer, model, language)
        results.append({
            'character': char,
            'source': text,
            'hypothesis': hyp,
            'reference': ref if has_reference else ''
        })

    pd.DataFrame(results).to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f"✓ Saved {out_csv}")