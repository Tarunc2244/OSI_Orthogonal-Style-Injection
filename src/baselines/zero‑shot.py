import torch
import os
import pandas as pd
from tqdm import tqdm
from utils.language_utils import LANGUAGE_CONFIGS, post_prune_generated_text
import config as cfg

def translate_vanilla(text: str, tokenizer, model, language: str) -> str:
    """Zero‑shot translation."""
    lang_name = LANGUAGE_CONFIGS[language]['name']
    messages = [
        {"role": "system", "content": "You are a helpful translation assistant. Provide only the translated sentence, no extra text."},
        {"role": "user", "content": f"Translate the following English dialogue into \n {lang_name}: {text}"}
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

def run_vanilla(
    input_csv: str,
    output_dir: str,
    language: str,
    has_reference: bool,
    tokenizer,
    model,
    model_prefix: str = "llama",
):
    """Run vanilla baseline for one language and save results."""
    os.makedirs(output_dir, exist_ok=True)
    out_csv = os.path.join(output_dir, f"{model_prefix}_{language}_vanilla.csv")
    target_col = LANGUAGE_CONFIGS[language]['column_suffix']

    df = pd.read_csv(input_csv)
    if has_reference:
        df = df.dropna(subset=["english_dialogue", target_col])
    else:
        df = df.dropna(subset=["english_dialogue"])

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Vanilla {language}"):
        text = row["english_dialogue"]
        char = row["character_name"]
        ref = row[target_col] if has_reference else ""
        hyp = translate_vanilla(text, tokenizer, model, language)
        results.append({
            'character': char,
            'source': text,
            'hypothesis': hyp,
            'reference': ref if has_reference else ''
        })

    pd.DataFrame(results).to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f"✓ Saved {out_csv}")