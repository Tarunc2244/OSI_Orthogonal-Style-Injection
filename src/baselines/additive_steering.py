# src/baselines/additive_steering.py
import torch
import torch.nn as nn
import os
import pandas as pd
from tqdm import tqdm
from utils.language_utils import LANGUAGE_CONFIGS, post_prune_generated_text
from utils.model_utils import load_model
from utils.filtering import filter_dialogues, extract_embeddings_batch
import config as cfg

class AdditiveSteeringConditioner(nn.Module):
    """Adds the character vector directly to the hidden state at the first decoding step."""
    def __init__(self, model, layer_idx: int, character_vector: torch.Tensor, alpha: float = cfg.ALPHA):
        super().__init__()
        self.model = model
        self.layer_idx = layer_idx
        self.character_vector = character_vector
        self.alpha = alpha
        self.applied = False

        if hasattr(model, "model") and hasattr(model.model, "layers"):
            self.layers = model.model.layers
        else:
            raise ValueError("Cannot locate transformer layers")

        self._orig_forward = self.layers[self.layer_idx].forward
        self._apply_hook()

    def _apply_hook(self):
        orig_forward = self._orig_forward
        c_vec = self.character_vector
        alpha = self.alpha
        conditioner = self

        def hooked_forward(*args, **kwargs):
            hidden_states = args[0]
            if not conditioner.applied:
                print("➕ Additive steering applied at first decoding step!")
                conditioner.applied = True
                h = hidden_states[:, -1, :]
                c = c_vec.to(h.device, h.dtype)
                h_new = h + alpha * c
                hidden_states = hidden_states.clone()
                hidden_states[:, -1, :] = h_new
                args = (hidden_states,) + args[1:]
            return orig_forward(*args, **kwargs)

        self.layers[self.layer_idx].forward = hooked_forward

    def restore(self):
        self.layers[self.layer_idx].forward = self._orig_forward

def translate_additive(
    text: str,
    character_vector: torch.Tensor,
    tokenizer,
    model,
    language: str,
    alpha: float = cfg.ALPHA,
) -> str:
    """Translate with additive steering."""
    lang_name = LANGUAGE_CONFIGS[language]['name']
    messages = [
        {"role": "system", "content": "You are a helpful translation assistant. Provide only the translated sentence, no extra text."},
        {"role": "user", "content": f"Translate to {lang_name}: {text}"}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    conditioner = AdditiveSteeringConditioner(model, cfg.CONDITION_LAYER_IDX, character_vector, alpha)
    conditioner.applied = False
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=cfg.MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    conditioner.restore()

    raw = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return post_prune_generated_text(raw, language)

def build_character_embeddings_for_baseline(df, tokenizer, model, layer_idx, max_len, batch_size, normalize=True):
    """Simplified version of OSI's build_character_embeddings for baselines."""
    char_embeddings = {}
    for char in df["character_name"].unique():
        lines = df[df["character_name"] == char]["english_dialogue"].tolist()
        clean_lines = [str(t) for t in lines if pd.notna(t) and len(str(t).strip()) > 0]
        if len(clean_lines) == 0:
            continue
        embs = extract_embeddings_batch(clean_lines, tokenizer, model, layer_idx, max_len, batch_size)
        char_emb = embs.mean(0)
        if normalize:
            char_emb = char_emb / (char_emb.norm() + 1e-8)
        char_embeddings[char] = char_emb
    return char_embeddings

def run_additive_steering(
    input_csv: str,
    output_dir: str,
    language: str,
    has_reference: bool,
    tokenizer,
    model,
    model_prefix: str = "llama",
    apply_filtering: bool = cfg.APPLY_FILTERING,
    top_percent: float = cfg.TOP_PERCENT,
    min_lines: int = cfg.MIN_LINES_PER_CHARACTER,
    alpha: float = cfg.ALPHA,
):
    """Run additive steering for one language."""
    os.makedirs(output_dir, exist_ok=True)
    out_csv = os.path.join(output_dir, f"{model_prefix}_{language}_additive.csv")
    target_col = LANGUAGE_CONFIGS[language]['column_suffix']

    df_full = pd.read_csv(input_csv)
    # Normalize column names
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

    # Prepare filtered data for character embeddings
    if apply_filtering:
        df_char_build = filter_dialogues(df_full, top_percent=top_percent, min_lines=min_lines)
        print(f"Using {len(df_char_build)} lines to build character embeddings.")
    else:
        df_char_build = df_full
        print("Using all lines to build character embeddings (no filtering).")

    char_embeddings = build_character_embeddings_for_baseline(
        df_char_build, tokenizer, model,
        layer_idx=cfg.CONDITION_LAYER_IDX,
        max_len=cfg.MAX_INPUT_LEN,
        batch_size=cfg.BATCH_SIZE_EMBED,
        normalize=True
    )

    # Translate
    results = []
    for _, row in tqdm(df_full.iterrows(), total=len(df_full), desc=f"Additive {language}"):
        text = row["english_dialogue"]
        char = row["character_name"]
        ref = row[target_col] if has_reference else ""

        vec = char_embeddings.get(char)
        if vec is None:
            # Fallback to vanilla translation (zero-shot style)
            lang_name = LANGUAGE_CONFIGS[language]['name']
            messages = [
                {"role": "system", "content": "You are a helpful translation assistant."},
                {"role": "user", "content": f"Translate this English dialogue to {lang_name}: {text}"}
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
            hyp = post_prune_generated_text(raw, language)
        else:
            hyp = translate_additive(text, vec, tokenizer, model, language, alpha=alpha)

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
    print(f"✓ Saved {out_csv}")