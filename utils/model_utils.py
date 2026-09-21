# utils/model_utils.py
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_name: str, device_map="auto"):
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.config.use_cache = True
    model.eval()
    print(f"✓ Model loaded on {model.device}")
    return tokenizer, model

def detect_languages_with_references(csv_path: str, lang_configs: dict):
    """Return lists of languages that have a reference column in the CSV."""
    df = pd.read_csv(csv_path, nrows=1)
    columns = df.columns.tolist()
    with_refs, without_refs = [], []
    for lang_key, cfg in lang_configs.items():
        if cfg['column_suffix'] in columns:
            with_refs.append(lang_key)
        else:
            without_refs.append(lang_key)
    return with_refs, without_refs