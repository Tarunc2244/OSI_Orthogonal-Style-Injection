# config.py
"""
Central configuration for OSI experiments.
All hyperparameters and paths can be set here and overridden via command line.
"""

# Model
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
# MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # uncomment for Llama

# Steering hyperparameters
CONDITION_LAYER_IDX = 21          # optimal 
USE_MULTI_AXIS = True
N_STYLE_AXES = 19                   # number of PCs to remove (90% variance on Titanic)
USE_CONTRASTIVE = False            # use raw centroids (not contrastive)

# Filtering options
APPLY_FILTERING = True
TOP_PERCENT = 0.4
MIN_LINES_PER_CHARACTER = 10

# Generation settings
MAX_INPUT_LEN = 128
MAX_NEW_TOKENS = 100
BATCH_SIZE_EMBED = 16

FEWSHOT_NUM_SHOTS = 3

DATASETS = {
    "Avatar": "../data/processed/Avatar.csv",
    "Titanic": "../data/processed/Titanic.csv",
    "Spiderman2": "../data/processed/Spiderman2.csv",
}

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
