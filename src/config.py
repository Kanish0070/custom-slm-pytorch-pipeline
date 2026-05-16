import yaml
import torch

with open("config.yaml", "r") as f:
    _cfg = yaml.safe_load(f)

# Data paths
DATA_DIR = _cfg["data_dir"]
TRAIN_BIN = _cfg["train_bin"]
VAL_BIN = _cfg["val_bin"]

# Model
VOCAB_SIZE = _cfg["vocab_size"]
BLOCK_SIZE = _cfg["block_size"]
N_LAYER = _cfg["n_layer"]
N_HEAD = _cfg["n_head"]
N_EMBD = _cfg["n_embd"]
DROPOUT = _cfg["dropout"]
BIAS = _cfg["bias"]

# Training
BATCH_SIZE = _cfg["batch_size"]
GRADIENT_ACCUMULATION_STEPS = _cfg["gradient_accumulation_steps"]
LEARNING_RATE = _cfg["learning_rate"]
MAX_ITERS = _cfg["max_iters"]
WARMUP_ITERS = _cfg["warmup_iters"]
MIN_LR = _cfg["min_lr"]
EVAL_ITERS = _cfg["eval_iters"]
EVAL_INTERVAL = _cfg["eval_interval"]

# Generation
GENERATE_MAX_TOKENS = _cfg["generate_max_tokens"]
TEMPERATURE = _cfg["temperature"]
TOP_K = _cfg["top_k"]

# Device / dtype
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_type = "cuda" if device.type == "cuda" else "cpu"
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]