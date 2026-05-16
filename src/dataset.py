import torch
import numpy as np
from src.config import DATA_DIR, TRAIN_BIN, VAL_BIN, BLOCK_SIZE, BATCH_SIZE, device, device_type

def get_batch(split: str):
    """Returns (x, y) tensors of shape (B, T)"""
    if split == 'train':
        data = np.memmap(f"{DATA_DIR}/{TRAIN_BIN}", dtype=np.uint16, mode='r')
    else:
        data = np.memmap(f"{DATA_DIR}/{VAL_BIN}", dtype=np.uint16, mode='r')

    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([torch.from_numpy((data[i:i+BLOCK_SIZE]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+BLOCK_SIZE]).astype(np.int64)) for i in ix])

    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y