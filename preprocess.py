import os
import numpy as np
from datasets import load_dataset
import tiktoken
from tqdm import tqdm

def process(example):
    enc = tiktoken.get_encoding("gpt2")
    ids = enc.encode_ordinary(example['text'])
    return {'ids': ids, 'len': len(ids)}

def main():
    ds = load_dataset("roneneldan/TinyStories")
    tokenized = ds.map(process, remove_columns=['text'], num_proc=8)

    os.makedirs("data/processed", exist_ok=True)
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = f"data/processed/{split}.bin"
        dtype = np.uint16
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024
        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()

if __name__ == "__main__":
    main()