import os
import torch
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm
from src.config import (LEARNING_RATE, MAX_ITERS, WARMUP_ITERS, MIN_LR,
                        GRADIENT_ACCUMULATION_STEPS, EVAL_INTERVAL,
                        device, dtype, ptdtype, device_type)
from src.model import GPT, GPTConfig
from src.dataset import get_batch
from src.utils import estimate_loss, plot_losses, save_checkpoint
from contextlib import nullcontext

def train():
    # Model
    config = GPTConfig()
    model = GPT(config).to(device)

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95), weight_decay=0.1, eps=1e-9)
    scheduler_warmup = LinearLR(optimizer, total_iters=WARMUP_ITERS)
    scheduler_decay = CosineAnnealingLR(optimizer, T_max=MAX_ITERS - WARMUP_ITERS, eta_min=MIN_LR)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_decay], milestones=[WARMUP_ITERS])

    # Mixed precision context
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16')) if device_type == 'cuda' else None

    # Training loop
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    os.makedirs("saved_models", exist_ok=True)

    for iter_num in tqdm(range(MAX_ITERS), desc="Training"):
        # Evaluate on train/val periodically
        if iter_num % EVAL_INTERVAL == 0 and iter_num != 0:
            losses = estimate_loss(model)
            train_losses.append(losses['train'])
            val_losses.append(losses['val'])
            print(f"\nStep {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, lr {optimizer.param_groups[0]['lr']:.6f}")

            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                save_checkpoint(model, optimizer, scheduler, iter_num, losses['val'], "saved_models/best_model.pt")
                print(f"  -> saved new best model (val loss {best_val_loss:.4f})")

        # Get batch and forward/backward
        X, Y = get_batch("train")
        with ctx:
            _, loss = model(X, Y)
            loss = loss / GRADIENT_ACCUMULATION_STEPS

        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (iter_num + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or (iter_num + 1) == MAX_ITERS:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        scheduler.step()

    # Final evaluation and plot
    final_losses = estimate_loss(model)
    print(f"\nFinal: train loss {final_losses['train']:.4f}, val loss {final_losses['val']:.4f}")
    plot_losses(train_losses, val_losses, save_path="saved_models/loss_plot.png")
    save_checkpoint(model, optimizer, scheduler, MAX_ITERS, final_losses['val'], "saved_models/final_model.pt")

if __name__ == "__main__":
    train()