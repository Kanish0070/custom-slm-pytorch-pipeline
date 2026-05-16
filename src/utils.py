import torch
import matplotlib.pyplot as plt
from src.dataset import get_batch
from src.config import EVAL_ITERS, device

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    with torch.inference_mode():
        for split in ['train', 'val']:
            losses = torch.zeros(EVAL_ITERS)
            for k in range(EVAL_ITERS):
                X, Y = get_batch(split)
                _, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
    model.train()
    return out

def plot_losses(train_losses, val_losses, save_path=None):
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='validation')
    plt.xlabel("Evaluation step (every EVAL_INTERVAL iterations)")
    plt.ylabel("Loss")
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def save_checkpoint(model, optimizer, scheduler, epoch, loss, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'loss': loss,
    }, path)

def load_checkpoint(model, optimizer, scheduler, path):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']