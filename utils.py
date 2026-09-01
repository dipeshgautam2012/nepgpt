import torch
from dataset import BatchSampler, CharTokenizer
@torch.no_grad()
def estimate_loss(model, train_sampler: BatchSampler, val_sampler: BatchSampler, eval_iters: int = 200):
    """Computes average evaluation loss over train and val samplers without mutating state."""
    out = {}
    # set model to evaluation mode
    model.eval()

    samplers = {"train": train_sampler, "val": val_sampler}

    for split_name, sampler in samplers.items():
        losses = torch.zeros(eval_iters)

        actual_steps = 0
        for i, (x, y) in enumerate(sampler):
            if i >= eval_iters:
                break
            logits, loss = model(x, y)
            losses[i] = loss.item()
            actual_steps = i + 1

        out[split_name] = losses[:actual_steps].mean().item()

    # set model back to training mode
    model.train()
    return out