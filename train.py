import argparse
import torch
from model import GPTModel, ModelConfig
from dataset import BatchSampler, CharTokenizer
from utils import estimate_loss

input_file = "munamadan.txt"
def parse_args():
    p = argparse.ArgumentParser(description="Train a small GPT on a text file.")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    p.add_argument("--input", default= input_file)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--block-size", type=int, default=32)
    p.add_argument("--num-embed", type=int, default=64)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--max-iters", type=int, default=5000)
    p.add_argument("--eval-interval", type=int, default=100)
    p.add_argument("--eval-iters", type=int, default=200)
    p.add_argument("--max-new-tokens", type=int, default=2000)
    p.add_argument("--with-replacement", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device
    torch.manual_seed(args.seed)

    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = CharTokenizer(text)
    vocab_size = tokenizer.num_tokens()
    print(f"Vocabulary size: {vocab_size}")

    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    config = ModelConfig(
        vocab_size=vocab_size,
        block_size=args.block_size,
        num_embed=args.num_embed,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    model = GPTModel(config).to(device)
    print(f"Model parameters: {model.count_parameters()}")
    print(f"Device: {device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_sampler = BatchSampler(
        train_data,
        batch_size=args.batch_size,
        block_size=config.block_size,
        device=device,
        shuffle=args.shuffle,
        with_replacement=args.with_replacement,
    )
    val_sampler = BatchSampler(
        val_data,
        batch_size=args.batch_size,
        block_size=config.block_size,
        device=device,
        shuffle=args.shuffle,
        with_replacement=args.with_replacement,
    )
    if args.with_replacement:
        eval_train_sampler = train_sampler
    else:
        eval_train_sampler = BatchSampler(
            train_data,
            batch_size=args.batch_size,
            block_size=config.block_size,
            device=device,
            shuffle=False,
            with_replacement=False,
        )

    train_iter = iter(train_sampler)
    for global_step in range(args.max_iters):
        if global_step % args.eval_interval == 0 or global_step == args.max_iters - 1:
            losses = estimate_loss(model, eval_train_sampler, val_sampler, eval_iters=args.eval_iters)
            print(f"Step {global_step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        item = next(train_iter, None)
        if item is None:
            train_iter = iter(train_sampler)
            x, y = next(train_iter)
        else:
            x, y = item

        logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(tokenizer.decode(model.generate(context, max_new_tokens=args.max_new_tokens)[0].tolist()))


if __name__ == "__main__":
    main()
