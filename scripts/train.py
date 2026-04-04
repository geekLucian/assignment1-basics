from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import wandb

from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.adamw import AdamW
from cs336_basics.utils import (
    get_batch,
    cross_entropy,
    gradient_clipping,
    save_checkpoint,
    get_lr_cosine_schedule,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--valid-data", type=Path, required=True)
    parser.add_argument("--dataset-dtype", type=str, default="uint16")

    # Model
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--d-model", type=int, required=True)
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument("--num-heads", type=int, required=True)
    parser.add_argument("--d-ff", type=int, required=True)
    parser.add_argument("--rope-theta", type=float, default=10000.0)

    # Optimizer / scheduler
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--max-lr", type=float, required=True)
    parser.add_argument("--min-lr", type=float, required=True)
    parser.add_argument("--warmup-iters", type=int, required=True)
    parser.add_argument("--cosine-cycle-iters", type=int, required=True)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    # Training loop
    parser.add_argument("--total-iters", type=int, required=True)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-iters", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=1000)

    # Checkpoint / device
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--wandb-project", type=str, default="assignment1-basics")
    parser.add_argument("--wandb-run-name", type=str, default=None)

    return parser.parse_args()



@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    eval_iters: int,
    device: str,
) -> float:
    model.eval()

    losses = []
    for _ in range(eval_iters):
        x, y = get_batch(
            dataset=dataset,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
        )
        logits = model(x)
        loss = cross_entropy(logits, y)
        losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses)


def main() -> None:
    args = parse_args()

    # 1. Load datasets with np.memmap
    train_data = np.memmap(args.train_data, dtype=args.dataset_dtype, mode="r")
    valid_data = np.memmap(args.valid_data, dtype=args.dataset_dtype, mode="r")

    # 2. Build model
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=args.device,
    )

    # 3. Build optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.max_lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
    )

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=vars(args),
    )

    model.train()

    for it in range(args.total_iters):
        # 4. Update LR from scheduler
        lr = get_lr_cosine_schedule(
            it=it,
            max_learning_rate=args.max_lr,
            min_learning_rate=args.min_lr,
            warmup_iters=args.warmup_iters,
            cosine_cycle_iters=args.cosine_cycle_iters,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        # 5. Sample training batch
        x, y = get_batch(
            dataset=train_data,
            batch_size=args.batch_size,
            context_length=args.context_length,
            device=args.device,
        )

        # 6. Forward pass
        logits = model(x)
        loss = cross_entropy(logits, y)

        # 7. Backward pass
        optimizer.zero_grad()
        loss.backward()

        # 8. Gradient clipping
        gradient_clipping(model.parameters(), args.max_grad_norm)

        # 9. Optimizer step
        optimizer.step()

        # 10. Periodic evaluation / logging
        if it % args.eval_every == 0:
            val_loss = evaluate(
                model=model,
                dataset=valid_data,
                batch_size=args.batch_size,
                context_length=args.context_length,
                eval_iters=args.eval_iters,
                device=args.device,
            )
            print(
                f"iter={it} "
                f"train_loss={loss.item():.6f} "
                f"val_loss={val_loss:.6f} "
                f"lr={lr:.6e}"
            )
            wandb.log(
                {
                    "train_loss": loss.item(),
                    "val_loss": val_loss,
                    "lr": lr,
                },
                step=it,
            )

        # 11. Periodic checkpoint save
        if it % args.save_every == 0 and it > 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                iteration=it,
                out=args.checkpoint_path,
            )

    # 12. Optional final checkpoint
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        iteration=it,
        out=args.checkpoint_path,
    )

    wandb.finish()


if __name__ == "__main__":
    main()
