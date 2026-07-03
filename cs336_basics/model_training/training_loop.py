from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from dataclasses import asdict
from exp_log import ExperimentLogger


from cs336_basics.transformer.module import (
    MyLLM,
    MyAdamw,
    MyLLMWithoutPreNorm,
    my_cross_entropy,
    my_get_batch,
    my_gradient_clipping,
    my_lr_cosine_learning_schedule,
    my_save_checkpoint,
    my_load_checkpoint,
)


@dataclass
class TrainConfig:
    train_path: str
    val_path: str

    batch_size: int
    context_length: int

    device: str

    max_iters: int

    lr_max: float
    lr_min: float
    warmup_iters: int

    log_interval: int
    eval_interval: int
    eval_iters: int

    checkpoint_interval: int
    checkpoint_dir: str

    grad_clip: float = 1.0

    experiment_name: str = "baseline"
    log_dir: str = "runs"


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataset: np.ndarray,
    config: TrainConfig,
) -> float:
    model.eval()

    losses = []

    for _ in range(config.eval_iters):
        x, y = my_get_batch(
            dataset,
            config.batch_size,
            config.context_length,
            config.device,
        )

        logits = model(x)

        loss = my_cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )

        losses.append(loss.item())

    model.train()

    return sum(losses) / len(losses)


def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    logger: ExperimentLogger,
    resume: str | None = None,
):
    logger = ExperimentLogger(
        log_dir=config.log_dir,
        experiment_name=config.experiment_name,
        config=asdict(config),
    )
    model = model.to(config.device)

    train_dataset = np.memmap(
        config.train_path,
        dtype=np.uint16,
        mode="r",
    )

    val_dataset = np.memmap(
        config.val_path,
        dtype=np.uint16,
        mode="r",
    )

    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    iteration = 0
    best_val = float("inf")

    if resume is not None:
        iteration = my_load_checkpoint(
            resume,
            model,
            optimizer,
        )

        print(f"Resume training from iteration {iteration}")

    model.train()

    while iteration < config.max_iters:
        lr = my_lr_cosine_learning_schedule(
            iteration,
            config.lr_max,
            config.lr_min,
            config.warmup_iters,
            config.max_iters,
        )

        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = my_get_batch(
            train_dataset,
            config.batch_size,
            config.context_length,
            config.device,
        )

        logits = model(x)

        loss = my_cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if config.grad_clip is not None and config.grad_clip > 0:
            my_gradient_clipping(
                model.parameters(),
                config.grad_clip,
            )

        optimizer.step()

        if iteration % config.log_interval == 0:
            print(
                f"[{iteration}] "
                f"lr={lr:.3e} "
                f"train_loss={loss.item():.4f}"
            )
            logger.log_metric(
                step=iteration,
                split="train",
                loss=loss.item(),
                lr=lr,
            )

        if iteration % config.eval_interval == 0:
            val_loss = evaluate(
                model,
                val_dataset,
                config,
            )

            print(
                f"[{iteration}] "
                f"val_loss={val_loss:.4f}"
            )
            logger.log_metric(
                step=iteration,
                split="val",
                loss=val_loss,
                lr=lr,
            )

            if val_loss < best_val:
                best_val = val_loss

                my_save_checkpoint(
                    model,
                    optimizer,
                    iteration,
                    checkpoint_dir / "best.pt",
                )

                print(
                    f"[{iteration}] "
                    f"saved best checkpoint "
                    f"val_loss={val_loss:.4f}"
                )

        if iteration % config.checkpoint_interval == 0:
            my_save_checkpoint(
                model,
                optimizer,
                iteration,
                checkpoint_dir / "latest.pt",
            )

            print(
                f"[{iteration}] "
                f"saved latest checkpoint"
            )

        iteration += 1

def main():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    vocab_size = 10000
    d_model = 512
    context_length = 256
    num_layers = 10
    num_heads = 16
    d_ff = 1344
    rope_theta = 10000.0

    config = TrainConfig(
        train_path="data/tinystories/train.bin",
        val_path="data/tinystories/val.bin",

        batch_size=32,
        context_length=context_length,

        device=device,

        max_iters=10_000,

        lr_max=3e-4,
        lr_min=3e-5,
        warmup_iters=500,

        log_interval=10,
        eval_interval=200,
        eval_iters=20,

        checkpoint_interval=500,
        checkpoint_dir="checkpoints/baseline",

        grad_clip=1.0,

        experiment_name="baseline",
        log_dir="runs",
    )

    model = MyLLMWithoutPreNorm(
        vocab_size=vocab_size,
        d_model=d_model,
        context_length=context_length,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
    )

    for name, p in model.named_parameters():
        assert not torch.isnan(p).any(), f"{name} contains NaN"
        assert not torch.isinf(p).any(), f"{name} contains Inf"    

    optimizer = MyAdamw(
        model.parameters(),
        lr=1e-6,              # 由 cosine scheduler 每 step 設定
        weight_decay=0.01,
        eps=1e-8,
        betas=(0.9, 0.999),
    )

    logger = ExperimentLogger(
        log_dir=config.log_dir,
        experiment_name=config.experiment_name,
        config=asdict(config),
    )

    train(
        model=model,
        optimizer=optimizer,
        config=config,
        logger=logger,
        resume=None,
    )


if __name__ == "__main__":
    main()