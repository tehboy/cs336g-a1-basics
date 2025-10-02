import argparse
import logging
import numpy as np
import random
import sys
import time
import torch
import wandb

from cs336_basics.basics import TransformerLanguageModel
from cs336_basics.training import AdamW, get_lr_cosine_schedule, cross_entropy, gradient_clipping
from cs336_basics.data import get_batch, save_checkpoint, load_checkpoint
from cs336_basics.utils import get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Run model training script.")
    parser.add_argument("--train-file", type=str, required=True, help="Path to the training file.")
    parser.add_argument("--vocab-size", type=int, required=True, help="Vocabulary size.")
    parser.add_argument(
        "--load-checkpoint",
        action="store_false",
        help="Attempt to load from the checkpoint file on startup.",
        default=False,
    )
    parser.add_argument(
        "--checkpoint-file",
        type=str,
        default="run_model.checkpoint",
        help="Path to save/load checkpoint file.",
    )
    parser.add_argument("--training-steps", type=int, default=100, help="Number of training steps.")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size.")
    parser.add_argument("--num-batches", type=int, default=1, help="Number of batches.")
    parser.add_argument("--context-length", type=int, default=256, help="Context length.")
    parser.add_argument("--d-model", type=int, default=512, help="Model dimension.")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of transformer layers.")
    parser.add_argument("--num-heads", type=int, default=16, help="Number of attention heads.")
    parser.add_argument("--d-ff", type=int, default=1344, help="Feedforward dimension.")
    parser.add_argument("--rope-theta", type=float, default=10000.0, help="RoPE theta parameter.")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        help="Torch dtype (e.g., float32, float16, bfloat16).",
    )
    parser.add_argument(
        "--max-lr", type=float, default=1e-3, help="Maximum learning rate for cosine schedule."
    )
    parser.add_argument(
        "--min-lr", type=float, default=1e-4, help="Minimum learning rate for cosine schedule."
    )
    parser.add_argument(
        "--warmup-iters", type=int, default=100, help="Number of warmup iterations."
    )
    parser.add_argument(
        "--cosine-cycle-iters",
        type=int,
        default=1000,
        help="Number of iterations for a cosine cycle.",
    )
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay.")
    parser.add_argument("--beta1", type=float, default=0.9, help="AdamW beta1.")
    parser.add_argument("--beta2", type=float, default=0.999, help="AdamW beta2.")
    parser.add_argument("--eps", type=float, default=1e-8, help="AdamW epsilon.")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed (for reproducibility)")
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
        help="Interval (in steps) to save checkpoints.",
    )
    parser.add_argument(
        "--profile", action="store_true", help="Profile the training process using cProfile."
    )
    return parser.parse_args()


def setup_logging():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout
    )


def main():
    start_time = time.time()
    args = parse_args()
    setup_logging()
    logging.info(f"Training file path: {args.train_file}")

    # Parse device and dtype
    device = get_device()
    dtype = getattr(torch, args.dtype)

    # Torch
    torch.manual_seed(args.seed)
    # NumPy
    np.random.seed(args.seed)
    # Python
    random.seed(args.seed)

    # Initialize wandb
    run = wandb.init(
        entity="natjambo",
        project="cs336",
        config={
            "train_file": args.train_file,
            "vocab_size": args.vocab_size,
            "training_steps": args.training_steps,
            "batch_size": args.batch_size,
            "num_batches": args.num_batches,
            "context_length": args.context_length,
            "d_model": args.d_model,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "d_ff": args.d_ff,
            "rope_theta": args.rope_theta,
            "dtype": args.dtype,
            "max_lr": args.max_lr,
            "min_lr": args.min_lr,
            "warmup_iters": args.warmup_iters,
            "cosine_cycle_iters": args.cosine_cycle_iters,
            "weight_decay": args.weight_decay,
            "beta1": args.beta1,
            "beta2": args.beta2,
            "eps": args.eps,
            "seed": args.seed,
        },
    )
    # Initialize TransformerLanguageModel
    model = TransformerLanguageModel(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device,
        dtype=dtype,
    )
    model.train()
    model.to(device)
    logging.info(
        f"Initialized TransformerLanguageModel with vocab_size={args.vocab_size}, context_length={args.context_length}, d_model={args.d_model}, num_layers={args.num_layers}, num_heads={args.num_heads}, d_ff={args.d_ff}, rope_theta={args.rope_theta}, device={device}, dtype={dtype}"
    )

    # Initialize AdamW optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.max_lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
    )
    logging.info(
        f"Initialized AdamW optimizer with weight_decay={args.weight_decay}, betas=({args.beta1}, {args.beta2}), eps={args.eps}"
    )

    if args.load_checkpoint:
        logging.info(f"Loading checkpoint from {args.checkpoint_file}")
        t = load_checkpoint(args.checkpoint_file, model, optimizer)
    else:
        t = 1

    train_file = np.load(args.train_file, mmap_mode="r")
    logging.info(
        f"Memmapped training file: {args.train_file} with shape {train_file.shape} and dtype {train_file.dtype}"
    )

    logging.info("Beginning training run.")
    global_step=0
    while t <= args.training_steps:
        iter_start_time = time.time()

        # Update learning rate
        lr = get_lr_cosine_schedule(
            t, args.max_lr, args.min_lr, args.warmup_iters, args.cosine_cycle_iters
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Get batch
        batch_losses = []
        batch_times = []
        for batch_n in range(args.batch_size):
            batch_start_time = time.time()
            input, target = get_batch(train_file, args.batch_size, args.context_length, device)
            batch_end_time = time.time()
            optimizer.zero_grad()

            # Forward pass
            forward_start_time = time.time()
            output = model(input)
            forward_end_time = time.time()
            # Compute loss
            loss_start_time = time.time()
            loss = cross_entropy(output, target)
            loss_end_time = time.time()
            # Backward pass
            backward_start_time = time.time()
            loss.backward()
            backward_end_time = time.time()
            # Gradient clipping
            g = gradient_clipping(model.parameters(), 1.0)
            # Update weights
            optimizer_step_start_time = time.time()
            optimizer.step()
            optimizer_step_end_time = time.time()

            batch_loss = loss.item()
            batch_losses.append(batch_loss)
            batch_times.append(batch_end_time - batch_start_time)
            run.log({"batch_loss": batch_loss, "lr": lr, "grad_norm": g}, step=global_step)
            if t % 10 == 0 or t == 1:
                logging.info(f"Step {t} Batch {batch_n}: loss={loss.item():.4f}, lr={lr:.6f}")
                logging.info(f"  get_batch time: {batch_end_time - batch_start_time:.4f}s")
                logging.info(f"  Forward pass time: {forward_end_time - forward_start_time:.4f}s")
                logging.info(f"  Loss calculation time: {loss_end_time - loss_start_time:.4f}s")
                logging.info(
                    f"  Backward pass time: {backward_end_time - backward_start_time:.4f}s"
                )
                logging.info(
                    f"  Optimizer step time: {optimizer_step_end_time - optimizer_step_start_time:.4f}s"
                )
            global_step += 1

        # Use average loss and time for logging
        running_loss = torch.tensor(batch_losses).mean()
        batch_time = sum(batch_times) / len(batch_times)

        run.log(
            {
                "epoch": t,
                "avg_epoch_loss": running_loss.item(),
                "avg_batch_time": batch_time,
            }, step=t
        )
        iter_end_time = time.time()

        # Logging
        if t % args.checkpoint_interval == 1:
            checkpoint_start = time.time()
            save_checkpoint(model, optimizer, t, args.checkpoint_file)
            checkpoint_elapsed = time.time() - checkpoint_start
            logging.info(f"Checkpointing at step {t} (took {checkpoint_elapsed:.2f} seconds)")
        if t % 10 == 0 or t == 1:
            logging.info(f"  Total iteration time: {iter_end_time - iter_start_time:.4f}s")
            logging.info(f"  Running loss: {running_loss:.4f}")
            logging.info(f"  Average batch time: {batch_time:.4f}s")
        t += 1
    # Final save
    save_checkpoint(model, optimizer, t, args.checkpoint_file)
    run.finish()
    elapsed = time.time() - start_time
    logging.info(f"Elapsed time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Total runtime: {end_time - start_time:.2f} seconds")
