"""Common utilities."""

import argparse
import functools
import json
import logging
import time
from collections.abc import Callable
from typing import Any

import torch
import yaml
from pathlib import Path


def stopwatch(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Measure the execution time of any function"""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs) -> Callable[..., Any]:
        start_time = time.time()
        result = fn(*args, **kwargs)
        end_time = time.time()
        logging.info(
            "Function %s took %.3f seconds to execute",
            fn.__name__,
            end_time - start_time,
        )
        return result

    return wrapper


def save_argparse(args: argparse.Namespace, out_path: str) -> None:
    """Serializes the argparse.Namespace to a JSON file.

    Args:
        args: The parsed command-line arguments.
        out_path: The path to save the JSON file.
    """
    config_dict = vars(args)
    with open(out_path, "w") as f:
        json.dump(config_dict, f)


def get_device() -> torch.device:
    device_str = "cpu"
    if torch.cuda.is_available():
        device_str = "cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device_str = "mps"
    return torch.device(device_str)


class ModelArgs:
    def __init__(self):
        parser = argparse.ArgumentParser(description="Model and training arguments")

        # Arguments from run_bpe.py, run_model.py, and run_tok.py
        parser.add_argument("--vocab_size", type=int)
        parser.add_argument("--special_tokens", type=str, nargs="+")
        parser.add_argument("--profile", action="store_true")
        parser.add_argument("--dry_run", action="store_true")
        parser.add_argument("--vocab_path", type=str)
        parser.add_argument("--merges_path", type=str)
        parser.add_argument("--bpe_path", type=str)
        parser.add_argument("--txt_path", type=str)
        parser.add_argument("--load_checkpoint", action="store_true")
        parser.add_argument("--checkpoint_file", type=str)
        parser.add_argument("--training_steps", type=int)
        parser.add_argument("--batch_size", type=int)
        parser.add_argument("--num_batches", type=int)
        parser.add_argument("--context_length", type=int)
        parser.add_argument("--d_model", type=int)
        parser.add_argument("--num_layers", type=int)
        parser.add_argument("--num_heads", type=int)
        parser.add_argument("--d_ff", type=int)
        parser.add_argument("--rope_theta", type=float)
        parser.add_argument("--dtype", type=str)
        parser.add_argument("--max_lr", type=float)
        parser.add_argument("--min_lr", type=float)
        parser.add_argument("--warmup_iters", type=int)
        parser.add_argument("--cosine_cycle_iters", type=int)
        parser.add_argument("--weight_decay", type=float)
        parser.add_argument("--beta1", type=float)
        parser.add_argument("--beta2", type=float)
        parser.add_argument("--eps", type=float)
        parser.add_argument("--seed", type=int)
        parser.add_argument("--checkpoint_interval", type=int)

        # New argument for model file
        parser.add_argument(
            "--model-file", type=str, help="Path to a YAML file with model arguments."
        )

        self.parser = parser
        self.args = self.parser.parse_args()
        self.model_config = {}
        if self.args.model_file:
            model_file_path = Path(self.args.model_file)
            if model_file_path.is_file():
                with open(model_file_path, "r") as f:
                    self.model_config = yaml.safe_load(f)

        self._defaults = {
            "vocab_size": 500,
            "special_tokens": ["<|endoftext|>"],
            "profile": False,
            "dry_run": False,
            "load_checkpoint": False,
            "checkpoint_file": "run_model.checkpoint",
            "training_steps": 100,
            "batch_size": 10,
            "num_batches": 1,
            "context_length": 256,
            "d_model": 512,
            "num_layers": 4,
            "num_heads": 16,
            "d_ff": 1344,
            "rope_theta": 10000.0,
            "dtype": "float32",
            "max_lr": 1e-3,
            "min_lr": 1e-4,
            "warmup_iters": 100,
            "cosine_cycle_iters": 1000,
            "weight_decay": 0.01,
            "beta1": 0.9,
            "beta2": 0.999,
            "eps": 1e-8,
            "seed": 1337,
            "checkpoint_interval": 10,
        }

    def __getattr__(self, name):
        # 1. Command line arguments
        if hasattr(self.args, name) and getattr(self.args, name) is not None:
            return getattr(self.args, name)
        # 2. Model file configuration
        if name in self.model_config:
            return self.model_config[name]
        # 3. Default values
        if name in self._defaults:
            return self._defaults[name]

        raise ValueError(f"{name} was not specified.")
