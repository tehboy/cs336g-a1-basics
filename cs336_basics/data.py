import os
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from torch import Tensor
import numpy as np

from itertools import count
from numpy.random import choice


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str | torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    if isinstance(device, str):
        device = torch.device(device)
    # Random starting points
    starting_points = np.random.choice(len(dataset) - context_length, batch_size, replace=False)
    # Build indices for all batch rows
    indices = starting_points[:, None] + np.arange(context_length)[None, :]
    inputs_np = dataset[indices]
    next_tokens_np = dataset[indices + 1]
    inputs = torch.as_tensor(inputs_np, dtype=torch.long, device=device)
    next_tokens = torch.as_tensor(next_tokens_np, dtype=torch.long, device=device)

    return inputs, next_tokens


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iteration": iteration},
        out,
    )


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
):
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer, optional): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    saved_state = torch.load(src)
    model.load_state_dict(saved_state["model"])
    if optimizer is not None:
        optimizer.load_state_dict(saved_state["optimizer"])
    return saved_state["iteration"]
