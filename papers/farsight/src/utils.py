import os
import random
import logging
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


def set_seed(seed: int):
    """
    Set seed for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_str: str = "cuda") -> torch.device:
    """
    Return a torch.device based on a device string and availability.
    """
    if device_str.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_str)
    return torch.device("cpu")


def setup_logging(log_dir: str, filename: str = "train.log", level: int = logging.INFO):
    """
    Configure root logger to output to both console and a file in `log_dir`.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, filename)
    # Clear existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging setup. Logs will be saved to {log_path}")


def save_checkpoint(state: dict, checkpoint_dir: str, filename: str = "checkpoint.pt"):
    """
    Save training state (model, optimizer, scheduler, epoch) to file.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    logging.info(f"Saved checkpoint to {filepath}")


def load_checkpoint(filepath: str, model: torch.nn.Module, optimizer=None, scheduler=None, map_location=None):
    """
    Load training state from a checkpoint file into model and optionally optimizer/scheduler.
    Returns the epoch number if present in the checkpoint.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"No checkpoint found at {filepath}")
    checkpoint = torch.load(filepath, map_location=map_location)
    # Load model state
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    # Load optimizer state if provided
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    # Load scheduler state if provided
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    epoch = checkpoint.get("epoch", None)
    logging.info(f"Loaded checkpoint from {filepath} (epoch {epoch})")
    return epoch


def get_summary_writer(log_dir: str, sub_dir: str = None) -> SummaryWriter:
    """
    Returns a TensorBoard SummaryWriter instance for logging metrics.
    If sub_dir is provided, create it under log_dir.
    """
    if sub_dir:
        log_dir = os.path.join(log_dir, sub_dir)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    logging.info(f"TensorBoard writer created at {log_dir}")
    return writer
