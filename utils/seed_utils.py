import random

import numpy as np
import torch


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch for more reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def build_torch_generator(seed: int) -> torch.Generator:
    """Create a torch Generator with a fixed seed for DataLoader shuffling."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def seed_worker(worker_id: int) -> None:
    """Seed each DataLoader worker deterministically from PyTorch's worker seed."""
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
