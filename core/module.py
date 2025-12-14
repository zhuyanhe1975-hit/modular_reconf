from dataclasses import dataclass
import numpy as np


@dataclass
class Module:
    id: int
    q: np.ndarray  # shape (2,)
    world_T: np.ndarray  # shape (4, 4)
