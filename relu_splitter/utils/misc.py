import uuid
from pathlib import Path
from random import uniform

import torch
import onnx
import numpy as np
from onnx2pytorch import ConvertModel
from onnxruntime import InferenceSession

from .logger import logger


def get_random_id(len=8):
    assert len <= 32
    return str(uuid.uuid4())[:len]

def adjust_mask_random_k(mask, k):
    assert k >= 0
    count = torch.sum(mask).item()
    if count > k:
        indices = torch.nonzero(mask, as_tuple=True)[0]
        selected_indices = indices[torch.randperm(len(indices))[:k]]
        mask = torch.zeros_like(mask, dtype=torch.bool)
        mask[selected_indices] = True
        return mask
    elif count <= k:
        return mask
    raise ValueError("Something wrong")

def adjust_mask_first_k(mask, k):
    assert k >= 0
    count = torch.sum(mask).item()
    if count > k:
        indices = torch.nonzero(mask, as_tuple=True)[0]
        selected_indices = indices[:k]  # Select the first k indices in order
        mask = torch.zeros_like(mask, dtype=torch.bool)
        mask[selected_indices] = True
        return mask
    elif count <= k:
        return mask
    raise ValueError("Something wrong")

def find_feasible_point(lb, ub, w, b, step_rate=1e-2, max_iters=1000):
    """
    Finds a point x within the bounds [lb, ub] that satisfies w @ x + b > 0
    using a gradient-like approach with a dynamically computed step size.

    Parameters:
    lb (torch.Tensor): Lower bound for x.
    ub (torch.Tensor): Upper bound for x.
    w (torch.Tensor): Weight tensor.
    b (torch.Tensor): Bias tensor.
    initial_step_size (float): Initial step size for iteratively adjusting x.
    max_iters (int): Maximum number of iterations to try.

    Returns:
    torch.Tensor: A feasible point x that satisfies w @ x + b > 0.
    """
    max_value = torch.where(w > 0, ub, lb) @ w + b
    min_value = torch.where(w > 0, lb, ub) @ w + b
    # assert max_value > 0, f"No feasible point within the given bounds can satisfy w @ x + b > 0, {max_value}"

    x = lb.clone()
    for i in range(max_iters):
        result = w @ x + b
        if result >= max_value/2:
            assert (x >= lb).all() and (x <= ub).all(), "Returned x is out of bounds!"
            assert result > 0, "Returned x does not satisfy w @ x + b > 0!"
            logger.debug(f"Found feasible point after {i} iterations.")
            return x
        gradient_direction = w.sign()
        x += (ub - lb) * gradient_direction * step_rate 
        x = torch.where(x < lb, lb + ((ub-lb)*step_rate*2), x)
        x = torch.where(x > ub, ub - ((ub-lb)*step_rate*2), x)
    logger.warning(f"No feasible point found after {max_iters} iterations, returning the last point.")
    return x