"""
Tensor operations and functions
"""

from typing import List, Literal, Tuple, Union

import torch

_MULTIDIM_REDUCTIONS = frozenset(
    {"mean", "sum", "median", "amax", "amin", "std", "var", "var_mean", "weighted_mean"}
)
_SINGLEDIM_REDUCTIONS = frozenset({"argmin", "argmax"})
_ALL_REDUCTIONS = _MULTIDIM_REDUCTIONS | _SINGLEDIM_REDUCTIONS


def reduce(
    tensor: torch.Tensor,
    reduction: str = "weighted_mean",
    dim: Union[int, Tuple[int, ...], None] = None,
    keepdims: bool = False,
    mask: torch.tensor = None,
    weight_mask: Union[float, None] = 0.5,
) -> torch.Tensor:
    """
    Apply any torch reduction on a tensor,
    inspired by neurite.functional
    """
    if reduction is None:
        return tensor

    if reduction not in _ALL_REDUCTIONS:
        raise ValueError(
            f"Invalid reduction '{reduction}'. Valid options: {sorted(_ALL_REDUCTIONS)}"
        )

    if reduction in _SINGLEDIM_REDUCTIONS and not isinstance(dim, int):
        raise ValueError(
            f"Reduction '{reduction}' requires dim to be an int, got {type(dim)}"
        )

    if reduction == "mean" and mask is not None:
        mask_bool = mask.to(tensor.device).bool()

        mask_bool = mask_bool.expand_as(tensor)

        masked_values = tensor[mask_bool]

        if masked_values.numel() == 0:
            return torch.tensor(0.0, device=tensor.device)

        return masked_values.mean()

    if reduction == "weighted_mean":
        assert mask is not None
        mask_bool = mask.to(tensor.device).bool()
        mask_bool = mask_bool.expand_as(tensor)
        weight_map = torch.zeros_like(tensor)
        weight_map[mask_bool] = weight_mask / torch.sum(mask_bool)
        weight_map[~mask_bool] = (1 - weight_mask) / (torch.sum(~mask_bool))

        return torch.sum(tensor * weight_map)

    elif mask is not None:
        raise NotImplementedError

    return getattr(torch, reduction)(tensor, dim=dim, keepdim=keepdims)


def mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    reduction: Union[str, None] = "weighted_mean",
    reduction_dim: Union[int, Tuple[int, ...], None] = None,
    keepdims: bool = False,
    weight_mask: Union[float, None] = 0.5,
) -> torch.Tensor:
    """
    Compute mean squared error
    """
    return reduce(
        tensor=((prediction - target) ** 2),
        reduction=reduction,
        dim=reduction_dim,
        keepdims=keepdims,
        mask=mask,
        weight_mask=weight_mask,
    )


def spatial_gradient(
    input_tensor: torch.Tensor,
    non_spatial_dims: Union[Tuple[int, ...], None] = None,
) -> List[torch.Tensor]:
    """
    Compute first-order spatial derivatives,
    inspired by neurite.functional
    """

    num_non_spatial = len(non_spatial_dims) if non_spatial_dims is not None else 0
    num_spatial = input_tensor.ndim - num_non_spatial

    if num_spatial < 1:
        raise ValueError(f"Need at least 1 spatial dim, got {num_spatial}")

    return [
        torch.diff(input_tensor, dim=num_non_spatial + i) for i in range(num_spatial)
    ]


def spatial_gradient_penalty(
    input_tensor: torch.Tensor,
    penalty: Literal["l1", "l2"] = "l2",
    reduction: Union[str, None] = "mean",
    non_spatial_dims: Union[Tuple[int, ...], None] = (0, 1),
    reduction_dim: Union[int, Tuple[int, ...], None] = None,
    keepdims: bool = False,
) -> torch.Tensor:
    """
    Compute spatial gradient penalty for tensors shaped (B, C, *spatial_dims),
    inspired by neurite.functional
    """
    if penalty not in ["l1", "l2"]:
        raise ValueError(f"penalty must be 'l1' or 'l2', got '{penalty}'")

    # Compute spatial gradients (list of tensors, one per spatial dim)
    grads = spatial_gradient(input_tensor, non_spatial_dims=non_spatial_dims)

    # Apply penalty
    if penalty == "l1":
        penalties = [g.abs() for g in grads]
    else:  # l2
        penalties = [g * g for g in grads]

    if reduction is None:
        return penalties

    # Reduce each penalty tensor and average across spatial dimensions
    reduced = []
    for p in penalties:
        r = reduce(tensor=p, reduction=reduction, dim=reduction_dim, keepdims=keepdims)
        reduced.append(r)

    # Average across spatial dimensions
    return sum(reduced) / len(reduced)


def spatial_hessian(
    input_tensor: torch.Tensor,
    non_spatial_dims: Union[Tuple[int, ...], None] = None,
) -> List[List[torch.Tensor]]:
    """
    Compute second‑order spatial derivatives (Hessian matrix entries) using finite
    differences along each spatial dimension
    """
    num_non_spatial = len(non_spatial_dims) if non_spatial_dims is not None else 0
    num_spatial = input_tensor.ndim - num_non_spatial

    if num_spatial < 1:
        raise ValueError(f"Need at least 1 spatial dim, got {num_spatial}")

    first_diffs = [
        torch.diff(input_tensor, dim=num_non_spatial + i) for i in range(num_spatial)
    ]

    hessians = []
    for i in range(num_spatial):
        row = []
        for j in range(num_spatial):
            sec = torch.diff(first_diffs[i], dim=num_non_spatial + j)
            row.append(sec)
        hessians.append(row)
    return hessians


def spatial_hessian_penalty(
    input_tensor: torch.Tensor,
    penalty: Literal["l1", "l2"] = "l2",
    reduction: Union[str, None] = "mean",
    non_spatial_dims: Union[Tuple[int, ...], None] = (0, 1),
    reduction_dim: Union[int, Tuple[int, ...], None] = None,
    keepdims: bool = False,
) -> torch.Tensor:
    """
    Compute spatial Hessian penalty for tensors shaped (B, C, *spatial_dims),
    same pattern as spatial_gradient inspired by neurite.functional
    """

    if penalty not in ["l1", "l2"]:
        raise ValueError(f"penalty must be 'l1' or 'l2', got '{penalty}'")

    hessians = spatial_hessian(input_tensor, non_spatial_dims=non_spatial_dims)

    all_terms = [term for row in hessians for term in row]

    if penalty == "l1":
        penalties = [torch.abs(h) for h in all_terms]
    else:
        penalties = [h * h for h in all_terms]

    if reduction is None:
        return penalties

    reduced = [
        reduce(tensor=p, reduction=reduction, dim=reduction_dim, keepdims=keepdims)
        for p in penalties
    ]

    return sum(reduced) / len(reduced)
