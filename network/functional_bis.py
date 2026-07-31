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


def _forward_diff(
    input_tensor: torch.Tensor,
    axis: int,
    spacing: float = 1.0,
) -> torch.Tensor:
    """
    Compute forward finite difference along a given axis, preserving input shape.

    For element i: diff[i] = (f[i+1] - f[i]) / spacing
    The last element along the axis is set to zero (zero-padded boundary).

    Args:
        input_tensor: Input tensor
        axis: Axis along which to compute the difference
        spacing: Voxel size along the axis (default: 1.0)

    Returns:
        Tensor of same shape as input containing forward differences
    """
    # Create output tensor same shape as input, initialized to zero
    diff = torch.zeros_like(input_tensor)

    # Build slices for indexing
    slc_start = [slice(None)] * input_tensor.ndim
    slc_end = [slice(None)] * input_tensor.ndim

    slc_start[axis] = slice(0, -1)
    slc_end[axis] = slice(1, None)

    # Forward difference for interior points
    diff[tuple(slc_start)] = (
        input_tensor[tuple(slc_end)] - input_tensor[tuple(slc_start)]
    ) / spacing

    # Last element remains zero (already set)
    return diff


def _backward_diff(
    input_tensor: torch.Tensor,
    axis: int,
    spacing: float = 1.0,
) -> torch.Tensor:
    """
    Compute backward finite difference along a given axis, preserving input shape.

    For element i: diff[i] = (f[i] - f[i-1]) / spacing
    Special handling at boundaries (adjoint-consistent):
        diff[0] = f[0] / spacing
        diff[-1] = -f[-2] / spacing

    Args:
        input_tensor: Input tensor
        axis: Axis along which to compute the difference
        spacing: Voxel size along the axis (default: 1.0)

    Returns:
        Tensor of same shape as input containing backward differences
    """
    # Create output tensor same shape as input, initialized to zero
    diff = torch.zeros_like(input_tensor)

    # Build slices for indexing
    slc_prev = [slice(None)] * input_tensor.ndim
    slc_curr = [slice(None)] * input_tensor.ndim

    slc_prev[axis] = slice(0, -1)
    slc_curr[axis] = slice(1, None)

    # Main backward difference: f[i] - f[i-1] for interior points
    diff[tuple(slc_curr)] = (
        input_tensor[tuple(slc_curr)] - input_tensor[tuple(slc_prev)]
    ) / spacing

    # Boundary: diff[0] = f[0] / spacing
    start_idx = [slice(None)] * input_tensor.ndim
    start_idx[axis] = 0
    diff[tuple(start_idx)] = input_tensor[tuple(start_idx)] / spacing

    # Boundary: diff[-1] = -f[-2] / spacing
    end_idx = [slice(None)] * input_tensor.ndim
    end_idx[axis] = -1
    prev_idx = [slice(None)] * input_tensor.ndim
    prev_idx[axis] = -2
    diff[tuple(end_idx)] = -input_tensor[tuple(prev_idx)] / spacing

    return diff


def spatial_gradient(
    input_tensor: torch.Tensor,
    non_spatial_dims: Union[Tuple[int, ...], None] = None,
    spacing: Union[Tuple[float, ...], None] = None,
) -> List[torch.Tensor]:
    """
    Compute first-order spatial derivatives using forward differences.

    Output tensors have the same shape as input (zero-padded boundaries).

    Args:
        input_tensor: Input tensor shaped (non_spatial..., *spatial_dims)
        non_spatial_dims: Number or tuple of non-spatial dimensions
                         (e.g., (0, 1) for (B, C, H, W, D))
        spacing: Voxel size along each spatial axis. If None, assumes 1.0

    Returns:
        List of gradient tensors, one per spatial dimension, same shape as input
    """
    if non_spatial_dims is None:
        non_spatial_dims = ()
    elif isinstance(non_spatial_dims, int):
        non_spatial_dims = (non_spatial_dims,)

    num_non_spatial = len(non_spatial_dims)
    num_spatial = input_tensor.ndim - num_non_spatial

    if num_spatial < 1:
        raise ValueError(f"Need at least 1 spatial dim, got {num_spatial}")

    if spacing is None:
        spacing = (1.0,) * num_spatial
    elif len(spacing) != num_spatial:
        raise ValueError(f"spacing length {len(spacing)} != num_spatial {num_spatial}")

    return [
        _forward_diff(input_tensor, axis=num_non_spatial + i, spacing=spacing[i])
        for i in range(num_spatial)
    ]


def spatial_gradient_penalty(
    input_tensor: torch.Tensor,
    penalty: Literal["l1", "l2"] = "l2",
    reduction: Union[str, None] = "mean",
    non_spatial_dims: Union[Tuple[int, ...], None] = (0, 1),
    reduction_dim: Union[int, Tuple[int, ...], None] = None,
    keepdims: bool = False,
    spacing: Union[Tuple[float, ...], None] = None,
) -> torch.Tensor:
    """
    Compute spatial gradient penalty for tensors shaped (B, C, *spatial_dims).

    Args:
        input_tensor: Input tensor
        penalty: 'l1' for |∇u|, 'l2' for |∇u|²
        reduction: Reduction to apply ('mean', 'sum', etc.)
        non_spatial_dims: Dimensions that are not spatial (default: (0, 1) for batch and channel)
        reduction_dim: Dimension(s) to reduce over
        keepdims: Whether to keep reduced dimensions
        spacing: Voxel size along each spatial axis

    Returns:
        Scalar or reduced tensor representing the gradient penalty
    """
    if penalty not in ["l1", "l2"]:
        raise ValueError(f"penalty must be 'l1' or 'l2', got '{penalty}'")

    # Compute spatial gradients (list of tensors, one per spatial dim)
    grads = spatial_gradient(
        input_tensor, non_spatial_dims=non_spatial_dims, spacing=spacing
    )

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
    spacing: Union[Tuple[float, ...], None] = None,
) -> List[List[torch.Tensor]]:
    """
    Compute second-order spatial derivatives (Hessian matrix entries) using
    forward finite differences twice.

    H[i,j] = ∂²u / ∂x_i ∂x_j  (forward diff of forward diff)

    Output tensors have the same shape as input (zero-padded boundaries).

    Args:
        input_tensor: Input tensor shaped (non_spatial..., *spatial_dims)
        non_spatial_dims: Number or tuple of non-spatial dimensions
        spacing: Voxel size along each spatial axis. If None, assumes 1.0

    Returns:
        List of lists: hessians[i][j] = ∂²/∂x_i∂x_j, same shape as input
    """
    if non_spatial_dims is None:
        non_spatial_dims = ()
    elif isinstance(non_spatial_dims, int):
        non_spatial_dims = (non_spatial_dims,)

    num_non_spatial = len(non_spatial_dims)
    num_spatial = input_tensor.ndim - num_non_spatial

    if num_spatial < 1:
        raise ValueError(f"Need at least 1 spatial dim, got {num_spatial}")

    if spacing is None:
        spacing = (1.0,) * num_spatial
    elif len(spacing) != num_spatial:
        raise ValueError(f"spacing length {len(spacing)} != num_spatial {num_spatial}")

    # First derivatives (forward differences)
    first_diffs = [
        _forward_diff(input_tensor, axis=num_non_spatial + i, spacing=spacing[i])
        for i in range(num_spatial)
    ]

    # Second derivatives: apply forward diff again to each first derivative
    hessians = []
    for i in range(num_spatial):
        row = []
        for j in range(num_spatial):
            second = _forward_diff(
                first_diffs[i], axis=num_non_spatial + j, spacing=spacing[j]
            )
            row.append(second)
        hessians.append(row)

    return hessians


def spatial_hessian_penalty(
    input_tensor: torch.Tensor,
    penalty: Literal["l1", "l2"] = "l2",
    reduction: Union[str, None] = "mean",
    non_spatial_dims: Union[Tuple[int, ...], None] = (0, 1),
    reduction_dim: Union[int, Tuple[int, ...], None] = None,
    keepdims: bool = False,
    spacing: Union[Tuple[float, ...], None] = None,
) -> torch.Tensor:
    """
    Compute spatial Hessian penalty for tensors shaped (B, C, *spatial_dims).

    Args:
        input_tensor: Input tensor
        penalty: 'l1' for |H|, 'l2' for |H|²
        reduction: Reduction to apply
        non_spatial_dims: Non-spatial dimensions (default: (0, 1))
        reduction_dim: Dimensions to reduce over
        keepdims: Keep reduced dimensions
        spacing: Voxel size along each spatial axis

    Returns:
        Scalar or reduced tensor representing the Hessian penalty
    """
    if penalty not in ["l1", "l2"]:
        raise ValueError(f"penalty must be 'l1' or 'l2', got '{penalty}'")

    hessians = spatial_hessian(
        input_tensor, non_spatial_dims=non_spatial_dims, spacing=spacing
    )

    # Flatten all Hessian entries
    all_terms = [term for row in hessians for term in row]

    # Apply penalty
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
