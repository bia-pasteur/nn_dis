"""
Loss functions for image registration.
"""

from typing import Literal, Tuple, Union

import torch
from torch import nn

import network.functional as nf


class _SpatialPenaltyBase(nn.Module):
    def __init__(
        self,
        penalty: Literal["l1", "l2"] = "l2",
        reduction: Union[str, None] = "mean",
        reduction_dim: Union[int, Tuple[int, ...], None] = None,
        keepdims: bool = False,
    ):
        super().__init__()
        self.penalty = penalty
        self.reduction = reduction
        self.reduction_dim = reduction_dim
        self.keepdims = keepdims

    @property
    def _penalty_kwargs(self) -> dict:
        return dict(
            penalty=self.penalty,
            reduction=self.reduction,
            reduction_dim=self.reduction_dim,
            keepdims=self.keepdims,
        )


class MSE(nn.Module):
    """Mean Squared Error, inspired from neurite.modules"""

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        reduction: Union[str, None] = "weighted_mean",
        weight_mask: Union[float, None] = 0.5,
        reduction_dim: Union[int, Tuple[int, ...], None] = None,
        keepdims: bool = False,
    ) -> torch.Tensor:
        return nf.mse(
            prediction=prediction,
            target=target,
            mask=mask,
            reduction=reduction,
            reduction_dim=reduction_dim,
            keepdims=keepdims,
            weight_mask=weight_mask,
        )


class SpatialGradient(_SpatialPenaltyBase):
    """Spatial gradient penalty, inspired from neurite.modules"""

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return nf.spatial_gradient_penalty(
            input_tensor=input_tensor, **self._penalty_kwargs
        )


class SpatialHessian(_SpatialPenaltyBase):
    """Spatial Hessian penalty, inspired from neurite.modules"""

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return nf.spatial_hessian_penalty(
            input_tensor=input_tensor, **self._penalty_kwargs
        )
