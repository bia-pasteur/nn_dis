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


class GT_comp(nn.Module):
    """Distance to ground truth value"""

    def forward(
        self,
        computed_dis: torch.Tensor,
        gt_dis: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:

        assert computed_dis.ndim == 5, (
            f"computed_dis must be of shape (B, 3, Z, H, W). Got {computed_dis.shape}."
        )

        assert mask.ndim == 4, f"mask must be of shape (B, Z, H, W). Got {mask.shape}."
        nb_mask = torch.sum(mask)
        assert nb_mask != 0, "Got an all zero mask"

        return torch.sum(((computed_dis - gt_dis) ** 2) * mask[:, None]) / nb_mask


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
