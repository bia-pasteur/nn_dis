"""
Loss functions for image registration.
"""

import torch
from torch import nn

import network.functional as nf


class _SpatialPenaltyBase(nn.Module):
    def __init__(
        self,
        penalty: str = "l2",
        reduction: str | None = "mean",
        reduction_dim: int | tuple[int] | None = None,
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
        reduction: str | None = "weighted_mean",
        weight_mask: float | None = 0.5,
        reduction_dim: int | tuple[int] | None = None,
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
        pred: torch.Tensor,
        target: torch.Tensor,
        spacing: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:

        assert pred.ndim == 5, (
            f"pred must be of shape (B, 3, Z, H, W). Got {pred.shape}."
        )

        assert mask.ndim == 4, f"mask must be of shape (B, Z, H, W). Got {mask.shape}."

        nb_mask = torch.sum(mask)
        assert nb_mask != 0, "Got an all zero mask"

        mask = mask.to(device=pred.device, dtype=pred.dtype)

        error_z = (pred[:, 0] - target[:, 0]) * spacing[:, 0].view(-1, 1, 1, 1)
        error_y = (pred[:, 1] - target[:, 1]) * spacing[:, 1].view(-1, 1, 1, 1)
        error_x = (pred[:, 2] - target[:, 2]) * spacing[:, 2].view(-1, 1, 1, 1)

        squared_error = error_z.square() + error_y.square() + error_x.square()

        return (squared_error * mask).sum() / (3 * mask.sum() + 1e-8)


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
