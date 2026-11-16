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
        return {
            "penalty": self.penalty,
            "reduction": self.reduction,
            "reduction_dim": self.reduction_dim,
            "keepdims": self.keepdims,
        }


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


class StrainComp(nn.Module):
    """Distance between the strain tensors of pred and target"""

    @staticmethod
    def _boundary_ring(mask: torch.Tensor) -> torch.Tensor:
        """
        First ring of pixels inside the mask, i.e. the pixels where
        distance_transform_edt(mask) == 1. For a binary mask, these are
        exactly the in-mask pixels with at least one out-of-mask neighbor.

        Args:
            mask (torch.Tensor): Binary mask of shape (B, Z, H, W).

        Returns:
            torch.Tensor: The boundary ring, same shape as mask.
        """
        m = mask.unsqueeze(1)  # (B, 1, Z, H, W)
        # Pad with 1 (in-mask): scipy's distance_transform_edt only measures
        # distance to zeros inside the array, so out-of-volume neighbors must
        # not count as background.
        padded = torch.nn.functional.pad(m, (1, 1, 1, 1, 1, 1), value=1.0)
        shifts = (
            padded[:, :, :-2, 1:-1, 1:-1],
            padded[:, :, 2:, 1:-1, 1:-1],
            padded[:, :, 1:-1, :-2, 1:-1],
            padded[:, :, 1:-1, 2:, 1:-1],
            padded[:, :, 1:-1, 1:-1, :-2],
            padded[:, :, 1:-1, 1:-1, 2:],
        )
        min_neighbor = shifts[0]
        for s in shifts[1:]:
            min_neighbor = torch.minimum(min_neighbor, s)
        # A pixel is on the ring if it is in the mask and has at least one
        # out-of-mask neighbor.
        return (m * (min_neighbor < m)).squeeze(1)

    @staticmethod
    def _jacobian(
        u: torch.Tensor,
        mask: torch.Tensor,
        spacing: torch.Tensor,
    ) -> torch.Tensor:
        """
        Jacobian of the displacement field, computed in torch so that
        gradients flow back to u. Central differences in the interior,
        one-sided at the volume edges (edge_order=1), matching np.gradient,
        with a Dirichlet boundary condition: the derivative is zeroed at the
        boundary ring of the mask and outside of it (same as
        experiments.src.quantities_computation.jacobian_mask).

        Args:
            u (torch.Tensor): Displacement field, shape (B, 3, Z, H, W),
                channels in (z, y, x) order.
            mask (torch.Tensor): Mask of shape (B, Z, H, W).
            spacing (torch.Tensor): Per-batch spacing, shape (B, 3),
                order (z, y, x).

        Returns:
            torch.Tensor: The Jacobian, shape (B, 3, 3, Z, H, W),
                with jac[:, i, j] = du_i / dx_j.
        """
        mask = mask.unsqueeze(1)  # (B, 1, Z, H, W)
        boundary = StrainComp._boundary_ring(mask.squeeze(1)).unsqueeze(1)
        grads = []
        for j, dim in enumerate((2, 3, 4)):
            dx = spacing[:, j].view(-1, 1, 1, 1, 1)
            grad = torch.gradient(u, spacing=1.0, dim=dim)[0] / dx
            # Dirichlet boundary condition, as in grad_domain
            grads.append(grad * (mask - boundary))
        # grads[j] holds du_i / dx_j for every component i: stack along a
        # new axis, then swap so that jac[:, i, j] = du_i / dx_j.
        return torch.stack(grads, dim=1).transpose(1, 2)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:

        assert pred.ndim == 5, (
            f"pred must be of shape (B, 3, Z, H, W). Got {pred.shape}."
        )

        assert mask.ndim == 4, f"mask must be of shape (B, Z, H, W). Got {mask.shape}."

        nb_mask = torch.sum(mask)
        assert nb_mask != 0, "Got an all zero mask"

        mask = mask.to(device=pred.device, dtype=pred.dtype)

        spacing = torch.ones(pred.shape[0], 3, device=pred.device, dtype=pred.dtype)

        jac_pred = self._jacobian(pred, mask, spacing)
        jac_target = self._jacobian(target, mask, spacing)
        strain_pred = 0.5 * (jac_pred + jac_pred.transpose(1, 2))
        strain_target = 0.5 * (jac_target + jac_target.transpose(1, 2))

        squared_error = (strain_pred - strain_target).square().sum(dim=(1, 2))

        return (squared_error * mask).sum() / (9 * mask.sum() + 1e-8)


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
