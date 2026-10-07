"""
Loss functions for image registration.
"""

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
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


class StrainGTComp(nn.Module):
    """
    MSE loss between predicted and target strain tensors.

    This reproduces the behavior of:

        strain_mask(u, spacing, maskcell)

    for tensors of shape:

        pred, target: (B, 3, Z, H, W)
        spacing:      (B, 3), ordered (z, y, x)
        mask:         (B, Z, H, W)

    The mask boundary is handled as in grad_domain(...,
    boundary_condition="Dirichlet"):
        - strain derivatives are zero at boundary voxels;
        - derivatives are zero outside the mask.
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def _dirichlet_boundary_mask(mask: torch.Tensor) -> torch.Tensor:
        """
        Reproduce:

            boundary_mask = distance_transform_edt(mask) == 1

        independently for each batch element.

        Returns:
            Boolean tensor of shape (B, Z, H, W).
        """
        mask_np = mask.detach().cpu().numpy()

        boundary_np = np.stack(
            [distance_transform_edt(mask_np[b]) == 1 for b in range(mask_np.shape[0])],
            axis=0,
        )

        return torch.from_numpy(boundary_np).to(mask.device)

    @staticmethod
    def _gradient_along_axis(
        f: torch.Tensor,
        spacing: torch.Tensor,
        axis: int,
    ) -> torch.Tensor:
        """
        Differentiable equivalent of np.gradient(f, spacing, axis=axis)
        with NumPy's default edge_order=1.

        Parameters
        ----------
        f:
            Tensor of shape (B, Z, H, W).

        spacing:
            Tensor of shape (B,), one spatial spacing value per batch item.

        axis:
            Spatial tensor dimension:
                1 -> Z
                2 -> H / Y
                3 -> W / X
        """
        if f.shape[axis] < 2:
            raise ValueError(
                f"Cannot compute gradient along axis {axis}: "
                f"dimension has size {f.shape[axis]}."
            )

        # Move requested spatial axis to the last dimension.
        # Example for axis=1:
        # (B, Z, H, W) -> (B, H, W, Z)
        x = f.movedim(axis, -1)

        # Shape appropriate for broadcasting over all non-batch dimensions.
        # For x=(B, ..., N), h becomes (B, 1, 1, ...)
        h = spacing.reshape(
            spacing.shape[0],
            *([1] * (x.ndim - 2)),
        )

        # Equivalent to np.gradient(..., edge_order=1)
        first = (x[..., 1] - x[..., 0]) / h
        last = (x[..., -1] - x[..., -2]) / h

        if x.shape[-1] > 2:
            middle = (x[..., 2:] - x[..., :-2]) / (2.0 * h.unsqueeze(-1))

            gradient = torch.cat(
                [
                    first.unsqueeze(-1),
                    middle,
                    last.unsqueeze(-1),
                ],
                dim=-1,
            )
        else:
            # For length 2, np.gradient gives the same first-order
            # derivative at both positions.
            gradient = torch.stack([first, last], dim=-1)

        # Move the differentiated dimension back to its original location.
        return gradient.movedim(-1, axis)

    def _strain(
        self,
        u: torch.Tensor,
        spacing: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Differentiable tensor equivalent of strain_mask(...).

        Parameters
        ----------
        u:
            Displacement field, shape (B, 3, Z, H, W).

        spacing:
            Shape (B, 3), ordered (spacing_z, spacing_y, spacing_x).

        mask:
            Shape (B, Z, H, W).

        Returns
        -------
        strain:
            Shape (B, 3, 3, Z, H, W), where:

                strain[:, i, j] = 0.5 * (
                    du_i / dx_j + du_j / dx_i
                )
        """
        batch_size, dim, z, h, w = u.shape

        if dim != 3:
            raise ValueError(f"Expected 3 displacement components, got {dim}.")

        mask = mask.to(device=u.device, dtype=u.dtype)

        # Equivalent to:
        #
        # boundary_mask = distance_transform_edt(mask) == 1
        #
        # in grad_domain.
        boundary_mask = self._dirichlet_boundary_mask(mask).to(u.device)

        # In your NumPy code:
        #
        # gradient[boundary_mask] = 0
        # return gradient * mask
        #
        # Therefore the valid derivative support is inside the mask
        # but excluding Dirichlet boundary voxels.
        valid_gradient_mask = mask * (~boundary_mask).to(u.dtype)

        derivatives = []

        # j = 0: derivative along Z, tensor axis 2
        # j = 1: derivative along Y, tensor axis 3
        # j = 2: derivative along X, tensor axis 4
        for j, tensor_axis in enumerate((2, 3, 4)):
            # Computes du_i / dx_j for every displacement component i.
            #
            # u[:, i] has shape (B, Z, H, W)
            # result has shape (B, 3, Z, H, W)
            du_dxj = self._gradient_along_axis(
                u,
                spacing[:, j],
                axis=tensor_axis,
            )

            # Same as grad_domain(...)*mask, with Dirichlet boundary.
            du_dxj = du_dxj * valid_gradient_mask[:, None]

            derivatives.append(du_dxj)

        # derivatives[j] contains du_i / dx_j
        #
        # Stack to obtain:
        # jac[b, i, j, z, y, x] = d u_i / d x_j
        #
        # Shape: (B, 3, 3, Z, H, W)
        jac = torch.stack(derivatives, dim=2)

        # Equivalent to:
        #
        # 0.5 * (jac + np.swapaxes(jac, 0, 1))
        #
        # except dimensions 1 and 2 are the displacement/derivative
        # component dimensions in the batched tensor representation.
        strain = 0.5 * (jac + jac.transpose(1, 2))

        return strain

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        spacing: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        pred, target:
            Shape (B, 3, Z, H, W)

        spacing:
            Shape (B, 3), with columns ordered as (z, y, x).

        mask:
            Shape (B, Z, H, W).
        """
        assert pred.ndim == 5, (
            f"pred must be of shape (B, 3, Z, H, W). Got {pred.shape}."
        )
        assert target.shape == pred.shape, (
            f"target must have same shape as pred. "
            f"Got target={target.shape}, pred={pred.shape}."
        )
        assert mask.ndim == 4, f"mask must be of shape (B, Z, H, W). Got {mask.shape}."
        assert spacing.ndim == 2 and spacing.shape[1] == 3, (
            f"spacing must be of shape (B, 3). Got {spacing.shape}."
        )

        if mask.sum() == 0:
            raise ValueError("Got an all-zero mask.")

        spacing = spacing.to(device=pred.device, dtype=pred.dtype)
        mask = mask.to(device=pred.device, dtype=pred.dtype)

        pred_strain = self._strain(pred, spacing, mask)
        target_strain = self._strain(target, spacing, mask)

        # Shape: (B, 3, 3, Z, H, W)
        squared_error = (pred_strain - target_strain).square()

        # Apply the original cell mask to every strain component.
        strain_mask = mask[:, None, None, :, :, :]

        # Nine entries of the 3x3 strain tensor.
        return (squared_error * strain_mask).sum() / (9.0 * mask.sum() + 1e-8)


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
