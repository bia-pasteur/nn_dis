"""
Core model for learning the dispalcement field.
"""

import torch
import torch.nn.functional as F
from torch import nn


class SpatialTransformer(nn.Module):
    """Inspired by VoxelMorph"""

    def __init__(self, mode: str = "bilinear"):
        super().__init__()
        self.mode = mode

    def forward(
        self,
        moving: torch.Tensor,
        displacement: torch.Tensor,
    ) -> torch.Tensor:
        shape = moving.shape[2:]
        ndim = len(shape)

        axes = [
            torch.arange(0, s, dtype=displacement.dtype, device=displacement.device)
            for s in shape
        ]

        meshgrid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=0).contiguous()

        new_locs = meshgrid + displacement

        sizes = torch.tensor(
            shape, dtype=displacement.dtype, device=displacement.device
        )

        scales = 2.0 / (sizes - 1).clamp(min=1)
        broadcast_shape = (ndim,) + (1,) * len(shape)
        scales = scales.view(broadcast_shape)

        grid = new_locs * scales - 1.0

        grid = grid.flip(
            dims=(1,)
        )  # Convert from ij (z,y,x) to (x,y,z) convention for grid_sample
        grid = grid.permute(0, 2, 3, 4, 1)  # To have a (N, D, H, W, C) shape

        return F.grid_sample(
            moving, grid, mode=self.mode, padding_mode="zeros", align_corners=True
        )


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                padding_mode="reflect",
            ),
            nn.BatchNorm3d(out_channels),
            nn.GELU(),
            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                padding_mode="reflect",
            ),
            nn.BatchNorm3d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DownsampleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool3d(
            kernel_size=(2, 2, 2),
            stride=(2, 2, 2),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.conv(x)
        pooled = self.pool(skip)
        return skip, pooled


class UpsampleConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ):
        super().__init__()
        self.conv = ConvBlock(
            in_channels=in_channels + skip_channels,
            out_channels=out_channels,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            x,
            size=skip.shape[2:],
            mode="trilinear",
            align_corners=True,
        )
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ConvBlockAnisotropic(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, is_2d: bool = False):
        super().__init__()
        kernel = (1, 3, 3) if is_2d else 3
        padding = (0, 1, 1) if is_2d else 1
        self.conv = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=kernel,
                padding=padding,
                padding_mode="reflect",
            ),
            nn.BatchNorm3d(out_channels),
            nn.GELU(),
            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=kernel,
                padding=padding,
                padding_mode="reflect",
            ),
            nn.BatchNorm3d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DownsampleConvAnisotropic(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, is_2d: bool = False):
        super().__init__()
        self.conv = ConvBlockAnisotropic(in_channels, out_channels, is_2d=is_2d)
        scale = (1, 2, 2) if is_2d else (2, 2, 2)
        self.pool = nn.MaxPool3d(kernel_size=scale, stride=scale)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.conv(x)
        pooled = self.pool(skip)
        return skip, pooled


class UpsampleConvAnisotropic(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        is_2d: bool = False,
    ):
        super().__init__()

        self.conv = ConvBlockAnisotropic(
            in_channels=in_channels + skip_channels,
            out_channels=out_channels,
            is_2d=is_2d,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            x,
            size=skip.shape[2:],
            mode="trilinear",
            align_corners=True,
        )

        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class NetworkPairwise(nn.Module):
    def __init__(
        self,
        ndim: int = 3,
        ref_channels: int = 1,
        moving_channels: int = 1,
        features: list[int] = [16, 32, 32],
        flow_multiplier: float = 10,
        device: str = "cpu",
    ):
        super().__init__()

        self.ndim = ndim
        self.device = device
        self.flow_multiplier = flow_multiplier

        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        in_ch = ref_channels + moving_channels

        for out_ch in features:
            self.encoder.append(ConvBlock(in_ch, out_ch))
            in_ch = out_ch

        self.bottleneck = ConvBlock(
            in_channels=in_ch,
            out_channels=in_ch,
        )

        self.decoder = nn.ModuleList()

        for level in reversed(range(len(features))):
            skip_ch = features[level]
            out_ch = features[level]

            self.decoder.append(
                UpsampleConv(
                    in_channels=in_ch,
                    skip_channels=skip_ch,
                    out_channels=out_ch,
                )
            )
            in_ch = out_ch

        self.flow_head = nn.Conv3d(
            in_channels=in_ch,
            out_channels=ndim,
            kernel_size=1,
        )

        nn.init.normal_(self.flow_head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.flow_head.bias)

        self.transformer = SpatialTransformer()

    def forward(
        self,
        ref: torch.Tensor,
        moving: torch.Tensor,
        return_warped_moving: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:

        assert ref.ndim == 5 and moving.ndim == 5, (
            f"Wrong shapes ! Got {ref.shape=}, {moving.shape=}"
        )

        x = torch.cat([ref, moving], dim=1)

        skips = []
        for enc_block in self.encoder:
            skip = enc_block(x)
            skips.append(skip)
            x = self.pool(skip)

        x = self.bottleneck(x)

        for dec_block, skip in zip(self.decoder, reversed(skips)):
            x = dec_block(x, skip)

        field = self.flow_head(x) * self.flow_multiplier

        if return_warped_moving:
            return field, self.transformer(moving, field)

        return field


class NetworkPairwiseAnisotropic(nn.Module):
    """
    Same as NetworkPairwise but for anisotropic volumes.
    The first n_2d_levels encoder/decoder levels only operate in the (H, W) plane.
    """

    def __init__(
        self,
        ndim: int = 3,
        ref_channels: int = 1,
        moving_channels: int = 1,
        features: list[int] = [16, 32, 32, 64, 64],
        n_2d_levels: int = 2,
        flow_multiplier: float = 10,
        device: str = "cpu",
    ):
        super().__init__()

        self.ndim = ndim
        self.device = device
        self.flow_multiplier = flow_multiplier

        self.encoder = nn.ModuleList()
        in_ch = ref_channels + moving_channels

        for i, out_ch in enumerate(features):
            self.encoder.append(
                DownsampleConvAnisotropic(in_ch, out_ch, is_2d=(i < n_2d_levels))
            )
            in_ch = out_ch

        self.bottleneck = ConvBlockAnisotropic(
            in_channels=in_ch, out_channels=in_ch, is_2d=(i < n_2d_levels)
        )

        self.decoder = nn.ModuleList()

        for level in reversed(range(len(features))):
            skip_ch = features[level]
            out_ch = features[level]

            self.decoder.append(
                UpsampleConvAnisotropic(
                    in_channels=in_ch,
                    skip_channels=skip_ch,
                    out_channels=out_ch,
                    is_2d=(level < n_2d_levels),
                )
            )
            in_ch = out_ch

        self.flow_head = nn.Conv3d(
            in_channels=in_ch,
            out_channels=ndim,
            kernel_size=1,
        )

        nn.init.normal_(self.flow_head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.flow_head.bias)

        self.transformer = SpatialTransformer()

    def forward(
        self,
        ref: torch.Tensor,
        moving: torch.Tensor,
        return_warped_moving: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:

        assert ref.ndim == 5 and moving.ndim == 5, (
            f"Wrong shapes ! Got {ref.shape=}, {moving.shape=}"
        )

        x = torch.cat([ref, moving], dim=1)

        skips = []
        for enc_block in self.encoder:
            skip, x = enc_block(x)
            skips.append(skip)

        x = self.bottleneck(x)

        for dec_block, skip in zip(self.decoder, list(reversed(skips))):
            x = dec_block(x, skip)

        field = self.flow_head(x) * self.flow_multiplier

        if return_warped_moving:
            return field, self.transformer(moving, field)

        return field
