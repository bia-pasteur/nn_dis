"""
Core model for unsupervised learning of the displacement
"""

import torch
import torch.nn.functional as F
from torch import nn


class SpatialTransformer(nn.Module):
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
        stride: int = 2,
        padding: int = 1,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                padding_mode="reflect",
            ),
            # nn.BatchNorm3d(out_channels),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DownsampleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvBlock(
            in_channels, out_channels, stride=2
        )  # stride = 2 pour le downsampling : à changer ? format (D, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UpsampleConv(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.Upsample(
            scale_factor=2.0, mode="trilinear", align_corners=True
        )  # scale_factor can be size = moving size

        self.conv = ConvBlock(
            in_channels=in_channels + skip_channels, out_channels=out_channels, stride=1
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
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
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=kernel,
                padding=padding,
                padding_mode="reflect",
            ),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.LeakyReLU(0.2, inplace=True),
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
        # scale = (1, 2, 2) if is_2d else 2.0
        # self.upsample = nn.Upsample(
        #     scale_factor=scale, mode="trilinear", align_corners=True
        # )
        # self.conv = ConvBlockAnisotropic(
        #     in_channels + skip_channels, out_channels, is_2d=is_2d
        # )

        self.conv = ConvBlockAnisotropic(
            in_channels=in_channels + skip_channels,
            out_channels=out_channels,
            is_2d=is_2d,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # x = self.upsample(x)
        # x = torch.cat([x, skip], dim=1)
        # return self.conv(x)

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
        self._upsample_mode = "trilinear"
        self.flow_multiplier = flow_multiplier

        in_channels = ref_channels + moving_channels

        self.encoder = nn.ModuleList()

        self.skip_channels = []
        in_ch = in_channels
        for feature in features:
            self.skip_channels.append(in_ch)
            self.encoder.append(DownsampleConv(in_ch, feature))
            in_ch = feature

        self.bottleneck = ConvBlock(features[-1], features[-1], stride=1)

        self.decoder = nn.ModuleList()
        reversed_features = list(reversed(features))
        reversed_skips = list(reversed(self.skip_channels))

        in_ch = features[-1]

        for i in range(len(reversed_features)):
            out_ch = reversed_features[i]
            skip_ch = reversed_skips[i]

            self.decoder.append(
                UpsampleConv(
                    in_channels=in_ch, skip_channels=skip_ch, out_channels=out_ch
                )
            )
            in_ch = out_ch

        self.flow_head = nn.Conv3d(in_ch, ndim, kernel_size=1, padding_mode="reflect")

        nn.init.normal_(self.flow_head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.flow_head.bias)

        self.transformer = SpatialTransformer()

    def forward(
        self,
        ref: torch.Tensor,
        moving: torch.Tensor,
        return_warped_moving: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:

        x = torch.cat([ref, moving], dim=1)

        skips = []
        for enc_block in self.encoder:
            skips.append(x)
            x = enc_block(x)

        x = self.bottleneck(x)

        skips = list(reversed(skips))
        for i, dec_block in enumerate(self.decoder):
            x = dec_block(x, skips[i])

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
        # self.skip_channels = []

        for i, out_ch in enumerate(features):
            # self.skip_channels.append(in_ch)
            self.encoder.append(
                DownsampleConvAnisotropic(in_ch, out_ch, is_2d=(i < n_2d_levels))
            )
            in_ch = out_ch

        self.bottleneck = ConvBlockAnisotropic(
            in_channels=in_ch, out_channels=in_ch, is_2d=(i < n_2d_levels)
        )

        self.decoder = nn.ModuleList()

        # reversed_features = list(reversed(features))
        # reversed_skips = list(reversed(self.skip_channels))

        # in_ch = features[-1]

        # for i in range(len(reversed_features)):
        #     encoder_level = len(features) - 1 - i
        #     out_ch = reversed_features[i]
        #     skip_ch = reversed_skips[i]
        #     self.decoder.append(
        #         UpsampleConvAnisotropic(
        #             in_channels=in_ch,
        #             skip_channels=skip_ch,
        #             out_channels=out_ch,
        #             is_2d=(encoder_level < n_2d_levels),
        #         )
        #     )
        #     in_ch = out_ch

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
