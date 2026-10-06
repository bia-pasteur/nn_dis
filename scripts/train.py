#!/usr/bin/env python3
"""
Training of the network
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from pathlib import Path

import jsonargparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

import network
from config import TrainConfig
from scripts.dataset import (
    TractionDataset,
    get_intensity_transforms,
    get_spatial_transforms,
)


def update_log(log_dict: dict, update: dict) -> None:
    """
    Updating the log dictionaries

    Args:
        log_dict (dict): Current log dictionary
        update (dict): New logs to add
    """
    for k, v in update.items():
        if k in log_dict:
            log_dict[k].append(v)
        else:
            log_dict[k] = [v]


def epoch_(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    image_loss_fn: nn.Module,
    grad_loss_fn: nn.Module,
    hessian_loss_fn: nn.Module,
    gt_loss_fn: nn.Module,
    loss_weights: list[float],
    device: str = "cuda",
    training: bool = True,
    weight_mask: float | None = 0.5,
    reduce_log: bool = False,
    reduction: str = "mean",
) -> float:
    """
    Train for one epoch.

    Parameters
    ----------
    model : nn.Module
        The VoxelMorph model to train.
    dataloader : torch.utils.data.DataLoader
        The dataloader to use for training.
    optimizer : torch.optim.Optimizer
        The optimizer to use for training.
    image_loss_fn : nn.Module
        The image loss function to use.
    grad_loss_fn : nn.Module
        The gradient loss function to use.
    loss_weights : Sequence[float]
        The weights for the image and gradient losses.
    steps_per_epoch : int
    training : bool
        True if training, False if evaluation
    """

    log = {
        "index": [],
        "loss": [],
        "loss_mse": [],
        "loss_grad": [],
        "loss_hess": [],
        "loss_gt": [],
        "max_dis": [],
        "avg_dis": [],
    }

    if training:
        model.train()

    else:
        model.eval()

    for idx, batch in enumerate(iter(dataloader)):
        if training:
            optimizer.zero_grad()

        # Move to device in training loop (not dataloader/dataset!)
        ref = batch["ref"].to(device)
        moving = batch["moving"].to(device)
        mask = batch["mask"].to(device)
        u_gt = batch["u_gt"].to(device)
        mask_gt = batch["mask_gt"].to(device)
        spacing = batch["spacing"].to(device)

        # Get the displacement and the warped ref image from the model
        displacement, warped_moving = model(ref, moving, return_warped_moving=True)

        img_loss = image_loss_fn(
            ref,
            warped_moving,
            mask,
            reduction=reduction,
            weight_mask=weight_mask,
        )

        grad_loss = grad_loss_fn(displacement)
        hessian_loss = hessian_loss_fn(displacement)

        gt_loss = gt_loss_fn(
            pred=displacement, target=u_gt, mask=mask_gt, spacing=spacing
        )

        loss = (
            loss_weights[0] * img_loss
            + loss_weights[1] * grad_loss
            + loss_weights[2] * hessian_loss
            + loss_weights[3] * gt_loss
        )

        if training:
            loss.backward()
            optimizer.step()

        log["index"].append(idx)
        log["loss"].append(loss.item())
        log["loss_mse"].append(img_loss.item())
        log["loss_grad"].append(grad_loss.item())
        log["loss_hess"].append(hessian_loss.item())
        log["loss_gt"].append(gt_loss.item())
        log["max_dis"].append(np.max(np.abs(displacement.detach().cpu().numpy())))
        log["avg_dis"].append(np.mean(displacement.detach().cpu().numpy()))

    return (
        {k: np.mean(v) for k, v in log.items() if k != "index"}
        if not reduce_log
        else log
    )


def main(train_config: TrainConfig):
    """
    Main entry for training

    Args:
        train_config (TrainConfig): _description_
    """
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create model
    if train_config.model_type == "iso":
        model = network.models.NetworkPairwise(
            ndim=3,
            ref_channels=1,
            moving_channels=1,
            features=train_config.features,
            flow_multiplier=train_config.flow_multiplier,
        ).to(device)

    elif train_config.model_type == "aniso":
        model = network.models.NetworkPairwiseAnisotropic(
            ndim=3,
            ref_channels=1,
            moving_channels=1,
            features=train_config.features,
            flow_multiplier=train_config.flow_multiplier,
        ).to(device)

    else:
        raise ValueError(
            f"The type can only be iso or aniso. Got {train_config.model_type}"
        )

    # Setup losses and optimizer
    image_loss_fn = network.losses.MSE()
    grad_loss_fn = network.losses.SpatialGradient("l2")
    hessian_loss_fn = network.losses.SpatialHessian("l2")
    gt_loss_fn = network.losses.GT_comp()

    loss_weights = [
        train_config.warp_param,
        train_config.alpha_param,
        train_config.beta_param,
        train_config.gt_param,
    ]
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.lr)

    # Create dataloader
    if train_config.combine:
        micro_train_dataset = TractionDataset(
            Path(train_config.train_data),
            spatial_transforms=get_spatial_transforms(),
            intensity_transforms=get_intensity_transforms(),
            channel=train_config.channel,
            mask=train_config.mask,
        )
        synthetic_train_dataset = TractionDataset(
            Path("/pasteur/helix/projects/bia/jlahmani/data/synthetic/train"),
            spatial_transforms=get_spatial_transforms(),
            intensity_transforms=get_intensity_transforms(),
            channel=None,
            mask=False,
        )

        train_dataset = ConcatDataset([micro_train_dataset, synthetic_train_dataset])

        train_loader = DataLoader(
            train_dataset,
            batch_size=train_config.batch_size,
            num_workers=train_config.workers,
            shuffle=True,
        )

        micro_val_dataset = TractionDataset(
            Path(train_config.val_data),
            channel=train_config.channel,
            mask=train_config.mask,
        )
        synthetic_val_dataset = TractionDataset(
            Path("/pasteur/helix/projects/bia/jlahmani/data/synthetic/val"),
            channel=None,
            mask=False,
        )

        val_dataset = ConcatDataset([micro_val_dataset, synthetic_val_dataset])

        validation_loader = DataLoader(
            val_dataset,
            batch_size=train_config.batch_size,
            num_workers=train_config.workers,
            shuffle=True,
        )

    else:
        train_dataset = TractionDataset(
            Path(train_config.train_data),
            spatial_transforms=None,  # get_spatial_transforms(),
            intensity_transforms=None,  # get_intensity_transforms(),
            channel=train_config.channel,
            mask=train_config.mask,
            default_spacing=(4.0, 1.0, 1.0),
            path_gt=Path(train_config.path_gt),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_config.batch_size,
            num_workers=train_config.workers,
            shuffle=True,
        )

        validation_dataset = TractionDataset(
            Path(train_config.val_data),
            channel=train_config.channel,
            mask=train_config.mask,
            path_gt=Path(train_config.path_gt),
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=train_config.batch_size,
            num_workers=train_config.workers,
        )

    # Create output directory
    output_dir = Path(train_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    # print(f"Training for {train_config.epochs} epochs...")

    logs_train, logs_validation = {}, {}
    best_loss = float("inf")
    for epoch in tqdm(range(train_config.epochs), desc="Epochs"):
        # Train for one epoch
        epoch_log_train = epoch_(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            image_loss_fn=image_loss_fn,
            grad_loss_fn=grad_loss_fn,
            hessian_loss_fn=hessian_loss_fn,
            gt_loss_fn=gt_loss_fn,
            loss_weights=loss_weights,
            device=device,
            training=True,
            reduction=train_config.reduction,
        )

        with torch.no_grad():
            epoch_log_validation = epoch_(
                model=model,
                dataloader=validation_loader,
                optimizer=optimizer,
                image_loss_fn=image_loss_fn,
                grad_loss_fn=grad_loss_fn,
                hessian_loss_fn=hessian_loss_fn,
                gt_loss_fn=gt_loss_fn,
                loss_weights=loss_weights,
                device=device,
                training=False,
                reduction=train_config.reduction,
            )

        update_log(logs_train, epoch_log_train)
        update_log(logs_validation, epoch_log_validation)

        pd.DataFrame.from_dict(logs_train).to_csv(output_dir / "logs_train.csv")
        pd.DataFrame.from_dict(logs_validation).to_csv(
            output_dir / "logs_validation.csv"
        )

        avg_loss = epoch_log_validation["loss"]

        # Save periodic checkpoints
        if (epoch + 1) % train_config.save_every == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch{epoch + 1}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = output_dir / "best.pt"
            torch.save(model.state_dict(), best_path)

    # Save final model
    final_path = output_dir / "final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved to {final_path}")


if __name__ == "__main__":
    jsonargparse.auto_cli(main, as_positional=False)
