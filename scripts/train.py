#!/usr/bin/env python3
"""
Training of the network
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from pathlib import Path
from typing import Sequence

import jsonargparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import network
from config import TrainConfig
from scripts.dataset import TractionDataset, get_transforms


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
    hessian_loss_fn,
    loss_weights: Sequence[float],
    device: str = "cuda",
    training: bool = True,
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
        "loss": [],
        "loss_mse": [],
        "loss_grad": [],
        "loss_hess": [],
        "max_dis": [],
        "avg_dis": [],
    }

    if training:
        model.train()

    else:
        model.eval()

    total_loss = 0.0

    for batch in iter(dataloader):
        if training:
            optimizer.zero_grad()

        # Move to device in training loop (not dataloader/dataset!)
        source = batch["source"].to(device)
        target = batch["target"].to(device)
        mask = batch["mask"].to(device)

        # Get the displacement and the warped source image from the model
        displacement, warped_source = model(source, target, return_warped_source=True)

        img_loss = image_loss_fn(target, warped_source, mask)
        grad_loss = grad_loss_fn(displacement)
        hessian_loss = hessian_loss_fn(displacement)

        loss = (
            loss_weights[0] * img_loss
            + loss_weights[1] * grad_loss
            + loss_weights[2] * hessian_loss
        )

        if training:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

        log["loss"].append(loss.item())
        log["loss_mse"].append(img_loss.item())
        log["loss_grad"].append(grad_loss.item())
        log["loss_hess"].append(hessian_loss.item())
        log["max_dis"].append(np.max(np.abs(displacement.detach().cpu().numpy())))
        log["avg_dis"].append(np.mean(displacement.detach().cpu().numpy()))

    return {k: np.mean(v) for k, v in log.items()}


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
            source_channels=1,
            target_channels=1,
            features=train_config.features,
        ).to(device)

    elif train_config.model_type == "aniso":
        model = network.models.NetworkPairwiseAnisotropic(
            ndim=3,
            source_channels=1,
            target_channels=1,
            features=train_config.features,
        ).to(device)

    else:
        raise ValueError(
            f"The type can only be iso or aniso. Got {train_config.model_type}"
        )

    # Setup losses and optimizer
    image_loss_fn = network.losses.MSE()
    grad_loss_fn = network.losses.SpatialGradient("l2")
    hessian_loss_fn = network.losses.SpatialHessian("l2")

    loss_weights = [1.0, train_config.alpha_param, train_config.beta_param]
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.lr)

    # Create dataloader
    train_dataset = TractionDataset(
        Path(train_config.train_data), transform=get_transforms()
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        num_workers=train_config.workers,
        shuffle=True,
    )

    validation_dataset = TractionDataset(Path(train_config.val_data))
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
            loss_weights=loss_weights,
            device=device,
            training=True,
        )

        with torch.no_grad():
            epoch_log_validation = epoch_(
                model=model,
                dataloader=validation_loader,
                optimizer=optimizer,
                image_loss_fn=image_loss_fn,
                grad_loss_fn=grad_loss_fn,
                hessian_loss_fn=hessian_loss_fn,
                loss_weights=loss_weights,
                device=device,
                training=False,
            )

        update_log(logs_train, epoch_log_train)
        update_log(logs_validation, epoch_log_validation)

        pd.DataFrame.from_dict(logs_train).to_csv(output_dir / "logs_train.csv")
        pd.DataFrame.from_dict(logs_validation).to_csv(
            output_dir / "logs_validation.csv"
        )

        avg_loss = epoch_log_validation["loss"]
        # Print progress
        # if (epoch + 1) % 10 == 0:
        #     print(f"Epoch {epoch + 1}/{train_config.epochs}, Loss: {avg_loss:.6f}")

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
