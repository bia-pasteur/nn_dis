from pathlib import Path

import matplotlib.pyplot as plt
import napari
import numpy as np
import pandas as pd
import pyvista as pv


def plot_losses(
    folder: str,
    loss_parts: list[str],
    weights: list[float],
    init_epoch: int = 0,
):
    """
    Plot the loss parts of a model

    Args:
        folder (str): Folder containing the .csv file with the loss over epochs.
        loss_parts (list[str]): Paths of the loss to be displayed.
        weights (list[float]): Weights associated with each loss part.
        init_epoch (int, optional): Epoch to start the visualization at. Defaults to 0.
    """
    train_logs = (
        pd.read_csv(f"{folder}/logs_train.csv")
        .iloc[init_epoch + 1 :]
        .reset_index(drop=True)
    )
    val_logs = (
        pd.read_csv(f"{folder}/logs_validation.csv")
        .iloc[init_epoch:]
        .reset_index(drop=True)
    )

    n_epochs = min(len(train_logs), len(val_logs))
    train_logs = train_logs.iloc[:n_epochs]
    val_logs = val_logs.iloc[:n_epochs]

    epochs = range(init_epoch + 1, init_epoch + n_epochs + 1)

    n_losses = len(loss_parts)

    _, axes = plt.subplots(
        1,
        n_losses,
        figsize=(6 * n_losses, 5),
    )

    if n_losses == 1:
        axes = [axes]

    for i, loss_part in enumerate(loss_parts):
        name = "loss" if loss_part == "total" else f"loss_{loss_part}"

        if name == "loss":
            train_values = train_logs[name]
            val_values = val_logs[name]
            title = "Total loss"
        else:
            weight = weights[i - 1]

            train_values = weight * train_logs[name]
            val_values = weight * val_logs[name]
            title = f"{name} * {weight}"

        axes[i].plot(
            epochs, train_values, label="Train loss (next epoch)", color="blue"
        )
        axes[i].plot(epochs, val_values, label="Validation loss", color="red")

        axes[i].set_title(title)
        axes[i].set_xlabel("Validation epoch")
        axes[i].set_ylabel("Loss")
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def add_flows_viewer(
    flow: np.ndarray,
    flow_name: str,
    ax: int,
    viewer: napari.Viewer,
    spacing: tuple[float, float, float],
) -> None:
    """
    Add a flow visualization on a given ax to a napari viewer.

    Args:
        flow (np.ndarray): The flow.
        flow_name (str): The name of the flow.
        ax (int): The ax index. Can only be 0, 1 or 2 in 3D.
        viewer (napari.Viewer): The existing viewer.
        spacing (tuple[float, float, float]): The spacing or scale of the image (z, y, x).

    Raises:
        ValueError: If ax is higher than the dimension.
    """
    if ax > flow.shape[0]:
        raise ValueError(f"ax can only be 0, 1 or 2 in 3D. Got {ax}")
    img = viewer.add_image(
        flow[ax, 0],
        name=f"{flow_name}, ax={ax}",
        scale=spacing,
        colormap="cividis",
        interpolation3d="nearest",
    )
    img.colorbar.visible = True


def get_viewer_flows(
    img: np.ndarray,
    flows: list[np.ndarray],
    flows_name: list[str],
    spacing: tuple[float, float, float],
    axes: list[int],
):
    """
    Create a napari viewer to observe an image and the different displacement
    fields on various axes.

    Args:
        img (np.ndarray): The image.
        flows (list[np.ndarray]): List of the displacement fields to display.
        flows_name (list[str]): Names of the displayed displacement fields.
        spacing (tuple[float, float, float]): Spacing or scale of the image (z, y, x).
        axes (list[int]): Axes to show the displacement fields on.
    """
    viewer = napari.Viewer()
    viewer.add_image(
        img,
        scale=spacing,
        interpolation3d="nearest",
    )

    for i, flow in enumerate(flows):
        for ax in axes:
            add_flows_viewer(flow, flows_name[i], ax, viewer, spacing)

    napari.run()


def plot_features_shape(
    save_path: str | Path,
    display: bool = False,
) -> None:
    """
    Create a figure of the evolution of the image shape
    and the features of the network vs the network stage.

    Args:
        save_path (str | Path): Path to the directory where the .png figure will be saved.
        display (bool, optional): Whether to display the figure or not. Defaults to False.
    """
    features = [16, 32, 32, 64, 64, 64]

    multiplier_HW = [1, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 16]
    multiplier_D = [1, 1, 1, 1 / 2, 1 / 4, 1 / 4]

    def mirror(values):
        return values + values[-2::-1]

    features = mirror(features)
    multiplier_HW = mirror(multiplier_HW)
    multiplier_D = mirror(multiplier_D)

    steps = np.arange(len(features))

    fig, ax_left = plt.subplots(figsize=(7, 3))
    ax_right = ax_left.twinx()

    fig.patch.set_alpha(0)
    ax_left.set_facecolor("none")
    ax_right.set_facecolor("none")

    (line_features,) = ax_left.plot(
        steps,
        features,
        color="green",
        linewidth=2,
        label="Features",
    )
    ax_left.set_ylabel("Features", color="green")
    ax_left.tick_params(axis="y", labelcolor="green")
    ax_left.set_yticks([16, 32, 48, 64])

    (line_hw,) = ax_right.plot(
        steps,
        multiplier_HW,
        color="red",
        linewidth=2,
        label="Height/Width multiplier",
    )
    (line_d,) = ax_right.plot(
        steps,
        multiplier_D,
        color="blue",
        linewidth=2,
        label="Depth multiplier",
    )

    ax_right.set_ylabel("Shape multiplier")
    ax_right.set_yticks([1 / 16, 1 / 8, 1 / 4, 1 / 2, 1])
    ax_right.set_yticklabels(["1/16", "1/8", "1/4", "1/2", "1"])

    ax_left.set_xticks(steps)
    ax_left.set_xticklabels(
        [
            "Enc. 1",
            "Enc. 2",
            "Enc. 3",
            "Enc. 4",
            "",
            "Bottleneck",
            "",
            "Dec. 1",
            "Dec. 2",
            "Dec. 3",
            "Dec. 4",
        ]
    )
    ax_left.set_xlabel("Network stage")

    ax_left.axvline(5, color="gray", linestyle=":", alpha=0.6)
    ax_left.grid(axis="y", linestyle="--", alpha=0.25)
    ax_left.set_axisbelow(True)

    lines = [line_features, line_hw, line_d]
    ax_left.legend(
        lines,
        [line.get_label() for line in lines],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=3,
        frameon=False,
    )

    fig.tight_layout()

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    output_path = save_path / "features_and_shapes.png"

    fig.savefig(
        output_path,
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    if display:
        plt.show()


def save_scatter_comparison(
    results: dict,
    results_dir: str | Path,
    methods: list[tuple[str, str]],
    strain: bool = False,
):
    """
    Save a plot that displays the NRMSE vs run time of different OF methods and models.

    Args:
        results (dict): Dictionary containing the results for every OF method.
        results_dir (str | Path): Directory where the .png figure will be saved.
        methods (list[tuple[str, str]]): Methods to display onthe scatter plot.
        strain (bool, optional): Whether to plot the strain vs the run time. Defaults to False.
    """
    fig, axes = plt.subplots(
        1, 2 if strain else 1, figsize=(12, 6) if strain else (6, 3), squeeze=False
    )
    axes = axes[0]

    colors = plt.get_cmap("tab10")
    markers = ["o", "s", "^", "D", "v", "P", "X"]

    quantities = [
        ("mean_rmse_disp", "Mean displacement NRMSE"),
    ]
    if strain:
        quantities.append(("mean_rmse_strain", "Mean strain NRMSE"))

    for _, (ax, (result_key, ylabel)) in enumerate(zip(axes, quantities)):
        for i, (name, method_key) in enumerate(methods):
            x = results["mean_runtime"][method_key]
            y = results[result_key][method_key]

            ax.scatter(
                x,
                y,
                color=colors(i % 10),
                marker=markers[i % len(markers)],
                s=150,
                zorder=3,
            )

            offset = (8, 8)
            va = "bottom"

            if method_key.lower() == "fista":
                offset = (-110, -10)
                va = "top"

            if result_key == "mean_rmse_disp":
                if method_key.lower() == "hs":
                    offset = (-40, 20)
                    va = "top"

            if result_key == "mean_rmse_strain":
                if method_key.lower() == "tvl1":
                    offset = (-70, 5)
                    va = "top"

            ax.annotate(
                name,
                (x, y),
                xytext=offset,
                textcoords="offset points",
                fontsize=17,
                fontweight="bold",
                va=va,
            )

        ax.set_xscale("log")
        ax.set_xlabel("Mean runtime (s)", fontsize=15)
        ax.set_ylabel(ylabel, fontsize=15)
        ax.grid(True, alpha=0.3)
        ax.margins(x=0.2, y=0.2)

    fig.tight_layout()

    fig.savefig(
        results_dir / "scatter_nrmse_mean.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_rmse_vs_noise(
    results: dict,
    results_dir: str | Path,
    methods: list[tuple[str, str]],
    metric: str | list[str] = "disp",
    display: bool = False,
) -> None:
    """
    Plot mean RMSE versus noise standard deviation for one or more metrics.

    Each metric is plotted in a separate subplot, with a single shared legend.

    Args:
        results: Results dictionary containing "noise_stds" and the
            "mean_rmse_<metric>_vs_noise" keys.
        results_dir: Directory where the PNG figure will be saved.
        methods: (display name, results key) pairs of methods to plot.
        metric: Metric or list of metrics to plot: "disp", "strain", "def",
            "stress", or "traction". Defaults to "disp".
        display: Whether to display the figure. Defaults to False.
    """
    metrics = [metric] if isinstance(metric, str) else list(metric)
    if not metrics:
        raise ValueError("'metric' must contain at least one metric.")

    required_keys = ["noise_stds"] + [f"mean_rmse_{m}_vs_noise" for m in metrics]
    for key in required_keys:
        if key not in results:
            raise ValueError(
                f"'{key}' not found in results: compute the results with "
                "`compute_of_strain_traction(..., noise_stds=...)` first."
            )

    noise_stds = np.asarray(results["noise_stds"])

    labels = {
        "disp": "Displacement",
        "strain": "Strain",
        "def": "Deformation",
        "stress": "Stress",
        "traction": "Traction",
    }

    n_metrics = len(metrics)
    fig, axes = plt.subplots(
        1,
        n_metrics,
        figsize=(7 * n_metrics, 7),
        sharex=True,
        squeeze=False,
    )

    colors = plt.get_cmap("tab10")
    markers = ["o", "s", "^", "D", "v", "P", "X"]

    for ax, m in zip(axes.flat, metrics):
        mean_key = f"mean_rmse_{m}_vs_noise"

        for i, (name, method_key) in enumerate(methods):
            means = np.asarray(results[mean_key][method_key])

            ax.plot(
                noise_stds,
                means,
                color=colors(i % 10),
                marker=markers[i % len(markers)],
                markersize=7,
                linewidth=2.0,
                label=name,
                zorder=3,
            )

        ax.set_xlabel("Noise standard deviation", fontsize=17)
        ax.set_ylabel(f"{labels.get(m, m)} NRMSE", fontsize=17)
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()

    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=len(handles),
        fontsize=19,
        frameon=True,
        fancybox=True,
        framealpha=1.0,
        edgecolor="0.7",
        facecolor="white",
        borderpad=0.6,
        columnspacing=2.0,
        handletextpad=0.6,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.85))

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    filename = f"rmse_{'_'.join(metrics)}_vs_noise.png"
    fig.savefig(
        results_dir / filename,
        dpi=300,
        bbox_inches="tight",
    )

    if display:
        plt.show()
    plt.close(fig)


def save_rmse_vs_train_size(
    results: dict,
    results_dir: str | Path,
    methods: list[tuple[str, str]],
    train_sizes: dict[str, float],
    metric: str | list[str] = "disp",
    display: bool = False,
) -> None:
    """
    Plot mean RMSE versus training dataset size for one or more metrics.

    Each metric is plotted in a separate subplot, as a single curve connecting
    the trainings sorted by training size.

    Args:
        results: Results dictionary containing the "mean_rmse_<metric>" keys,
            as returned by `compute_of_strain_traction` on the trained models.
        results_dir: Directory where the PNG figure will be saved.
        methods: (display name, results key) pairs of trainings to plot.
        train_sizes: Mapping from results key to the number of training images
            of the corresponding training. Every method key must be present.
        metric: Metric or list of metrics to plot: "disp", "strain", "def",
            "stress", or "traction". Defaults to "disp".
        display: Whether to display the figure. Defaults to False.
    """
    metrics = [metric] if isinstance(metric, str) else list(metric)
    if not metrics:
        raise ValueError("'metric' must contain at least one metric.")

    required_keys = [f"mean_rmse_{m}" for m in metrics]
    for key in required_keys:
        if key not in results:
            raise ValueError(
                f"'{key}' not found in results: compute the results with "
                "`compute_of_strain_traction(...)` first."
            )

    missing_sizes = [
        method_key for _, method_key in methods if method_key not in train_sizes
    ]
    if missing_sizes:
        raise ValueError(
            f"No training size given for: {missing_sizes}. "
            "Provide every method key in 'train_sizes'."
        )

    labels = {
        "disp": "Displacement",
        "strain": "Strain",
        "def": "Deformation",
        "stress": "Stress",
        "traction": "Traction",
    }

    n_metrics = len(metrics)
    fig, axes = plt.subplots(
        1,
        n_metrics,
        figsize=(7 * n_metrics, 7),
        squeeze=False,
    )

    color = "blue"
    marker = "o"

    for ax, m in zip(axes.flat, metrics):
        mean_key = f"mean_rmse_{m}"

        # Sort the methods by training size so the curve is drawn in order.
        sorted_methods = sorted(methods, key=lambda item: train_sizes[item[1]])

        xs = [train_sizes[method_key] for _, method_key in sorted_methods]
        ys = [results[mean_key][method_key] for _, method_key in sorted_methods]

        ax.plot(
            xs,
            ys,
            color=color,
            marker=marker,
            markersize=7,
            linewidth=2.0,
            zorder=3,
        )

        all_sizes = sorted(train_sizes[method_key] for _, method_key in methods)
        ax.set_xticks(all_sizes)
        ax.set_xticklabels(all_sizes)
        ax.set_xlabel("Training dataset size (number of images)", fontsize=17)
        ax.set_ylabel(f"{labels.get(m, m)} RMSE", fontsize=17)
        ax.grid(True, which="both", alpha=0.3)
        ax.set_axisbelow(True)

    fig.tight_layout(rect=(0, 0, 1, 0.85))

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    filename = f"rmse_{'_'.join(metrics)}_vs_train_size.png"
    fig.savefig(
        results_dir / filename,
        dpi=300,
        bbox_inches="tight",
    )

    if display:
        plt.show()
    plt.close(fig)


def get_visu_img(
    img: np.ndarray,
    name: str,
    cmap: str,
    save_path: str | Path,
    display: bool = False,
) -> None:
    """
    Get a 3D visualization of the image.

    Args:
        img (np.ndarray): Image.
        name (str): Name to use to save the figure.
        cmap (str): Colormap to use in the figure.
        save_path (str | Path): Path to the directory where the .png figure will be saved.
        display (bool, optional): Whether to display the figure or not. Defaults to False.
    """
    nz, ny, nx = img.shape

    iz, iy, ix = nz // 2, ny // 2, nx // 2

    fig = plt.figure(figsize=(10, 10), layout="none")

    gap = 0.005
    main = 0.72
    side = 0.24
    left = bottom = 0.02

    ax_xy = fig.add_axes([left, bottom + side + gap, main, main])
    ax_yz = fig.add_axes([left + main + gap, bottom + side + gap, side, main])
    ax_xz = fig.add_axes([left, bottom, main, side])

    ax_xy.imgshow(img[iz, :, :], cmap=cmap, aspect="auto")
    ax_yz.imshow(img[:, :, ix].T, cmap=cmap, aspect="auto")
    ax_xz.imshow(img[:, iy, :], cmap=cmap, aspect="auto")

    for ax, label in [(ax_xy, "XY"), (ax_yz, "YZ"), (ax_xz, "XZ")]:
        ax.set_axis_off()
        ax.text(
            0.01,
            0.99,
            label,
            transform=ax.transAxes,
            color="white",
            fontsize=25,
            fontweight="bold",
            ha="left",
            va="top",
        )

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    output_path = save_path / f"{name}_coupes.png"

    fig.savefig(
        output_path, dpi=300, bbox_inches="tight", transparent=True, pad_inches=0
    )

    if display:
        plt.show()


def show_dis(
    img: np.ndarray,
    img_clean: np.ndarray,
    flow: np.ndarray,
    name: str,
    arrow_factor: int,
    stride: int,
    zoom: float,
    z_size: float,
    save_path: str | Path,
    display: bool = False,
    colorbar: bool = False,
) -> None:
    """
    Get a visualization of a vector field on an image.

    Args:
        img (np.ndarray): Image.
        flow (np.ndarray): Field.
        name (str): Name for saving.
        arrow_factor (int): Factor to mutliply the arrows for visualization.
        stride (int): Step between each displayed arrow.
        zoom (float): Zoom value on the image for visualization.
        z_size (float): Size of the voxels in z.
        save_path (str | Path): Path to the directory where the .png figure will be saved.
        display (bool, optional): Whether to display the figure or not. Defaults to False.
        colorbar (bool, optional): Whether to display to colorbars for each quantity. Defaults to False.
    """
    image = img[0]
    mask = img_clean > 0.1

    # Define spacing ((x, y, z) for pyvista)
    spacing = np.array([1.0, 1.0, z_size])

    # Image of the cell to display in background
    grid = pv.ImageData(
        dimensions=image.shape[::-1],
        spacing=tuple(spacing),
    )
    grid["intensity"] = np.where(mask, image, 0).ravel(order="C")

    # Picks vectors in the mask, sampling with stride
    z, y, x = (np.argwhere(mask[::stride, ::stride, ::stride]) * stride).T

    # Convert from (z, y, x) (numpy) to (x, y, z) (pyvista).
    points = np.column_stack((x, y, z)) * spacing
    vectors = flow[:, z, y, x].T[:, ::-1] * spacing

    plotter = pv.Plotter(off_screen=not display)
    plotter.set_background("black")

    plotter.add_volume(
        grid,
        scalars="intensity",
        cmap="gray",
        clim=(0, float(image.max())),
        shade=False,
        show_scalar_bar=False,
    )

    if len(points):
        # Get the magnitude of the displacement
        magnitude = np.linalg.norm(vectors, axis=1)

        # Create arrows data (necessary because of arrow_factor the changes the scalar values associated with the arrows)
        arrows_data = pv.PolyData(points)
        arrows_data["vectors"] = vectors
        arrows_data["magnitude"] = magnitude

        arrows = arrows_data.glyph(
            orient="vectors",
            scale="magnitude",
            factor=arrow_factor,
        )

        plotter.add_mesh(
            arrows,
            scalars="magnitude",
            cmap="afmhot",
            show_scalar_bar=colorbar,
            scalar_bar_args={
                "title": "Displacement",
                "vertical": True,
                "position_x": 0.85,
                "position_y": 0.15,
                "width": 0.08,
                "height": 0.7,
                "color": "white",
                "fmt": "%.0f",
            },
        )

    plotter.view_isometric()
    plotter.camera.zoom(zoom)

    output_dir = Path(save_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    plotter.show(
        screenshot=str(output_dir / f"{name}_flow.png"),
        auto_close=True,
    )


def show_cells(
    img: np.ndarray,
    name: str,
    z_size: float,
    save_path: str | Path,
    display: bool = False,
    zoom: float = 1.0,
) -> None:
    """
    Get a visualization of two time frames in a cell image
    layered on top of one another.

    Args:
        img (np.ndarray): Image of a moving cell.
        name (str): Name for saving.
        z_size (float): Size of voxel in z.
        save_path (str | Path): Path to the directory where the .png figure will be saved.
        display (bool, optional): Whether to display the figure or not. Defaults to False.
        zoom (float, optional): Zoom value on the image for visualization. Defaults to 1.0.
    """
    img = np.asarray(img)
    spacing = (1.0, 1.0, z_size)

    grid = pv.ImageData(
        dimensions=img.shape[1:][::-1],
        spacing=spacing,
    )

    grid["mask"] = (img[0] > 0.1).astype(float).ravel(order="C")
    cell0 = grid.contour([0.5], scalars="mask")

    grid["mask"] = (img[1] > 0.1).astype(float).ravel(order="C")
    cell1 = grid.contour([0.5], scalars="mask")

    plotter = pv.Plotter(
        off_screen=not display,
        window_size=(600, 600),
        border=False,
    )
    plotter.set_background("black")

    plotter.add_mesh(
        cell0,
        color="cyan",
        opacity=0.4,
        label="Reference",
    )
    plotter.add_mesh(
        cell1,
        color="magenta",
        opacity=0.4,
        label="Moving",
    )
    legend = plotter.add_legend(
        size=(0.3, 0.15),
        loc="upper right",
    )

    legend.SetPosition(0.65, 0.8)

    legend.GetEntryTextProperty().SetFontSize(24)

    plotter.add_axes(
        xlabel="X",
        ylabel="Y",
        zlabel="Z",
        color="white",  # couleur des labels
        viewport=(0.0, 0.0, 0.2, 0.2),  # coin inférieur gauche
    )

    plotter.view_isometric()
    plotter.camera.zoom(zoom)
    plotter.camera.SetWindowCenter(-0.1, 0.0)

    output_dir = Path(save_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    plotter.show(
        screenshot=str(output_dir / f"{name}_cells.png"),
        auto_close=True,
    )


def show_dis_def_traction(
    img: np.ndarray,
    results: dict,
    name: str,
    arrow_factor_dis: float,
    stride_dis: int,
    z_size: float,
    arrow_factor_trac: list[float] = [1.0, 1.0, 1.0],
    stride_trac: int = 2,
    cell_opacity: float = 0.10,
    display: bool = False,
    save_path: str | Path = ".",
    zoom: float = 1.0,
    colorbar: bool = False,
) -> None:
    """
    Plot displacement, deformation, and traction for each method
    on an image.

    Args:
        img (np.ndarray): Image.
        results (dict): Dictionary containing the results.
        name (str): Name for saving.
        arrow_factor_dis (float): Factor to multiply the displacement vectors by for visualization.
        stride_dis (int): Step between each displayed displacement vector.
        z_size (float): Size of the voxels in z.
        arrow_factor_trac (list[float], optional): Factor to multiply the tracion vectors by for visualization.
                                                   One value per method. Defaults to [1.0, 1.0, 1.0].
        stride_trac (int, optional): Step between each displayed traction vector. Defaults to 2.
        cell_opacity (float, optional): Opacity of the cell displayed under the vector fields. Defaults to 0.10.
        display (bool, optional): Whether to display the figure or not. Defaults to False.
        save_path (str | Path, optional): Path to the directory where the .png figure will be saved. Defaults to ".".
        zoom (float, optional): Zoom value on the image for visualization. Defaults to 1.0.
        colorbar (bool, optional): Whether to display to colorbars for each quantity. Defaults to False.

    Raises:
        ValueError: If the mechanical quantities do not all have th esame amount of methods.
    """
    img = np.asarray(img)
    flows = [np.asarray(f, dtype=float) for f in results["flows"]]
    tractions = [np.asarray(t, dtype=float) for t in results["tractions"]]
    deformations = [np.asarray(d, dtype=float) for d in results["def"]]

    n = len(flows)
    if n == 0:
        raise ValueError("At least one method is required.")

    if len(tractions) != n or len(deformations) != n:
        raise ValueError(
            "flows, tractions, and def must contain the same number of methods."
        )

    spacing = np.array([1.0, 1.0, z_size])
    mask = img[0] > 0.1

    grid = pv.ImageData(
        dimensions=mask.shape[::-1],
        spacing=tuple(spacing),
    )
    grid["mask"] = mask.astype(float).ravel(order="C")
    cell0 = grid.contour([0.5], scalars="mask")

    # Displacement: sample vectors within the reference cell.
    z, y, x = (
        np.argwhere(mask[::stride_dis, ::stride_dis, ::stride_dis]) * stride_dis
    ).T

    points_dis = np.column_stack((x, y, z)) * spacing
    vectors_dis = [flow[:, z, y, x].T[:, ::-1] * spacing for flow in flows]

    max_dis = max(
        np.linalg.norm(v, axis=1).max() if len(v) else 0.0 for v in vectors_dis
    )

    # Deformation: common color limits.
    max_def = max(np.max(np.abs(d)) for d in deformations)

    # Traction: sample nonzero vectors.
    points_trac = []
    vectors_trac = []

    for traction in tractions:
        magnitude = np.linalg.norm(traction, axis=0)
        indices = np.argwhere(magnitude > 0)[::stride_trac]
        z, y, x = indices.T

        points_trac.append(np.column_stack((x, y, z)) * spacing)
        vectors_trac.append(traction[:, z, y, x].T[:, ::-1])

    max_trac = max(np.linalg.norm(traction, axis=0).max() for traction in tractions)
    # Layout:
    # column 0: measure names
    # columns 1 ... n: methods
    # column n+1: optional colorbars
    window_height = 2000
    label_width = 240
    method_width = 1000
    colorbar_width = 400

    column_widths = [label_width] + [method_width] * n
    if colorbar:
        column_widths.append(colorbar_width)

    plotter = pv.Plotter(
        shape=(3, len(column_widths)),
        col_weights=column_widths,
        off_screen=not display,
        window_size=(sum(column_widths), window_height),
        border=False,
    )
    plotter.set_background("black", all_renderers=True)

    colorbar_rows = set()

    for i in range(n):
        column = i + 1

        for row in range(3):
            plotter.subplot(row, column)

            plotter.add_mesh(
                cell0,
                color="white",
                opacity=cell_opacity,
            )

            actor = None

            # Displacement and traction: arrows.
            if row in (0, 2):
                if row == 0:
                    points = points_dis
                    vectors = vectors_dis[i]
                    factor = arrow_factor_dis
                    color_max = max_dis
                else:
                    points = points_trac[i]
                    vectors = vectors_trac[i]
                    factor = arrow_factor_trac[i]
                    color_max = max_trac

                magnitude = np.linalg.norm(vectors, axis=1)
                valid = np.isfinite(magnitude) & (magnitude > 0)

                if valid.any():
                    arrows_data = pv.PolyData(points[valid])
                    arrows_data["vectors"] = vectors[valid]
                    arrows_data["magnitude"] = magnitude[valid]

                    arrows = arrows_data.glyph(
                        orient="vectors",
                        scale="magnitude",
                        factor=factor,
                    )

                    actor = plotter.add_mesh(
                        arrows,
                        scalars="magnitude",
                        cmap="afmhot" if row == 0 else "coolwarm",
                        clim=(0, color_max or 1.0),
                        show_scalar_bar=False,
                    )

            # Deformation: scalar volume.
            else:
                volume = grid.copy()
                volume["deformation"] = np.where(mask, deformations[i], 0).ravel(
                    order="C"
                )

                actor = plotter.add_volume(
                    volume,
                    scalars="deformation",
                    cmap="viridis",
                    clim=(0, max_def or 1.0),
                    shade=False,
                    show_scalar_bar=False,
                )

            # One colorbar per row, using the first available actor.
            if colorbar and actor is not None and row not in colorbar_rows:
                plotter.subplot(row, n + 1)
                plotter.add_scalar_bar(
                    title=" " * (row + 1),
                    mapper=actor.mapper,
                    vertical=True,
                    position_x=0.1,
                    position_y=0.15,
                    width=0.3,
                    height=0.8,
                    color="white",
                    fmt="%.2g" if row == 1 else "%.0f",
                    font_family="arial",
                    label_font_size=35,
                )
                colorbar_rows.add(row)

    # Set the camera from the first method's displacement panel.
    plotter.subplot(0, 1)
    plotter.view_isometric()
    plotter.camera.zoom(zoom)
    camera = plotter.camera

    # Share the camera only among the 3D panels.
    for i in range(n):
        for row in range(3):
            plotter.subplot(row, i + 1)
            plotter.camera = camera

    output_dir = Path(save_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    plotter.show(
        screenshot=str(output_dir / f"{name}_comparison.png"),
        auto_close=True,
    )
