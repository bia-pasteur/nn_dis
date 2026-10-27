"""
Created on Sat Aug 05 23:55:12 2017

@author: Kazushige Okayasu, Hirokatsu Kataoka
"""

import matplotlib

matplotlib.use("Agg")  # useful if running without GUI/server
import numpy as np


class IFSFunction3D:
    """
    Iterated function system in 3D, generating a fractal point cloud.

    Holds a set of affine maps x -> R x + T with selection probabilities.
    Each iteration picks one map according to those probabilities and applies
    it to the current point; the resulting trajectory of points is the
    fractal, rasterized later by ``make_intensity_volume_zyx``.
    """

    def __init__(self, prev_x, prev_y, prev_z, rng: np.random.Generator):
        """
        Initialize the IFS with a starting point and a random generator.

        Args:
            prev_x (float): x coordinate of the starting point.
            prev_y (float): y coordinate of the starting point.
            prev_z (float): z coordinate of the starting point.
            rng (np.random.Generator): Random generator for reproducibility.
        """
        # Previous position (x, y, z)
        self.prev_x, self.prev_y, self.prev_z = prev_x, prev_y, prev_z
        # IFS functions (each is a tuple (R, T) where R is 3x3 and T is 3x1)
        self.functions = []
        # Iterative results
        self.xs, self.ys, self.zs = [], [], []
        # Add initial value
        self.xs.append(prev_x)
        self.ys.append(prev_y)
        self.zs.append(prev_z)
        # Selection probabilities
        self.select_function = []
        # Calculate selection function
        self.temp_proba = 0.0
        self.rng = rng

    def set_param(self, R, T, proba, **kwargs):
        """
        Add a transformation x -> R x + T with selection probability proba.

        Args:
            R (np.ndarray): (3, 3) affine matrix (rotation/scaling).
            T (np.ndarray): (3,) translation vector.
            proba (float): Probability weight for this transformation;
                cumulative across calls, so the proba values should sum to 1.
            **kwargs: Optional weights weight_R and weight_T applied
                to the matrix and the translation before storing.
        """
        # Apply weights if provided
        if "weight_R" in kwargs:
            R = R * kwargs["weight_R"]
        if "weight_T" in kwargs:
            T = T * kwargs["weight_T"]

        # Store the transformation and update probabilities
        self.functions.append((R, T))
        self.temp_proba += proba
        self.select_function.append(self.temp_proba)

    def calculate(self, nb_iterations):
        """
        Run the IFS iterations in 3D space, appending one point per iteration.

        Starting from the initial position, each iteration picks a map
        according to the selection probabilities and applies it:
        next = R @ current + T.

        Args:
            nb_iterations (int): Number of iterations to perform.
        """
        # Fix random seed for reproducibility
        rng = self.rng
        rand = rng.random(nb_iterations)

        prev_x, prev_y, prev_z = self.prev_x, self.prev_y, self.prev_z

        for i in range(nb_iterations - 1):
            # Find which transformation to apply
            for j in range(len(self.select_function)):
                if rand[i] <= self.select_function[j]:
                    # Get the transformation components
                    R, T = self.functions[j]

                    # Apply the transformation: next = R * current + T
                    current = np.array([prev_x, prev_y, prev_z])
                    next_pos = R @ current + T

                    # Store the result
                    next_x, next_y, next_z = next_pos
                    self.xs.append(next_x)
                    self.ys.append(next_y)
                    self.zs.append(next_z)

                    # Update previous position
                    prev_x, prev_y, prev_z = next_x, next_y, next_z
                    break

    def make_intensity_volume_zyx(
        self,
        size_img_x,
        size_img_y,
        size_img_z,
        pad_x,
        pad_y,
        pad_z,
    ):
        """
        Rasterize the point cloud into a (z, y, x) intensity volume.

        Each axis is scaled so the point cloud spans [pad, size - 1 - pad]
        (leaving a border margin), points are rounded to voxel indices and
        counted. Counts are log1p-compressed, then normalized to [0, 1] by
        clipping at the 1st/99th percentiles of the nonzero voxels.

        Args:
            size_img_x (int): Size of the output volume along x.
            size_img_y (int): Size of the output volume along y.
            size_img_z (int): Size of the output volume along z.
            pad_x (int): Margin in voxels left empty on each x edge.
            pad_y (int): Margin in voxels left empty on each y edge.
            pad_z (int): Margin in voxels left empty on each z edge.

        Returns:
            np.ndarray: (size_img_z, size_img_y, size_img_x) float32 volume
            with values in [0, 1].
        """
        assert (
            size_img_x > 2 * pad_x and size_img_y > 2 * pad_y and size_img_z > 2 * pad_z
        ), (
            "Size must be strictly greater than 2 * pad on each axis, check config file !"
        )

        xs = np.asarray(self.xs, dtype=np.float64)
        ys = np.asarray(self.ys, dtype=np.float64)
        zs = np.asarray(self.zs, dtype=np.float64)

        valid = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(zs)

        xs = xs[valid]
        ys = ys[valid]
        zs = zs[valid]

        def scale_axis(values, size, pad):
            vmin = np.min(values)
            vmax = np.max(values)
            vrange = vmax - vmin

            return ((values - vmin) / vrange) * float(size - 2 * pad) + float(pad)

        xs = scale_axis(xs, size_img_x, pad_x)
        ys = scale_axis(ys, size_img_y, pad_y)
        zs = scale_axis(zs, size_img_z, pad_z)

        xs = np.rint(xs).astype(np.uint32)
        ys = np.rint(ys).astype(np.uint32)
        zs = np.rint(zs).astype(np.uint32)

        xs = np.clip(xs, 0, size_img_x - 1)
        ys = np.clip(ys, 0, size_img_y - 1)
        zs = np.clip(zs, 0, size_img_z - 1)

        image = np.zeros(
            (size_img_z, size_img_y, size_img_x),
            dtype=np.uint32,
        )

        np.add.at(image, (zs, ys, xs), 1)

        image = np.log1p(image)

        nonzero = image[image > 0]
        if nonzero.size == 0:
            return image.astype(np.float32)

        lo, hi = np.percentile(nonzero, [1.0, 99.0])
        if hi <= lo + 1e-12:
            hi = nonzero.max()
            lo = 0.0

        image = np.clip(image, lo, hi)
        image = (image - lo) / max(hi - lo, 1e-12)
        image = np.clip(image, 0.0, 1.0)

        return image.astype(np.float32)
