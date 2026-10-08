"""
Created on Sat Aug 05 23:55:12 2017

@author: Kazushige Okayasu, Hirokatsu Kataoka
"""

import matplotlib

matplotlib.use("Agg")  # useful if running without GUI/server
import numpy as np


class IFSFunction3D:
    def __init__(self, prev_x, prev_y, prev_z, rng: np.random.Generator):
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
        Add a transformation with rotation matrix R and translation vector T.

        Parameters:
        - R: 3x3 rotation/scaling matrix
        - T: 3x1 translation vector
        - proba: probability weight for this transformation
        - kwargs: optional weights to apply to the matrix/vector
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
        Perform the IFS iterations in 3D space.

        Parameters:
        - nb_iteration: number of iterations to perform
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

        # --- Compress the huge dynamic range of the point histogram ---
        image = np.log1p(image)

        # --- Percentile-based normalization to [0, 1], robust to outlier peaks ---
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
