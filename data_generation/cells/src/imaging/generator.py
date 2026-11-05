"""Generator of the images"""

# pylint: disable=invalid-name
# pylint: disable=line-too-long
# pylint: disable=trailing-whitespace
import os

import dolfinx
import numba
import numpy as np
import tifffile
from dolfinx import fem
from dolfinx.mesh import compute_midpoints
from petsc4py.PETSc import ScalarType  # pylint: disable=no-name-in-module

from data_generation.cells.src.fem.solver import (
    finite_elements_force_zone_3d,
)
from data_generation.cells.src.mesh.creation import (
    create_mesh_file_3d,
    gmsh_volume_from_image_3d,
)


def closest_cell(
    mesh: dolfinx.mesh.Mesh, point: np.ndarray, candidate_cells: np.ndarray
) -> int:
    """
    Find the cell (element) in the mesh that is closest to a given point,
    restricted to a list of candidate cells.

    Args:
        mesh (dolfinx.mesh.Mesh): Computational mesh.
        point (np.ndarray): Target point in physical coordinates, shape (3,).
        candidate_cells (np.ndarray): Array of candidate cell indices.

    Returns:
        int: Index of the cell in `candidate_cells` whose midpoint
             is closest to the given point.
    """
    midpoints = compute_midpoints(
        mesh, mesh.topology.dim, np.array(candidate_cells, dtype=np.int32)
    )
    dists = np.linalg.norm(midpoints - point, axis=1)
    return candidate_cells[np.argmin(dists)]


@numba.njit
def fill_3d(
    H: int,
    W: int,
    D: int,
    pixels: np.ndarray,
    pixels_interpolation: np.ndarray,
    u_values: np.ndarray,
    val: np.ndarray,
    mask: np.ndarray,
):
    """
    Fill a 3D displacement field and validity mask on a regular grid.

    Args:
        H (int): Height of the 2D grid.
        W (int): Width of the 2D grid.
        D (int): Depth of the 2D grid.
        pixels (np.ndarray): Array of shape `(H, W, 3)` containing the physical
            coordinates of the grid points.
        pixels_interpolation (np.ndarray): Array of shape `(N, 3)` containing the
            subset of grid points for which interpolation values are available.
        u_values (np.ndarray): Array of shape `(N, m)` containing the evaluated
            field values at `pixels_interpolation`. Only the first two components
            are used.
        val (np.ndarray): Output array of shape `(H, W, D, 3)` where the interpolated
            2D vector values are written in place.
        mask (np.ndarray): Output array of shape `(H, W, D, 3)` used as a validity
            mask. Entries are set to 1 where interpolation was successful.

    Returns:
        None
    """
    count = 0
    for i in range(H):
        for j in range(W):
            for k in range(D):
                if count < len(pixels_interpolation) and np.all(
                    pixels[i, j, k] == pixels_interpolation[count]
                ):
                    val[i, j, k] = u_values[count, :3]
                    mask[i, j, k] = 1
                    count += 1


def interpolation(
    mesh: dolfinx.mesh.Mesh,
    u: fem.Function,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float] | None = None,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """
    Interpolates a finite element vector field u on a regular 2D or 3D grid.

    Args:
        mesh (dolfinx.mesh.Mesh): Computational mesh where u is defined.
        u (fem.Function): Vector-valued FE function (displacement field)
        x_range (Tuple[float, float]): (xmin, xmax) bounds of interpolation domain along x-axis.
        y_range (Tuple[float, float]): (ymin, ymax) bounds of interpolation domain along y-axis.
        z_range (Optional[Tuple[float, float]]): (zmin, zmax) bounds along z-axis. If None → 2D interpolation. Defaults to None.
        spacing (Tuple[float, float, float], optional): voxel size as (dz, dy, dx). Defaults to (1.0, 1.0, 1.0).

    Returns:
        np.ndarray: Array of interpolated values.
            - Shape (H, W, 2) for 2D interpolation
            - Shape (H, W, D, 3) for 3D interpolation
    """
    dz, dy, dx = spacing

    H, W = int(y_range[1]), int(x_range[1])

    x = (np.linspace(x_range[0], x_range[1], W, endpoint=True)) * dx
    y = (np.linspace(y_range[0], y_range[1], H, endpoint=True)) * dy

    if z_range is None:
        # 2D case
        X, Y = np.meshgrid(x, y, indexing="ij")
        pixels = np.stack((X, Y, np.zeros_like(X)), axis=-1)
        dim = 2
        val_shape = (H, W, 2)
    else:
        # 3D case
        D = int(z_range[1])
        z = (np.linspace(z_range[0], z_range[1], D, endpoint=False)) * dz
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        pixels = np.stack((X, Y, Z), axis=-1)
        dim = 3
        val_shape = (W, H, D, 3)

    flattened_pix = pixels.reshape(-1, 3)

    bounding_box = dolfinx.geometry.bb_tree(mesh, dim)
    cells = dolfinx.geometry.compute_collisions_points(bounding_box, flattened_pix)
    colliding_cells = dolfinx.geometry.compute_colliding_cells(
        mesh, cells, flattened_pix
    )

    pixels_interpolation = []
    cells_interpolation = []

    for i, pixel in enumerate(flattened_pix):
        if len(colliding_cells.links(i)) > 0:
            best_cell = closest_cell(mesh, pixel, colliding_cells.links(i))
            pixels_interpolation.append(pixel)
            cells_interpolation.append(best_cell)

    pixels_interpolation = np.array(pixels_interpolation, dtype=np.float64)

    u_values = u.eval(pixels_interpolation, cells_interpolation)

    val = np.zeros(val_shape)
    mask = np.zeros(val_shape)

    fill_3d(H, W, D, pixels, pixels_interpolation, u_values, val, mask)

    return val, mask


@numba.njit
def _map_cells_to_image_3d(
    coords: np.ndarray, img: np.ndarray, dz: float, dy: float, dx: float
) -> np.ndarray:
    """
    Assign an image intensity to each mesh cell from its midpoint coordinates.

    For each cell midpoint coordinate `(X, Y, Z)`, the corresponding voxel
    index `(z, y, x)` is computed using the physical voxel spacing. The voxel
    indices are clipped to remain within image bounds, and the intensity at
    that voxel is assigned to the cell.

    Args:
        coords (np.ndarray): Array of cell midpoint coordinates of shape `(N, 3)`,
            where each row contains the physical coordinates `[X, Y, Z]` of a cell.
        img (np.ndarray): 3D image array of shape `(Z, Y, X)` containing the
            intensity values to map onto the mesh cells.
        dz (float): Voxel size along the z-axis.
        dy (float): Voxel size along the y-axis.
        dx (float): Voxel size along the x-axis.

    Returns:
        np.ndarray: Array of shape `(N,)` containing the image intensity assigned
        to each cell midpoint.
    """

    num_cells = coords.shape[0]
    nz, ny, nx = img.shape
    values_per_cell = np.zeros(num_cells, dtype=np.float64)

    for i in range(num_cells):
        X = coords[i, 0]
        Y = coords[i, 1]
        Z = coords[i, 2]

        x = int(np.floor(X / dx))
        y = int(np.floor(Y / dy))
        z = int(np.floor(Z / dz))

        if x < 0:
            x = 0
        elif x >= nx:
            x = nx - 1

        if y < 0:
            y = 0
        elif y >= ny:
            y = ny - 1

        if z < 0:
            z = 0
        elif z >= nz:
            z = nz - 1

        values_per_cell[i] = img[z, y, x]

    return values_per_cell


def map_intensities_from_image_3d(
    mesh: dolfinx.mesh.Mesh,
    img: np.ndarray,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    normalize: bool = False,
    percentiles: tuple[float, float] = (1, 99),
) -> dolfinx.fem.Function:
    """
    Create a discontinuous Galerkin (DG) function on the mesh where
    the cell-wise values follow the brightness pattern of a 3d cell image

    Args:
        mesh (dolfinx.mesh.Mesh): The input computational mesh.
        img (np.ndarray): Image of a masked cell
        spacing (Tuple[float, float, float], optional): voxel size as (dz, dy, dx). Defaults to (1.0, 1.0, 1.0).
        normalize (bool, optional): Wether to normalize the intensities. Defaults to False.
        percentiles (tuple[float, float], optional): Percentiles used for normalization. Defaults to (1, 99).

    Returns:
        dolfinx.fem.Function: DG function representing the intensity of the cell image over the mesh.

    """
    Q = fem.functionspace(mesh, ("DG", 0))
    intensities = fem.Function(Q)

    tdim = mesh.topology.dim
    num_cells = mesh.topology.index_map(tdim).size_local

    cell_indices = np.arange(num_cells, dtype=np.int32)
    coords = compute_midpoints(mesh, tdim, cell_indices)

    dz, dy, dx = spacing

    values_per_cell = _map_cells_to_image_3d(coords, img, dz, dy, dx)

    if normalize:
        min_val, max_val = np.percentile(values_per_cell, percentiles)
        if max_val > min_val:
            values_per_cell = np.clip(values_per_cell, min_val, max_val)
            values_per_cell = (values_per_cell - min_val) / (max_val - min_val)
        else:
            values_per_cell[:] = 0.0

    intensities.x.array[:] = values_per_cell
    return intensities


def deform_mesh(mesh: dolfinx.mesh.Mesh, u: dolfinx.fem.Function) -> dolfinx.mesh.Mesh:
    """
    Deform the input mesh according to a given displacement field.

    The displacement field u is evaluated at the mesh geometry points,
    and each node of the mesh is updated by adding the corresponding
    displacement vector.

    Args:
        mesh (dolfinx.mesh.Mesh): The computational mesh to deform.
        u (dolfinx.fem.Function): Displacement field defined on the mesh.

    Returns:
        dolfinx.mesh.Mesh: The same mesh object with updated coordinates.
    """
    x = mesh.geometry.x
    # Initialise cell search
    bounding_box = dolfinx.geometry.bb_tree(mesh, mesh.topology.dim)
    cells = []
    points_on_proc = []
    # Find cells whose bounding-box collide with the the points
    cell_candidates = dolfinx.geometry.compute_collisions_points(bounding_box, x)
    # Choose one of the cells that contains the point
    colliding_cells = dolfinx.geometry.compute_colliding_cells(mesh, cell_candidates, x)
    for i, point in enumerate(x):
        if len(colliding_cells.links(i)) > 0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])
    points_on_proc = np.array(points_on_proc, dtype=np.float64)
    # Evaluate u
    u_values = u.eval(points_on_proc, cells)
    # Update mesh coordinates
    x += u_values

    return mesh


def create_image_simu_3d(
    img: np.ndarray,
    spacing: tuple[float, float, float],
    t_end: float,
    num_time_steps: int,
    youngs_modulus: float,
    nu: float,
    eta: float,
    range_traction: tuple[float, float],
    rng: np.random.Generator,
    name: str,
    min_gap_factor: float = 0.0,
    n_zones: int | None = None,
    axes: tuple[int, ...] | None = None,
) -> tuple[np.ndarray, str]:
    """
    Generate a sequence of simulated 3D microscopy images based on a finite element
    (FE) deformation simulation.

    Creates a mesh from an image, applies a time-dependent displacement due to a local
    traction force, and interpolates the deformed intensity field on a regular grid to
    produce a synthetic 3D image sequence saved as a TIFF.

    Args:
        img (np.ndarray): 3d image of a masked cell.
        spacing (Tuple[float, float, float], optional): voxel size as (dz, dy, dx). Defaults to (1.0, 1.0, 1.0).
        t_end (float): Total simulation time
        num_time_steps (int): Number of discrete time steps
        youngs_modulus (float): Young’s modulus of the cell, outside of eventual subparts
        nu (float): Poisson's ratio of the cell, outside of eventual subparts
        eta (float): Viscosity of the cell, outside of eventual subparts
        nb_zones (int): Number of random traction zones to be created
        range_traction (tuple[float, float]): Range of values for the traction applied in the zone
        rng (np.random.Generator): Random generator for reproducibility
        name (str): Base name for the output TIFF file and saved figures.
        min_gap_factor (float, optional): Minimum permitted gap between two zones of the traction. Defaults to 0.0.

    Returns:
        Tuple[np.ndarray, str]:
            - u_list (np.ndarray): Displacement fields interpolated on a regular grid,
              shape (num_time_steps, D, H, W, 3), dtype float32, where H and W are the
              grid dimensions inferred from y_range and x_range.
            - warped_image_path (str): Path to the saved multi-frame TIFF file
              containing the warped image sequence, with axes 'TZYX'.
    """
    _, _, msh = create_mesh_file_3d(gmsh_volume_from_image_3d, img, spacing)

    xmin, ymin, zmin = msh.geometry.x.min(axis=0)
    xmax, ymax, zmax = msh.geometry.x.max(axis=0)
    bbox_diag = np.linalg.norm([xmax - xmin, ymax - ymin, zmax - zmin])
    range_radius = (0.1 * bbox_diag, 0.3 * bbox_diag)

    V = fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim,)))

    def cond_init(x):
        return np.array([0.0 * x[0], 0.0 * x[1], 0.0 * x[2]], dtype=ScalarType)

    dis, zones = finite_elements_force_zone_3d(
        msh=msh,
        V=V,
        cond_init=cond_init,
        t_end=t_end,
        num_time_steps=num_time_steps,
        youngs_modulus=youngs_modulus,
        nu=nu,
        eta=eta,
        range_radius=range_radius,
        range_traction=range_traction,
        rng=rng,
        n_zones=n_zones,
        min_gap_factor=min_gap_factor,
        axes=axes,
    )

    intensities = map_intensities_from_image_3d(msh, img, spacing)
    x_range = [0, img.shape[2]]
    y_range = [0, img.shape[1]]
    z_range = [0, img.shape[0]]
    H, W, D = y_range[1], x_range[1], z_range[1]

    warped_images = np.zeros((num_time_steps, D, H, W), dtype=np.float32)
    u_list = np.zeros((num_time_steps, D, H, W, 3), dtype=np.float32)

    uh_prev = fem.Function(V)
    u_total = fem.Function(V)
    u_total.x.array[:] = 0.0

    coords_initial = msh.geometry.x.copy()
    coords_current = coords_initial.copy()

    for time in range(num_time_steps):
        uh = dis[time]
        du = fem.Function(V)
        du.x.array[:] = uh.x.array - uh_prev.x.array
        u_total.x.array[:] += du.x.array

        msh.geometry.x[:] = coords_initial
        u_interp, _ = interpolation(msh, u_total, x_range, y_range, z_range, spacing)
        u_list[time] = np.transpose(u_interp, (2, 1, 0, 3))

        msh.geometry.x[:] = coords_current
        msh = deform_mesh(msh, du)

        coords_current = msh.geometry.x.copy()

        warped_im, _ = interpolation(
            msh, intensities, x_range, y_range, z_range, spacing
        )
        warped_images[time] = np.transpose(warped_im[..., 0], (2, 1, 0))
        uh_prev.x.array[:] = uh.x.array

    base_type = "elas" if eta == 0.0 else "viscoelas"
    base_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", f"data/raw/{base_type}/3d"
    )
    output_dir = os.path.join(base_dir, f"E_{youngs_modulus}_eta_{eta}_nu_{nu}_T_multi")
    os.makedirs(output_dir, exist_ok=True)

    zones_path = os.path.join(output_dir, f"{name}_zones.json")
    # save_zones_json(zones, zones_path)
    warped_image_path = os.path.join(output_dir, f"{name}.tiff")
    tifffile.imwrite(
        warped_image_path, warped_images, metadata={"axes": "TZYX"}, imagej=True
    )

    return u_list, warped_image_path, zones_path
