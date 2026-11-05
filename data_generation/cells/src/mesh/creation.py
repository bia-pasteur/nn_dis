"""Creates the gmsh mesh for finite elements computations"""

from collections.abc import Callable

import dolfinx
import gmsh  # type: ignore
import numpy as np
import pyvista
from dolfinx import io
from dolfinx.io import XDMFFile
from dolfinx.io import gmsh as gmshio
from mpi4py import MPI


def gmsh_volume_from_image_3d(
    img: np.ndarray,
    model,
    name: str,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[object, np.ndarray, np.ndarray]:
    """
    Create a 3D tetrahedral mesh from a binary image by subdividing
    each active voxel into tetrahedra, using physical voxel spacing.

    Args:
        img (np.ndarray): 3D image
        model: gmsh.model
        name (str): model name
        spacing (Tuple[float, float, float], optional): voxel size as (dz, dy, dx). Defaults to (1.0, 1.0, 1.0).

    Returns:
        Tuple[gmsh.model, np.ndarray, np.ndarray]:
            - model (gmsh.model): Gmsh model with the cell mesh added.
            - vertices (np.ndarray): Array of vertex coordinates.
            - tets (np.ndarray): Array of tetrahedral connectivity.
    """
    model.add(name)
    model.setCurrent(name)

    vol = (img > 0).astype(np.uint8)
    nz, ny, nx = vol.shape

    dz, dy, dx = spacing

    node_map = {}
    vertices = []
    tets = []

    def get_node(ix, iy, iz):
        key = (ix, iy, iz)
        if key not in node_map:
            node_map[key] = len(vertices) + 1  # Gmsh tags start at 1
            vertices.append([float(ix * dx), float(iy * dy), float(iz * dz)])
        return node_map[key]

    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                if vol[z, y, x] == 0:
                    continue

                n0 = get_node(x, y, z)
                n1 = get_node(x + 1, y, z)
                n2 = get_node(x + 1, y + 1, z)
                n3 = get_node(x, y + 1, z)
                n4 = get_node(x, y, z + 1)
                n5 = get_node(x + 1, y, z + 1)
                n6 = get_node(x + 1, y + 1, z + 1)
                n7 = get_node(x, y + 1, z + 1)

                tets.extend(
                    [
                        [n0, n1, n2, n6],
                        [n0, n2, n3, n6],
                        [n0, n3, n7, n6],
                        [n0, n7, n4, n6],
                        [n0, n4, n5, n6],
                        [n0, n5, n1, n6],
                    ]
                )

    if len(vertices) == 0 or len(tets) == 0:
        raise ValueError("Input image contains no active voxel.")

    vertices = np.asarray(vertices, dtype=np.float64)
    tets = np.asarray(tets, dtype=np.int64)

    vol_tag = model.addDiscreteEntity(3)

    node_tags = np.arange(1, len(vertices) + 1, dtype=np.int64)
    model.mesh.addNodes(3, vol_tag, node_tags, vertices.ravel())

    tet_tags = np.arange(1, len(tets) + 1, dtype=np.int64)

    model.mesh.addElements(3, vol_tag, [4], [tet_tags], [tets.ravel()])

    pg = model.addPhysicalGroup(3, [vol_tag])
    model.setPhysicalName(3, pg, "CellVolume")

    return model, vertices, tets


def create_mesh(
    comm: MPI.Comm, model: gmsh.model, name: str, filename: str, mode: str
) -> None:
    """
    Create a DOLFINx mesh from a Gmsh model and write it to an XDMF file.

    Args:
        comm (MPI.Comm): MPI communicator to create the mesh on.
        model (gmsh.model): Gmsh model.
        name (str): Name (identifier) of the mesh.
        filename (str): Path to the output XDMF file.
        mode (str): XDMF file writing mode. "w" to write, "a" to append.

    Returns:
        None
    """
    mesh_data = gmshio.model_to_mesh(model, comm, rank=0)

    mesh_data.mesh.name = name
    if mesh_data.cell_tags is not None:
        mesh_data.cell_tags.name = f"{name}_cells"
    if mesh_data.facet_tags is not None:
        mesh_data.facet_tags.name = f"{name}_facets"
    if mesh_data.ridge_tags is not None:
        mesh_data.ridge_tags.name = f"{name}_ridges"
    if mesh_data.peak_tags is not None:
        mesh_data.peak_tags.name = f"{name}_peaks"
    with XDMFFile(mesh_data.mesh.comm, filename, mode) as file:
        mesh_data.mesh.topology.create_connectivity(
            mesh_data.mesh.topology.dim - 1, mesh_data.mesh.topology.dim
        )
        file.write_mesh(mesh_data.mesh)
        if mesh_data.cell_tags is not None:
            file.write_meshtags(
                mesh_data.cell_tags,
                mesh_data.mesh.geometry,
                geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{name}']/Geometry",
            )
        if mesh_data.facet_tags is not None:
            file.write_meshtags(
                mesh_data.facet_tags,
                mesh_data.mesh.geometry,
                geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{name}']/Geometry",
            )
        if mesh_data.ridge_tags is not None:
            file.write_meshtags(
                mesh_data.ridge_tags,
                mesh_data.mesh.geometry,
                geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{name}']/Geometry",
            )
        if mesh_data.peak_tags is not None:
            file.write_meshtags(
                mesh_data.peak_tags,
                mesh_data.mesh.geometry,
                geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{name}']/Geometry",
            )


def create_mesh_file(
    mesh_function: Callable,
    img: np.ndarray = None,
    physical_length: float = None,
    rng=None,
    num_points=None,
    noise_amplitude=None,
    num_fourier_modes=None,
    lc=None,
) -> tuple[np.ndarray, np.ndarray, dolfinx.mesh.Mesh]:
    """
    Creates a computational mesh using a user-defined mesh function,
    writes it to XDMF, and reads it back as a DOLFINx mesh.

    Args:
        mesh_function (Callable): Function that generates the Gmsh model. Either
            gmsh_cell_from_image or gmsh_cell_shape.
        img (np.ndarray, optional): Binary mask image of a single cell if creating
            the mesh from an image. Defaults to None.
        physical_length (float, optional): Base radius of the cell if creating
            the mesh from a random shape. Defaults to None.
        rng (np.random.Generator, optional): Random number generator instance to
            ensure reproducibility. Defaults to None.
        num_points (int, optional): Number of boundary points if creating the mesh
            from a random shape. Defaults to None.
        noise_amplitude (float, optional): Amplitude of Fourier noise if creating
            the mesh from a random shape. Defaults to None.
        num_fourier_modes (int, optional): Number of Fourier modes if creating
            the mesh from a random shape. Defaults to None.
        lc (float, optional): Characteristic mesh length. Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray, dolfinx.mesh.Mesh]:
            - x_coords (np.ndarray): x coordinates of the cell boundary.
            - y_coords (np.ndarray): y coordinates of the cell boundary.
            - msh (dolfinx.mesh.Mesh): The generated computational mesh ready for FEM simulation.
    """
    # Initialize gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)  # 1 to enable terminal output

    # Create model
    model_ = gmsh.model()

    if img is not None:
        model, x_coords, y_coords = mesh_function(img, model_, "Grid")

    else:
        model, x_coords, y_coords = mesh_function(
            model_,
            "Grid",
            physical_length,
            rng,
            num_points,
            noise_amplitude,
            num_fourier_modes,
            lc,
        )

    # Fix the order of the elements // à préciser
    model.mesh.set_order(1)

    # Create the file
    create_mesh(MPI.COMM_SELF, model, "Grid", "out_gmsh/mesh_grid.xdmf", "w")

    with io.XDMFFile(MPI.COMM_WORLD, "./out_gmsh/mesh_grid.xdmf", "r") as xdmf:
        # Read the mesh
        msh = xdmf.read_mesh(name="Grid")

    return x_coords, y_coords, msh


def create_mesh_file_3d(
    mesh_function: Callable,
    img: np.ndarray,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[np.ndarray, np.ndarray, dolfinx.mesh.Mesh]:
    """
    Creates a computational mesh using a user-defined mesh function,
    writes it to XDMF, and reads it back as a DOLFINx mesh.

    Args:
        mesh_function (Callable): Function that generates the Gmsh model. For now,
            can only be gmsh_volume_from_image_3d
        img (np.ndarray, optional): Binary mask image of a single cell if creating
            the mesh from an image. Defaults to None.
        spacing (Tuple[float, float, float], optional): voxel size as (dz, dy, dx). Defaults to (1.0, 1.0, 1.0).

    Returns:
        Tuple[np.ndarray, np.ndarray, gmsh.model]:
            - vertices (np.ndarray): Array of vertex coordinates.
            - tets (np.ndarray): Array of tetrahedral connectivity.
            - model (gmsh.model): Gmsh model with the cell mesh added.
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)

    model_ = gmsh.model

    model, vertices, faces = mesh_function(img, model_, "Grid", spacing)

    model.mesh.setOrder(1)
    for dim in [2, 3]:
        for _, tag in model.getEntities(dim):
            elem_types, elem_tags, node_tags = model.mesh.getElements(dim, tag)
    create_mesh(MPI.COMM_SELF, model, "Grid", "out_gmsh/mesh_grid.xdmf", "w")

    with io.XDMFFile(MPI.COMM_WORLD, "./out_gmsh/mesh_grid.xdmf", "r") as xdmf:
        msh = xdmf.read_mesh(name="Grid")

    gmsh.finalize()

    return vertices, faces, msh


def visualize_mesh(V: dolfinx.fem.functionspace) -> None:
    """
    Creates an interactive visualization of a mesh using PyVista.

    Args:
        V (dolfinx.fem.FunctionSpace): The function space associated with the mesh to visualize.

    Returns:
        None
    """
    p = pyvista.Plotter()
    topology, cell_types, geometry = dolfinx.plot.vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

    p.add_mesh(grid, style="wireframe", color="k")

    p.view_yx()
    p.show_axes()
    p.camera.roll -= 90
    p.show()
