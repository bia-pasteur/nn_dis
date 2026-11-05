"""Solves the FEM problem"""

# pylint: disable=invalid-name
# pylint: disable=reportMissingImports
from collections.abc import Callable
from dataclasses import dataclass

import dolfinx
import numpy as np
import ufl
from dolfinx import default_scalar_type, fem, mesh
from dolfinx.fem.petsc import apply_lifting, assemble_matrix, assemble_vector, set_bc
from petsc4py import PETSc
from ufl import TestFunction, TrialFunction, inner


def strain(u: ufl.Argument) -> ufl.Form:
    """
    Computes the strain of the displacement field

    Args:
        u (ufl.Argument): Displacement field
            Shape: (d,)
    Returns:
        ufl.Form: The strain tensor
            Shape: (d, d)
    """
    return ufl.sym(ufl.grad(u))


def stress_elas(u: ufl.Argument, lambda_e: float, mu_e: float) -> ufl.Form:
    """
    Computes the elastic stress of the displacement field (Hooke's law)

    Args:
        u (ufl.Argument): Displacement field
            Shape: (d,)
        lambda_e (float): Lamé elastic parameter
        mu_e (float): Lamé elastic parameter

    Returns:
        ufl.Form: The elastic stress tensor
            Shape: (d, d)
    """
    return lambda_e * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu_e * strain(u)


def stress_visc(
    u: ufl.Argument, lambda_v: float, mu_v: float, delta_t: float
) -> ufl.Form:
    """
    Computes the viscous stress tensor of the displacmeent field

    Args:
        u (ufl.Argument): Displacement field
            Shape: (d,)
        lambda_v (float): Lamé viscous parameter
        mu_v (float): Lamé viscous parameter
        delta_t (float): The time step

    Returns:
        ufl.Form: The viscous stress tensor
            Shape: (d, d)
    """
    return (lambda_v / delta_t) * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * (
        mu_v / delta_t
    ) * strain(u)


@dataclass
class TractionZone:
    """Class for a zone of traction application on the cell"""

    tag: int
    facets: np.ndarray
    max_traction: float
    center: fem.Constant
    radius: float
    sigma: float
    direction: np.ndarray


def extreme_boundary_node(msh: mesh.Mesh, axis: int, sign: int = 1) -> int:
    """
    Return the boundary node with the most extreme coordinate along `axis`.

    Among nodes tied at the extreme value (e.g. a whole voxel face at max x),
    the node closest to the boundary centroid in the two other axes is chosen,
    so the traction patch sits in the middle of the extreme face.

    Args:
        msh (mesh.Mesh): Finite element mesh of the computational domain
        axis (int): Axis along which to take the extreme (0 = x, 1 = y, 2 = z)
        sign (int): +1 for the maximum coordinate, -1 for the minimum

    Returns:
        int: Node index into `msh.geometry.x`
    """
    fdim = msh.topology.dim - 1
    msh.topology.create_connectivity(fdim, 0)
    msh.topology.create_connectivity(fdim, msh.topology.dim)

    boundary_facets = mesh.exterior_facet_indices(msh.topology)
    f_to_v = msh.topology.connectivity(fdim, 0)
    boundary_nodes = np.unique(
        np.concatenate([f_to_v.links(int(f)) for f in boundary_facets])
    )

    coords = msh.geometry.x
    values = coords[boundary_nodes, axis]
    extreme = values.max() if sign > 0 else values.min()
    tol = 1e-9 * max(1.0, abs(extreme))
    tied = boundary_nodes[np.abs(values - extreme) <= tol]

    centroid = coords[boundary_nodes].mean(axis=0)
    other_axes = [a for a in range(3) if a != axis]
    dists = np.linalg.norm(coords[tied][:, other_axes] - centroid[other_axes], axis=1)
    return int(tied[np.argmin(dists)])


def single_traction_zone(
    msh: mesh.Mesh,
    range_radius: tuple[float, float],
    range_traction: tuple[float, float],
    rng: np.random.Generator,
    tag: int,
    existing_zones: list[TractionZone],
    min_gap_factor: float = 0.0,
    axis: int | None = None,
    sign: int = 1,
) -> TractionZone | None:
    """
    Creates a traction zone with a random center and radius on the boundary of the cell.

    If `axis` is given, the zone is instead pinned to the most extreme boundary
    node along that axis (sign +1 for max, -1 for min) and pulls along that axis.

    Args:
        msh (mesh.Mesh): Finite element mesh of the computational domain
        range_radius (tuple[float, float]): Range of values for the radius of the zone
        range_traction (tuple[float, float]): Range of values for the traction applied in the zone
        rng (np.random.Generator): Random generator for reproducibility
        tag (int): Numerical identifier of the zone
        existing_zones (list[TractionZone]): Existing zones of traction of application to avoid overlapping
        min_gap_factor (float, optional): Minimum permitted gap between two zones of the traction. Defaults to 0.0.
        axis (int, optional): If given, pin the zone to the extreme boundary node along this
            axis (0 = x, 1 = y, 2 = z) and pull along it. Defaults to None (random zone).
        sign (int, optional): Direction of the pull on the pinned axis (+1 or -1). Defaults to 1.

    Returns:
        TractionZone | None: The traction zone
    """
    dim = msh.topology.dim
    fdim = dim - 1
    coords = msh.geometry.x
    num_nodes = coords.shape[0]

    if axis is not None:
        node_id = extreme_boundary_node(msh, axis, sign)
        center = coords[node_id]
        direction = np.zeros(3)  # Order (x, y, z)
        direction[axis] = float(sign)
        radius = rng.uniform(*range_radius)
        magnitude = rng.uniform(*range_traction)
    else:
        max_tries = 30
        node_id = None
        for _ in range(max_tries):
            # maximum number of trials to get a valid center
            node_id = rng.integers(num_nodes)
            # # maximum number of trials to get a valid center
            # node_id = rng.integers(num_nodes)
            center = coords[node_id]
            # print(center)
            radius = rng.uniform(*range_radius)

            # test distance to other existing centers
            overlap = False
            for z in existing_zones:
                dist = np.linalg.norm(center - np.array(z.center.value))
                if dist < (radius + z.radius) * (1 - min_gap_factor):
                    overlap = True
                    break
            if overlap:
                continue

            direction = rng.normal(size=3)  # Order (x, y, z)
            direction /= np.linalg.norm(direction)
            magnitude = rng.uniform(*range_traction)
            break
        else:
            return None

    # facettes inside the sphere
    def is_inside_radius(x, center=center, radius=radius):
        d = np.linalg.norm(x.T - center, axis=1)
        return d <= radius

    facets = mesh.locate_entities_boundary(msh, fdim, is_inside_radius)
    if len(facets) == 0:
        return None

    return TractionZone(
        tag=tag,
        facets=facets,
        max_traction=magnitude,
        center=fem.Constant(msh, center),
        radius=radius,
        sigma=2,
        direction=direction,
    )


def create_traction_zones(
    msh: mesh.Mesh,
    range_radius: tuple[float, float],
    range_traction: tuple[float, float],
    rng: np.random.Generator,
    n_zones: int | None = None,
    min_gap_factor: float = 0.0,
    axes: tuple[int, ...] | None = None,
) -> list[TractionZone]:
    """
    Create at most one traction zone per selected axis.

    Axis mapping:
        0 = x, 1 = y, 2 = z

    Selection:
        - axes given: use those axes explicitly.
        - axes omitted, n_zones given: randomly select n_zones distinct axes.
        - both omitted: randomly select between 1 and 3 distinct axes.

    Zones are placed at the positive extreme of each selected axis and
    pull in that axis's positive direction.
    """
    if axes is not None:
        selected_axes = list(axes)

        if not selected_axes:
            raise ValueError("axes must contain at least one axis.")

        if any(axis not in (0, 1, 2) for axis in selected_axes):
            raise ValueError("Axes must be 0 (x), 1 (y), or 2 (z).")

        if len(set(selected_axes)) != len(selected_axes):
            raise ValueError("Each axis may appear only once.")

        if n_zones is not None and n_zones != len(selected_axes):
            raise ValueError("n_zones must match len(axes) when both are given.")

    else:
        if n_zones is None:
            n_zones = int(rng.integers(1, 4))

        if n_zones not in (1, 2, 3):
            raise ValueError("n_zones must be 1, 2, or 3.")

        selected_axes = [
            int(axis) for axis in rng.choice(3, size=n_zones, replace=False)
        ]

    zones: list[TractionZone] = []

    for tag, axis in enumerate(selected_axes, start=1):
        zone = single_traction_zone(
            msh=msh,
            range_radius=range_radius,
            range_traction=range_traction,
            rng=rng,
            tag=tag,
            existing_zones=zones,
            min_gap_factor=min_gap_factor,
            axis=axis,
            sign=1,
        )

        if zone is not None:
            zones.append(zone)

    return zones


def build_zone_markers(
    msh: mesh.Mesh, zones: list[TractionZone]
) -> mesh.MeshTags | None:
    """
    Constructs a MeshTags object labeling each boundary facet of the mesh
    according to the traction zone it belongs to.

    Args:
        msh (mesh.Mesh): Finite element mesh of the computational domain.
        zones (list[TractionZone]): List of traction zone objects, each containing
            the set of boundary facets and the associated zone tag.

    Returns:
        mesh.MeshTags | None:
            A MeshTags instance mapping boundary facets to their corresponding
            zone tag, or None if no zone was provided.
    """
    if not zones:
        return None

    fdim = msh.topology.dim - 1

    facet_to_tag = {int(f): zone.tag for zone in zones for f in zone.facets}
    indices = np.fromiter(sorted(facet_to_tag), dtype=np.int32)
    values = np.fromiter((facet_to_tag[i] for i in indices), dtype=np.int32)
    return mesh.meshtags(msh, fdim, indices, values)


def finite_elements_force_zone_3d(
    msh: dolfinx.mesh.Mesh,
    V: fem.FunctionSpace,
    cond_init: Callable[[np.ndarray], np.ndarray],
    t_end: float,
    num_time_steps: int,
    youngs_modulus: float,
    nu: float,
    eta: float,
    range_radius: tuple[float, float],
    range_traction: tuple[float, float],
    rng: np.random.Generator,
    min_gap_factor: float = 0.0,
    n_zones: int | None = None,
    axes: tuple[int, ...] | None = None,
) -> list[np.ndarray]:
    """
    Performs a simulation of a Kelvin-Voigt viscoelastic or linear elastic deformation
    on a 3d mesh representing a cell under different traction forces applied at its
    boundary.

    The traction zones are randomly generated inside the function.

    Args:
        msh (mesh.Mesh): Finite element mesh of the computational domain
        V (fem.FunctionSpace): Function space for displacements
        cond_init (Callable): Initial condition function (displacement field)
        t_end (float): Total simulation time
        num_time_steps (int): Number of discrete time steps
        youngs_modulus (float): Young’s modulus of the cell, outside of eventual subparts
        nu (float): Poisson's ratio of the cell, outside of eventual subparts
        eta (float): Viscosity of the cell, outside of eventual subparts
        nb_zones (int): Number of random traction zones to be created
        range_radius (tuple[float, float]): Range of values for the radius of the zone
        range_traction (tuple[float, float]): Range of values for the traction applied in the zone
        rng (np.random.Generator): Random generator for reproducibility
        min_gap_factor (float, optional): Minimum permitted gap between two zones of the traction. Defaults to 0.0.
    Returns:
        Tuple(List[np.ndarray], List[TractionZone]):
            - Displacement field at each time step
            - List of the traction zones
    """

    # Define the time step
    delta_t = t_end / num_time_steps

    # Compute Lamé parameters
    def lame(E__, nu__, eta__):
        # Elastic Lamé parameters
        lambda_e = E__ * nu__ / ((1 + nu__) * (1 - 2 * nu__))
        mu_e = E__ / (2 * (1 + nu__))

        # Viscous Lamé parameters
        lambda_v = -eta__ / 3
        mu_v = eta__ / 2
        return lambda_e, mu_e, lambda_v, mu_v

    # Assign interior elements a Young's modulus, Poisson's ratio and viscosity
    F = fem.functionspace(msh, ("DG", 0))
    youngs_modulus_ = fem.Function(F)
    eta_ = fem.Function(F)
    nu_ = fem.Function(F)
    youngs_modulus_.x.array[:] = youngs_modulus
    nu_.x.array[:] = nu
    eta_.x.array[:] = eta
    lambda_e, mu_e, lambda_v, mu_v = lame(youngs_modulus_, nu_, eta_)

    # Define a function for the old values
    u_old = fem.Function(V, name="Previous step")
    u_old.interpolate(cond_init)

    # Define solution variable, and interpolate initial solution
    uh = fem.Function(V, name="Solution variable")
    uh.interpolate(cond_init)

    # Definition of the integration measures
    dx = ufl.Measure("dx", domain=msh)
    displacement_history = []

    zones = create_traction_zones(
        msh=msh,
        range_radius=range_radius,
        range_traction=range_traction,
        rng=rng,
        n_zones=n_zones,
        min_gap_factor=min_gap_factor,
        axes=axes,
    )
    zone_markers = build_zone_markers(msh, zones)

    fdim = msh.topology.dim - 1
    all_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.full(x.shape[1], True)
    )
    traction_facets = (
        np.concatenate([z.facets for z in zones])
        if zones
        else np.array([], dtype=np.int32)
    )
    dirichlet_facets = np.setdiff1d(all_facets, traction_facets)

    dofs_dirichlet = fem.locate_dofs_topological(V, fdim, dirichlet_facets)
    bc = fem.dirichletbc(np.zeros(3, dtype=default_scalar_type), dofs_dirichlet, V)

    # Create the trial and test function
    u, v = TrialFunction(V), TestFunction(V)
    # Weak form
    a = (
        inner(
            stress_elas(u, lambda_e, mu_e) + stress_visc(u, lambda_v, mu_v, delta_t),
            strain(v),
        )
        * dx
    )
    L = inner(stress_visc(u_old, lambda_v, mu_v, delta_t), strain(v)) * dx

    x = ufl.SpatialCoordinate(msh)

    ds_zones = ufl.Measure("ds", domain=msh, subdomain_data=zone_markers)

    for zone in zones:
        distance = ufl.sqrt(ufl.dot(x - zone.center, x - zone.center))

        bump = ufl.conditional(
            distance <= zone.radius, (1.0 - (distance / zone.radius) ** 2) ** 2, 0.0
        )

        magnitude = zone.max_traction * bump
        trac = magnitude * ufl.as_vector(zone.direction)
        L += ufl.inner(trac, v) * ds_zones(zone.tag)

    # Convert the left hand side (LSH) and the right hand side (RHS) into a DOLFINx-compatible representation
    compiled_a = fem.form(a)
    compiled_L = fem.form(L)
    # Allocate memory for the solution
    b = fem.Function(V)

    # Assemble the stiffness matrix A from a and apply Dirichlet boundary conditions
    # Linear system : Au = b
    A = assemble_matrix(compiled_a, bcs=[bc])
    A.assemble()
    # Create the solver
    solver = PETSc.KSP().create(msh.comm)  # pylint: disable=no-member

    # Set the system matrix
    solver.setOperators(A)

    # Using direct factorization
    solver.setType(PETSc.KSP.Type.PREONLY)  # pylint: disable=no-member

    # Using LU decomposition as a precondition
    # Fast for small to medium problems, memory-intensive
    solver.getPC().setType(PETSc.PC.Type.LU)  # pylint: disable=no-member

    displacement_history.append(uh.copy())
    # Iteratively solve the problem
    for _ in range(num_time_steps):
        # Reset the RHS
        b.x.array[:] = 0
        # Assemble the RHS => construct Au = b
        assemble_vector(b.x.petsc_vec, compiled_L)
        # Apply the Dirichlet boundary conditions
        apply_lifting(b.x.petsc_vec, [compiled_a], [[bc]])
        set_bc(b.x.petsc_vec, [bc])
        # Solve the linear system
        solver.solve(b.x.petsc_vec, uh.x.petsc_vec)
        uh.x.scatter_forward()
        set_bc(uh.x.petsc_vec, [bc])
        # Update the old values
        u_old.x.array[:] = uh.x.array.copy()
        # Store displacement (either as an array or a copy of uh)
        displacement_history.append(uh.copy())
    return displacement_history, zones
