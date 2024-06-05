# # Mesh generation
#
# So far, we have covered how to generate code for the local element assembly.
# We can now look into how to solve specific problems.
# Let us start with generating the computational domain.
#
# In DOLFINx, the mesh creation requires 4 inputs:
#
# - **MPI communicator**: This is used to decide how the partitioning is performed. It is usually `MPI.COMM_WORLD` or `MPI.COMM_SELF`.
# - **Nodes**: A set of coordinates in 1, 2 or 3D, that represents all the points in the mesh
# - **Connectivity**: A nested list, where each row corresponds to the node indices of a single cell
# - **Coordinate element**: A finite element used for pushing coordinates from the reference element to the physical element and its inverse.
#
# As an example, let us consider a single element mesh, of a triangle with straight edges
#

import ipyparallel as ipp
import dolfinx
import basix.ufl
import numpy as np
import ufl
from mpi4py import MPI
import pyvista
pyvista.start_xvfb(1.0)

nodes = np.array([[1., 0.], [2., 0.], [3., 2.], [1, 3]], dtype=np.float64)
connectivity = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)

c_el = ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 1, shape=(nodes.shape[1],)))
domain = dolfinx.mesh.create_mesh(MPI.COMM_SELF, connectivity, nodes, c_el)

# We start by creating a simple plotter by interfacing with Pyvista


def plot_mesh(mesh: dolfinx.mesh.Mesh):
    """
    Given a DOLFINx mesh, create a `pyvista.UnstructuredGrid`,
    and plot it and the mesh nodes
    """
    plotter = pyvista.Plotter()
    ugrid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(mesh))
    if mesh.geometry.cmap.degree > 1:
        plotter.add_mesh(ugrid, style="points", color="b", point_size=10)
        ugrid = ugrid.tessellate()
        show_edges = False
    else:
        show_edges = True
    plotter.add_mesh(ugrid, show_edges=show_edges)

    plotter.show_axes()
    plotter.view_xy()
    plotter.show()


# We can then easily visualize the domain
plot_mesh(domain)


# ## Higher order meshes
# If we want to create a mesh with higher order edges, we can supply the extra nodes to the mesh geometry, and adjust the coordinate element

nodes = np.array([[1., 0.], [2., 0.], [3., 2.],
                  [2.9, 1.3], [1.5, 1.5], [1.5, -0.2]], dtype=np.float64)
connectivity = np.array([[0, 1, 2, 3, 4, 5]], dtype=np.int64)

c_el = ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 2, shape=(nodes.shape[1],)))
domain = dolfinx.mesh.create_mesh(MPI.COMM_SELF, connectivity, nodes, c_el)
plot_mesh(domain)

# ## Interfacing with other software
# As DOLFINx works on `numpy`-arrays it is easy to convert any mesh format that be converted into this structure.
# DOLFINx has a `gmsh`-interface, using the GMSH Python-API to read `GMSH`-models or `.msh` files.
# For more information about mesh-partitioning and mesh input, see: https://jsdokken.com/dolfinx_docs/meshes.html


# ## Cell ownership
#
# For the remainder of this section we will consider a 3x3 unit square mesh:

domain = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 3, 3)

# The mesh consists of cells, edges and vertices.
# A mesh is created by supplying the information regarding the connectivity between the cells and the mesh nodes.
# As a higher order mesh has more nodes than vertices, we can get the connectivity between the cells and mesh vertices
# through the mesh-topology.

# We start by creating a simple function for inspecting these outputs in serial and parallel


def inspect_mesh(shared_facet: bool = False):
    from mpi4py import MPI
    import dolfinx
    ghost_mode = dolfinx.mesh.GhostMode.shared_facet if shared_facet else dolfinx.mesh.GhostMode.none
    domain = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD, 3, 3, ghost_mode=ghost_mode)
    topology = domain.topology
    tdim = topology.dim
    print(
        f"Number of cells in process: {topology.index_map(tdim).size_local}", flush=True)
    print(
        f"Number of shared cells: {topology.index_map(tdim).num_ghosts}", flush=True)
    print(f"Global range {topology.index_map(tdim).local_range}")
    cell_to_vertices = topology.connectivity(tdim, 0)
    print(cell_to_vertices)


# We start by inspecting the outputs in serial

inspect_mesh()

# We observe that we have 18 cells in the mesh, that are connected to the 18 vertices.
# All indices start from 0, and can be mapped to its global owner by calling `domain.topology.index_map(tdim).local_to_global([idx])`

# ## Parallel execution
# We use IPython-Parallel for running DOLFINx on multiple processes inside our script.

cluster = ipp.Cluster(engines="mpi", n=2)
rc = cluster.start_and_connect_sync()

# This has started to processes that can execute code with a MPI communicator.
# We run the mesh code on two processes, instructing DOLFINx not to share any cells between the processes

query = rc[:].apply_async(inspect_mesh, False)
query.wait()
query.display_outputs()

# Next, we instruct DOLFINx to share cells between two processes if a facet is shared between the two processes.

new_query = rc[:].apply_async(inspect_mesh, True)
new_query.wait()
new_query.display_outputs()
# + tags=["hide-output"]
cluster.stop_cluster_sync()
