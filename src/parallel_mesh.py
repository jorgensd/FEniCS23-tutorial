# # Extra topic: Reading in meshes in parallel
# For more information about mesh-partitioning and mesh input, see: https://jsdokken.com/dolfinx_docs/meshes.html
# ## Cell ownership
#
# For the remainder of this section we will consider a 3x3 unit square mesh:

from mpi4py import MPI
import dolfinx
import ipyparallel as ipp

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
    domain = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 3, 3, ghost_mode=ghost_mode)
    topology = domain.topology
    tdim = topology.dim
    print(f"Number of cells in process: {topology.index_map(tdim).size_local}", flush=True)
    print(f"Number of shared cells: {topology.index_map(tdim).num_ghosts}", flush=True)
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
