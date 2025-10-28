# (mesh_generation:eshelby)=
# # Meshes from external sources
# As DOLFINx works on `numpy`-arrays it is quite straightforward to convert any mesh format into this structure.
# DOLFINx has a `gmsh`-interface, using the {term}`GMSH` Python-API to read `GMSH`-models or `.msh` files.
# We will start with creating a simple circular geometry (similar to the one used in the
# [Eshelby problem](https://github.com/msolides-2024/MU5MES01-2024/tree/main/02-Eshelby_inclusion)) using GMSH.
# The Eshelby has an ellipsoid inclusion in the circular domain.

from mpi4py import MPI
import dolfinx
import gmsh

# The first thing we do when using GMSH is to initialize it explicitly

gmsh.initialize()

# A single GMSH instance can create multiple separate geometries, named models.
# We start by creating a model

gmsh.model.add("eshelby")

# ## Generating parameterized geometries

# Next, we use the Open Cascade {term}`OCC` backend to create a two disks, `inner_disk` with radius `R_i`
# and an `outer_disk` with radius `R_e`.
# The disks will have a center at the origin and we can select an aspect ratio to make the inner disk elliptical

# +
center = (0, 0, 0)
aspect_ratio = 0.5
R_i = 0.3
R_e = 0.8

inner_disk = gmsh.model.occ.addDisk(*center, R_i, aspect_ratio * R_i)
outer_disk = gmsh.model.occ.addDisk(*center, R_e, R_e)
# -

# What GMSH returns for each of the `addDisk` calls is an integer, which internally represents a geometry
# object of topological dimension two.
#

# + tags = ["remove-input"]
print(f"{inner_disk=}")
print(f"{outer_disk=}")
# -

# ```{admonition} GMSH objects and mesh resolution
# The disk objects has no concept of mesh resolution.
# This is controlled at a later stage using GMSH size field or meshing options, where a finer
# resolution will more closely approximate the disks.
# Two-dimensional meshing algorithms uses the parameterization of the circle to create the mesh.
# ```

# ## Boolean operations of entities

# In GMSH, you can combine multiple geometries into a single mesh (by unions, intersections or differences).
# We use the `fragment` function to embed the boundary of the inner circle in the outer disk.

# Now that we have created two parametric representations of a disk, we would like to create a combined surface,
# where each of the circular boundaries are included.

whole_domain, map_to_input = gmsh.model.occ.fragment([(2, outer_disk)], [(2, inner_disk)])

# We can inspect the first output of the fragment function

# + tags = ["remove-input"]
print(f"{whole_domain=}")
# -

# We observe that we get a list of tuples. Each tuple `(dim, obj_idx)` represents an {term}`OCC` object of
# dimension `dim` and internal index `obj_index`. We observe that we have two surfaces of dimension 2.
# The second output of fragment is a tuple of such lists, where the first list contains objects
# that are embedded in the first input object (`outer_disk`), and the second list contains objects
# that are embedded in the second input object (`inner_disk`).
# As the `outer_disk` contains the inner disk, we observe that the first list and second list shares
# an object

# + tags = ["remove-input"]
assert len(map_to_input) == 2
print(f"{map_to_input[0]=}")
print(f"{map_to_input[1]=}")
# -

# To make it possible for other functions in GMSH than those in the OCC module to modify
# the `domain`, we need to synchronize the model.

gmsh.model.occ.synchronize()

# ## Physical groups

# In order to ensure that only the parts of the mesh that you would like to use is generated (and no overlapping cells),
# you need to create `Physical Groups` for all volumes, surfaces and curves you would like to have in your mesh.
# We start by creating a physical marker for the inner circle.
# The `addPhysicalGroup` function takes three arguments:
# 1. The topological dimension of the entity
# 2. A list of GMSH objects that should be tagged
# 3. The marker that should be assigned to the tagged objects

# ### Physical surfaces

# We start by marking the inner circle

circle_inner = [idx for (dim, idx) in map_to_input[1] if dim == 2]

# Next, we known that the outer circle is the only remaining object of dimension 2

circle_outer = [idx for (dim, idx) in map_to_input[0] if dim == 2 and idx not in circle_inner]

# + tags=["remove-input"]
print(f"{circle_inner=}")
print(f"{circle_outer=}")
# -

# We create a unique physical group for each surface

# + tags=["remove-output"]
gmsh.model.addPhysicalGroup(2, circle_inner, tag=3)
gmsh.model.addPhysicalGroup(2, circle_outer, tag=7)
# -

# ### Physical curves

# Next, we would like to mark the external boundary, such that we can apply boundary conditions to it.
# We also want to mark the interface between the two domains
#
# We start by getting the boundary of each of the surfaces.

# ```{admonition} Input to getBoundary
# Note that the input format to `getBoundary` is a list of tuples, similar to the output of `fragment`.
# However, as we have split the groups into to lists of just indices, we need to convert this back
# to this format.
# `recursive` is a boolean. If `True` would return the entities of lowest dimension (0), instead
# of the dimension one lower than the input entities.
# `oriented` will multiply the object index with `-1` if this gives a consistent orientation of the boundary.
# ```

inner_boundary = gmsh.model.getBoundary([(2, e) for e in circle_inner], recursive=False, oriented=False)
outer_boundary = gmsh.model.getBoundary([(2, e) for e in circle_outer], recursive=False, oriented=False)

# We note that the outer circle now has a "donut" shape, i.e. it has two boundaries

# + tags=["remove-input"]
print(f"{inner_boundary=}")
print(f"{outer_boundary=}")
# -

# We remove this boundary from the external boundaries before creating a physical group

interface = [idx for (dim, idx) in inner_boundary if dim == 1]
ext_boundary = [idx for (dim, idx) in outer_boundary if idx not in interface and dim == 1]

# + tags=["remove-output"]
gmsh.model.addPhysicalGroup(1, interface, tag=12)
gmsh.model.addPhysicalGroup(1, ext_boundary, tag=15)
# -

# Next, we can generate the mesh with a given minimal resolution

gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.2)
gmsh.model.mesh.generate(2)

# In GMSH we can choose to mesh first, second or third order meshes. We choose a 3rd order mesh for the circle.

gmsh.model.mesh.setOrder(3)

# The mesh is generated, but not yet saved to disk. We could save it to disk using `gmsh.write` command, or import it
# directly into DOLFINx.
# We choose the latter, and use {py:func}`dolfinx.io.gmsh.model_to_mesh` to convert the GMSH model to a DOLFINx mesh.
# For this function we need to send in a few objects:
# 1. The `gmsh.model` instance
# 2. The `MPI` communicator we want to distribute the mesh over
# 3. The `MPI` rank that holds the mesh
#    ```{admonition} GMSH and MPI
#    GMSH does not have a concept of an MPI distributed mesh, and will generate a mesh on each process if used as above.
#    Usually, one would call `gmsh.initialize()` on all processors, then add an `if-else` clause that only generates
#    the mesh on a single processor (usually rank 0).
#    ```
# 4. The geometrical dimension of the mesh when used in DOLFINx
#    ```{admonition} Geometrical dimension of a mesh from GMSH
#    GMSH always generates three-dimensional points, which means that a 1D or 2D mesh could be interpreted as a manifold in
#    a higher dimensional space. The `gdim` argument is used to specify the dimension of the mesh when used in DOLFINx.
#    ```
#
# With these inputs we can generate the mesh and the cell and facet markers from the {py:class}`dolfinx.io.gmsh.MeshData` object.

mesh_data = dolfinx.io.gmsh.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
circular_mesh = mesh_data.mesh
cell_marker = mesh_data.cell_tags
facet_marker = mesh_data.facet_tags

# We can now finalize GMSH (as we will not use it further in this section), and inspect the `cell_marker` and `facet_marker`.

gmsh.finalize()

# (mesh_generation:tags)=
# ## The DOLFINx Meshtags object
# A {py:class}`dolfinx.mesh.MeshTags` object is a collection of entities (peaks, ridges, facets or cells) in
# a mesh that have been tagged with some markers.
# The {py:class}`MeshTags<dolfinx.mesh.MeshTags>` object is used to store information about the mesh,
# such as boundary conditions or material markers.
# We can access the dimension of the tag by calling {py:attr}`MeshTags.dim<dolfinx.mesh.MeshTags.dim>`.
# To get a list of the entities that have been tagged, we can call {py:attr}`MeshTags.indices<dolfinx.mesh.MeshTags.indices>`.
# To get the corresponding marker to each entity in `indices`, we can call {py:attr}`MeshTags.values<dolfinx.mesh.MeshTags.values>`.

# + tags=["remove-input"]
print(f"{cell_marker.dim=}")
print(f"{cell_marker.values=}")
print(f"{facet_marker.dim=}")
print(f"{facet_marker.indices=}")
print(f"{facet_marker.values=}")
# -

# We can also call {py:attr}`MeshTags.find(value)<dolfinx.mesh.MeshTags.find>`
# to get a list of all indices (local to process) that is marked with the given `value`

# + tags=["remove-input"]
print(f"{cell_marker.find(3)=}")
# -

# We can now plot the mesh with the `plot_mesh` function from the previous section.

# + tags=["hide-input"]
import pyvista


def plot_mesh(mesh: dolfinx.mesh.Mesh, values=None):
    """
    Given a DOLFINx mesh, create a `pyvista.UnstructuredGrid`,
    and plot it and the mesh nodes
    """
    plotter = pyvista.Plotter()
    V_linear = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    linear_grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(V_linear))
    if mesh.geometry.cmap.degree > 1:
        ugrid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(mesh))
        if values is not None:
            ugrid.cell_data["Marker"] = values
        plotter.add_mesh(ugrid, style="points", color="b", point_size=10)
        ugrid = ugrid.tessellate()
        plotter.add_mesh(ugrid, show_edges=False)
        plotter.add_mesh(linear_grid, style="wireframe", color="black")

    else:
        if values is not None:
            linear_grid.cell_data["Marker"] = values
        plotter.add_mesh(linear_grid, show_edges=True)
    plotter.show_axes()
    plotter.view_xy()
    if not pyvista.OFF_SCREEN:
        plotter.show()


# -

plot_mesh(circular_mesh, cell_marker.values)
