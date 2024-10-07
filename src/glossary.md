# Glossary

```{glossary}
UFL
    The Unified Form Language. Documentation at: https://docs.fenicsproject.org/ufl/main/

FFCx
    The FEniCSx Form Compiler. Documentation at: https://docs.fenicsproject.org/ffcx/main/

DOLFINx
    The user-interface of the FEniCS project. Documentation at: https://docs.fenicsproject.org/dolfinx/main/

FEM
    The Finite Element Method

FE
    A finite element. A encyclopedia can be found at: https://defelement.com/

Tabulation
    Filling a n-th order tensor with data. Term often used when computing basis values and derivatives at a set of points.

Assemble
    Integrate a compiled DOLFINx form into a n-th order (global) {term}`tensor`.

PDE
    Partial differential equation

Basix
    The FEniCS project finite element tabulator. Documentation at: https://docs.fenicsproject.org/basix/main/

Tensor
    A n-dimensional array. A 0-dimensional tensor is a scalar value. A 1-dimensional tensor is a vector and a 2-dimensional tensor
    is a matrix. These are the most common tensors used in {term}`assemble`.

DAG
    A graph consisting of vertices and edges with each edge directed from one vertex to another, where following those directions will never form a closed loop. See: https://en.wikipedia.org/wiki/Directed_acyclic_graph

MPI
    Message Passing Interface. A standard for passing messages in a distributed memory environment. `MPI.COMM_WORLD` is the most common communicator, which will use how many processors the user specify with `mpirun -n M` or `mpiexec -n M`, where `M` are the number
    of processors to distribute data over.

PETSc
    PETSc, the Portable, Extensible Toolkit for Scientific Computation. See: https://petsc.org/release/ for more information.

MUMPS
    MUltifrontal Massively Parallel sparse direct Solver. See: https://mumps-solver.org/index.php for more information

GMSH
    Open source meshing software. See: https://gmsh.info/ for more information

OCC
    The Open Cascade project. An open source 3D geometry library.
    See: https://dev.opencascade.org/ for more information.
```
