# # Generating code for assembling {term}`tensor`s
# All the code in the previous section is Python code that is a symbolic representation the variational form of a projection.

# Creating code for each and every variational formulation is a tedious task, and it is error-prone.
# Therefore, in FEniCS, we exploit the symbolic representation of a variational form to generate code
# for assembling local element {term}`tensor`s.
# One can interpret the variational form written in UFL as a directed acyclic graph ({term}`DAG`) of operations,
# where each operation can be implemented in C.

# ## The FEniCSx Form Compiler ({term}`FFCx`)
# We use FFCx to generate the code used for finite element assembly.
#
# The previous section of the tutorial can be found in the file
# [ufl_formulation.py](https://github.com/jorgensd/fenics23-tutorial/tree/release/src/ufl_formulation.py)
# on the FEniCS@Sorbonne [GitHub repository](https://github.com/jorgensd/fenics23-tutorial/).
# We can use the FFCx main-module to generate C code for all the objects in this file:

import os
from pathlib import Path

import ffcx.main

cwd = Path.cwd()
infile = cwd / "ufl_formulation.py"

# + tags=["remove-output"]
ffcx.main.main(["-o", str(cwd), "--visualise", str(infile)])

# -
# What happens when we run the command above?
# The first thing that happens is that each line of code in the file is executed by the Python interpreter.
# Then, all variational formulations that have been given the name `a`, `L` or `J` is being extracted.
#
# ```{note}
#  If you only want to extract a specific form, or a form that is not named as above, create a list called `forms`
#  and add the form to this list.
# ```
#
# For each of this forms, FFCx uses UFL to convert the initial expression into a low level
# representation with the {term}`DAG`,
# and then generates C code for the assembly of the local element {term}`tensor`s.
# This computes the computational graph of the bi-linear and linear form
# ## Bilinear graph
# ![Bilinear graph](S_a.png)
# ## Linear graph
# ![Linear graph](S_L.png)

# The generated code can be found in the file `name_of_file.h` and `name_of_file.c` in the current working directory.

os.system("ls ufl_formulation.*")

# We can look at the assembly code for the local matrix. We start by inspecting the signature of the `tabulate_tensor`
# function that computes the local element matrix

os.system("head -102 ufl_formulation.c | tail +29")

# ## Optional Exercise
# Study the computational graph for the bi-linear form, and the kernel for the assembly of this form below.
# Do you understand each step?
