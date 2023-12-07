
# # Code generation
# All the code in the previous section is Python code symbolically describing the variational form of the projection.

# This is why we in FEniCS rely on code generation.
# One can interpret the variational form written in UFL as a directed acyclic graph of operations,
# where we for each simple operation can implement it as C code.

# ## FFCx
# We use FFCx to generate the code used for finite element assembly.
#
# The previous section of the tutorial can be found in the file `introduction.py`.
# We can use the FFCx main module to generate C code for all the objects in this file

import ffcx.main
from pathlib import Path
import os
cwd = Path.cwd()
infile = cwd / "ufl_formulation.py"

ffcx.main.main(["-o", str(cwd), "--visualise", str(infile)])

# This computes the computational graph of the bi-linear and linear form
# ## Bilinear graph
# ![Bilinear graph](S_a.png)
# ## Linear graph
# ![Linear graph](S_L.png)

# With this graph, we also get compiled `c` code:

os.system("ls ufl_formulation.*")

# We can look at the assembly code for the local matrix. We start by inspecting the signature of the `tabulate_tensor` function,
# that computes the local element matrix

os.system("head -336 ufl_formulation.c | tail +283")
