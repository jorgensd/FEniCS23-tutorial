
# # Code generation
# All the code above is Python code symbolically describing the variational form of the projection.

# To do so, we would define each of the functions as the linear
# combinations of the basis functions
# $u=\sum_{i=0}^{\mathcal{N}}u_i\phi_i(x)\qquad
# v=\sum_{i=0}^{\mathcal{N}}v_i\phi_i(x)\qquad
# f=\sum_{k=0}^{\mathcal{M}}f_k\psi_k(x)\qquad
# g=\sum_{l=0}^{\mathcal{T}}g_l\varphi_l(x)$
#
# We next use the map $M_K:K_{ref}\mapsto K$
# \begin{align}
# \int_\Omega u v~\mathrm{d}x&= \sum_{K\in\mathcal{K}}\int_K u(x) v(x)~\mathrm{d}x\\
# &= \sum_{K\in\mathcal{K}}\int_{M_K(K_{ref})} u(x)v(x)~\mathrm{d}x\\
# &= \sum_{K\in\mathcal{K}}\int_{K_{ref}}u(M_K(\bar x))v(M_K(\bar x))\vert \mathrm{det} J_K(\bar x)\vert~\mathrm{d}\bar x
# \end{align}
# where $K$ is each element in the physical space, $J_K$ the Jacobian of the mapping.
# Next, we can insert the expansion of $u$ into the formulation and identify the matrix system $Au=b$, where
# \begin{align}
# A_{j, i} &= \int_{K_{ref}} \phi_i(M_K(\bar x))\phi_j(M_K(\bar x))\vert \mathrm{det} J_K(\bar x)\vert~\mathrm{d}\bar x\\
# b_j &= \int_{K_{ref}} \frac{\Big(\sum_{k=0}^{\mathcal{M}}f_k\psi_i(M_K(\bar x))\Big)}
# {\Big(\sum_{l=0}^{\mathcal{T}}g_k\varphi_i(M_K(\bar x))\Big)}\phi_j(M_K(\bar x))\vert \mathrm{det} J_K(\bar x)\vert~\mathrm{d}\bar x
# \end{align}
# Next, one can choose an appropriate quadrature rule with points and weights, include the
# correct mapping/restrictions of degrees of freedom for each cell.
# All of this becomes quite tedious and error prone work, and has to be repeated for every variational form!
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
cwd = Path.cwd()
infile = cwd / "introduction.py"

ffcx.main.main(["-o", str(cwd), "--visualise", str(infile)])

# This computes the computational graph of the bi-linear and linear form
# ## Bilinear graph
# ![Bilinear graph](S_a.png)
# ## Linear graph
# ![Linear graph](S_L.png)
