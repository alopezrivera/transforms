# SPDX-FileCopyrightText: © 2021 Antonio López Rivera <antonlopezr99@gmail.com>
# SPDX-License-Identifier: GPL-3.0-only

import numpy as np
import sympy as sp

from transforms import Tx, Ty, Tz

# Variables
a = sp.Symbol("alpha")
b = sp.Symbol("beta")
c = sp.Symbol("gamma")

# Transforms
Ta = Tx(a)
Tb = Ty(b)
Tc = Tz(c)
Tt = Ta*Tb*Tc
TT = Ta+Tb+Tc

# Position vector in c reference frame
r = np.array([1, 1, 1])

# Rotated vector wrt a reference frame
r_tr = Tt*r

# Lambdify
r_tr_num = r_tr(0.2, 0.3, np.pi/2)

# To LaTeX
r_tr.latex()
