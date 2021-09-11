# SPDX-FileCopyrightText: © 2021 Antonio López Rivera <antonlopezr99@gmail.com>
# SPDX-License-Identifier: GPL-3.0-only


import unittest
import sympy as sp
import numpy as np

from transforms import Tx, Ty, Tz

from alexandria.math.units import rad

# Variables
a = sp.Symbol("alpha")
b = sp.Symbol("beta")
c = sp.Symbol("gamma")

d = rad(5)
e = rad(10)
f = rad(15)


class Tests(unittest.TestCase):

    def test_init_s(self):
        Ta = Tx(a)
        Tb = Ty(b)
        Tc = Tz(c)

    def test_init_n(self):
        Td = Tx(d)
        Te = Ty(e)
        Tf = Tz(f)

    def test_trans_mult(self):
        Ta = Tx(a)
        Tb = Ty(b)
        Tc = Tz(c)
        Td = Tx(d)
        Te = Ty(e)
        Tf = Tz(f)
        """
        Test
        """
        Tt  = Ta * Tb * Tc
        TT  = Td * Te * Tf
        TtT = Ta * Tf

    def test_get_params(self):
        Ta = Tx(a)
        Tb = Ty(b)
        Tc = Tz(c)
        Tt = Ta * Tb * Tc
        """
        Test
        """
        Tt.get_params()

    def test_inv(self):
        Ta = Tx(a)
        Tb = Ty(b)
        Tc = Tz(c)
        Tt  = Ta * Tb * Tc
        """
        Test
        """
        Tt_inv = Tt.Inv()
        print(Tt_inv)

    def test_trans_sum(self):
        Ta = Tx(a)
        Tb = Ty(b)
        Tc = Tz(c)
        Td = Tx(d)
        Te = Ty(e)
        Tf = Tz(f)
        """
        Test
        """
        Tt = Ta + Tb - Tc
        TT = Td + Te - Tf
        TtT = Ta + Tf

    def test_vector_mult(self):
        Ta = Tx(a)
        Tb = Ty(b)
        Tc = Tz(c)
        Tt = Ta * Tb * Tc
        """
        Test
        """
        # Position vector in c reference frame
        r = np.array([1, 1, 1])
        # Rotated vector wrt a reference frame
        r_tr = Tt * r
        # Sum of rotated vector (still a symbolic transform instance) and other vector
        r_rtr = r_tr + r

    def test_lambd(self):
        Ta = Tx(a)
        Tb = Ty(b)
        Tc = Tz(c)
        Tt = Ta * Tb * Tc
        # Position vector in c reference frame
        r = np.array([1, 1, 1])
        # Rotated vector wrt a reference frame
        r_tr = Tt * r
        """
        Test
        """
        # Get lambda function of rotated vector
        f = r_tr.get_lambda()
        # Rotated vector wrt a reference frame, by angles alpha, beta and gamma
        r_tr.suppress = False
        r_tr_num = r_tr(0.2, 0.3, np.pi / 2)

    def test_latex(self):
        Ta = Tx(a)
        Tb = Ty(b)
        Tc = Tz(c)
        Tt = Ta * Tb * Tc
        # Position vector in c reference frame
        r = np.array([1, 1, 1])
        # Rotated vector wrt a reference frame
        r_tr = Tt * r
        """
        Test
        """
        # Rotated vector: LaTeX output
        r_tr.to_latex()
