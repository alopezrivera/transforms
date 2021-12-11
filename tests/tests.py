# SPDX-FileCopyrightText: © 2021 Antonio López Rivera <antonlopezr99@gmail.com>
# SPDX-License-Identifier: GPL-3.0-only

import unittest

import sympy as sp
import numpy as np

from alexandria.math.units import rad

from transforms import Tx, Ty, Tz, R


a = sp.Symbol('alpha')
b = sp.Symbol('beta')
g = sp.Symbol('gamma')


class Tests(unittest.TestCase):

    def test1(self):
        t = Tx(a)
        assert isinstance(t(20/180*np.pi), np.ndarray)

    def test2(self):
        t = Tx(a)(20/180*np.pi)
        assert isinstance(t*2, np.ndarray)

    def test3(self):
        t = Tx(a)(20/180*np.pi)
        for i in range(t.shape[0]):
            for j in range(t.shape[1]):
                assert (t*t)[i, j] == (t**2)[i, j]
        assert isinstance(t*6, np.ndarray)

    def test4(self):
        t1 = Tx(a)
        t2 = Ty(b)
        t3 = t1*t2
        assert isinstance(t3(rad(10), rad(15)), np.ndarray)

    def test5(self):
        alphas = np.array([rad(10), rad(20), rad(30)])
        betas  = np.array([rad(30), rad(40), rad(50)])

        t1 = Tx(a)
        t2 = Ty(b)
        t3 = t1 * t2

        assert all([isinstance(matrix, np.ndarray) for matrix in t3(alpha=alphas, beta=betas)])

    def test6(self):

        t1 = Tx(a)
        t2 = Ty(b)
        t3 = t1 * t2

        vnp = np.array([1, 2, 3]).reshape((3, 1))

        vsp = sp.Matrix([a, b, g])

        assert isinstance(t3*vnp, R)
        assert isinstance(t3*vsp, R)
        assert (t3*vsp).params == t3.params | vsp.free_symbols
        assert isinstance((t3*vsp)(rad(10), rad(20), rad(30)), np.ndarray)

