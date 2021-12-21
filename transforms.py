# SPDX-FileCopyrightText: © 2021 Antonio López Rivera <antonlopezr99@gmail.com>
# SPDX-License-Identifier: GPL-3.0-only

"""
Transforms
----------
"""


import numpy as np
import sympy as sp

from sympy.utilities.lambdify import lambdify

from alexandria.shell import print_color


class M:
    """
    General matrix class
    --------------------
    """
    def __init__(self, matrix, params):
        """
        :param matrix: Rotation matrix
        :param params: Set of parameters
        """

        self.matrix = matrix
        self.params = params

    def __getitem__(self, item):
        """
        Retrieve
        """
        return self.matrix[item]

    def __getattr__(self, item):
        """
        Pass unknown attributes to matrix class
        """
        if item in self.__dict__.keys():
            return self.item
        else:
            try:
                self.matrix.__dict__[item]
            except KeyError:
                raise AttributeError(f"{self.matrix.__class__} has no attribute <{item}>")

    """
    Properties
    """
    def T(self):
        """
        Rotation matrix Transpose
        """
        return M(matrix=self.matrix.T, params=self.params)

    def I(self):
        """
        Rotation matrix Inverse
        -----------------------

        COUNTERROTATION of that described by the
        inverted rotation matrix.
        - Inverse matrix
        - Negative angles
        """
        # Transpose
        inv_matrix = self.matrix.T
        # Negative angles
        for param in self.params:
            inv_matrix = inv_matrix.subs(param, f'-{param}')
        return M(matrix=inv_matrix, params=self.params)

    def diff(self):
        dm = sp.diff(self.matrix, self.params)
        return M(matrix=dm, params=self.params | dm.free_symbols)

    """
    Operators
    """
    def _operate(self, other, f):

        # Another rotation matrix
        if isinstance(other, M):
            return M(matrix=f(self.matrix, other.matrix), params=self.params | other.params)
        # A NumPy array
        elif isinstance(other, np.ndarray):
            return M(matrix=f(self.matrix, other), params=self.params)
        # A SymPy array
        elif 'sympy' in other.__module__:
            return M(matrix=f(self.matrix, other), params=self.params | other.free_symbols)
        else:
            raise ValueError("Multiplication input invalid. It should be either a rotation, or a NumPy or SymPy array.")

    def __mul__(self, other):
        return self._operate(other, lambda m1, m2: m1 * m2)

    def __add__(self, other):
        return self._operate(other, lambda m1, m2: m1 + m2)

    """
    Export
    """
    def __call__(self, *args, **kwargs):
        """
        Retrieve numerical rotation matrix
        ----------------------------------

        :param args:
            1. Single parameter
            2. List of matrix parameters
            3. List of NumPy arrays
                -> 1 array per parameter
        """

        f = lambdify(list(self.params), self.matrix)

        # Check arguments
        assert not (args and kwargs), \
            "Combination of non-keyword and keyword arguments not supported"

        arguments = []
        if args:
            arguments += args
        if kwargs:
            arguments += list(kwargs.values())

        assert len(arguments) == len(self.params), \
            f"Argument number mismatch. Arguments in the matrix:\n\n   {', '.join(str(p) for p in self.params)}"

        # Return functions
        def arg_rec_return(argument):

            # Input type
            a_tuple  = isinstance(argument, tuple)
            an_array = isinstance(argument, np.ndarray)

            # Return
            if a_tuple:
                if isinstance(argument[0], np.ndarray):
                    return [arg_rec_return(v) for v in np.vstack(argument).T]
                else:
                    return f(*argument)
            elif an_array:
                return [f(v) for v in argument]
            else:
                return f(*argument)

        def kwarg_rec_return(dictionary):

            # Argument type
            arrays = isinstance(list(dictionary.values())[0], np.ndarray)

            # Return
            if arrays:

                array_list = list(dictionary.values())
                array_size = array_list[0].size
                assert [a.size == array_size for a in array_list], \
                    "Input arrays do not have the same internal size."

                keys   = list(dictionary.keys())
                values = np.vstack(list(dictionary.values())).T

                return [f(**dict(zip(keys, list(values[i])))) for i in range(array_size)]

            else:
                return f(**kwargs)

        # Out
        if args:
            return arg_rec_return(args)
        if kwargs:
            return kwarg_rec_return(kwargs)

    """
    Representation
    """
    def __repr__(self):
        return sp.pretty(self.matrix)

    def latex(self, var="T"):
        lx = "\n"\
             r"\begin{equation}"                   + \
             f"\n   {var} = {sp.latex(self.matrix)}\n" + \
             r"\end{equation}"
        print_color(lx, "blue")
        return lx


class T(M):
    """
    Transformation Matrix
    ---------------------

    By convention, a transformation matrix maps the coordinates of
    objects in an INERTIAL REFERENCE FRAME to their coordinates in
    a ROTATING one.
    """
    def __init__(self, delta):

        super(T, self).__init__(self._matrix(delta), {delta})

        # Transform transpose
        self.T = self.T()


class Tx(T):
    """
    Transformation: X axis
    ----------------------
    """

    def _matrix(self, delta):
        return sp.Matrix([[1,  0,              0],
                          [0,  sp.cos(delta),  sp.sin(delta)],
                          [0, -sp.sin(delta),  sp.cos(delta)]])


class Ty(T):
    """
    Transformation: Y axis
    ----------------------
    """
    def _matrix(self, delta):
        return sp.Matrix([[sp.cos(delta), 0, -sp.sin(delta)],
                          [0,             1,  0],
                          [sp.sin(delta), 0,  sp.cos(delta)]])


class Tz(T):
    """
    Transformation: Z axis
    ----------------------
    """
    def _matrix(self, delta):
        return sp.Matrix([[sp.cos(delta),  sp.sin(delta), 0],
                          [-sp.sin(delta), sp.cos(delta), 0],
                          [0,              0,             1]])
