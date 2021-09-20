# SPDX-FileCopyrightText: © 2021 Antonio López Rivera <antonlopezr99@gmail.com>
# SPDX-License-Identifier: GPL-3.0-only

"""
Transforms
----------
"""


import sys
import numpy as np
import sympy as sp
from sympy.utilities.lambdify import lambdify

from alexandria.shell import print_color
from alexandria.data_structs.array import pretty_array
from alexandria.data_structs.string import tuple_to_equal


class T:

    def __init__(self, delta=None, matrix=None, params=None, suppress=True):
        """
        Linear transformation
            Describes coordinates in the **NON-ROTATED REFERENCE FRAME** from the **BODY REFERENCE FRAME**

        :param delta:       Rotation angle
        :param suppress:    Suppress console output (True by default)
        """
        if not isinstance(delta, type(None)):
            # Angle
            self.delta = delta
            self.is_symbolic = isinstance(delta, sp.core.symbol.Symbol)
            # Parameter vector for lambdification
            self.params = [delta]
            # Rotation matrix creation
            self.matrix = self.symbolic() if self.is_symbolic else self.numerical()
        elif not isinstance(matrix, type(None)) and not(isinstance(params, type(None))):
            # Angle
            self.is_symbolic = any(isinstance(angle, sp.core.symbol.Symbol) for angle in params)
            # Parameter vector for lambdification
            self.params = params
            # Initialize matrix
            self.matrix = matrix
        else:
            sys.exit("Attempted to initialize transformation without providing angle or matrix and params")

        # Output suppression
        self.suppress = suppress

    def T(self):
        """
        Transform transpose
            Describes coordinates in the **BODY REFERENCE FRAME** from the **NON-ROTATING REFERENCE FRAME**.
        :return:
        """
        return globals()[type(self).__name__](matrix=self.matrix.T, params=self.params)

    def Inv(self):
        """
        Transform inverse
            Describes coordinates in the **FINAL REFERENCE FRAME** after **N LINEAR TRANSFORMATIONS** from the original
            reference frame.
        """
        # Transpose
        inv_matrix = self.matrix.T
        # Negative angles
        for param in self.params:
            inv_matrix.subs(param, -param)
        return globals()[type(self).__name__](matrix=inv_matrix, params=self.params)

    def __mul__(self, other):
        """
        Multiplication between transforms, as well as between transforms and NumPy/SymPy arrays
        """

        T_res = globals()[type(self).__name__]

        if type(other).__bases__ == type(self).__bases__:
            # Transform-Transform multiplication
            #       If self and other are instances of classes with the same parent class <class 'transform.T'>
            if isinstance(self.matrix, type(other.matrix)):
                # Both matrices are either SymPy matrices or NumPy arrays
                matrix = self.matrix * other.matrix if self.is_symbolic else np.matmul(self.matrix, other.matrix)
                params = self.params + other.params
                return T_res(matrix=matrix, params=params)
            else:
                # Type mismatch: surely 1 will be symbolic, and 1 will not
                self.type_mismatch(other)
                matrix = sp.Matrix(np.matmul(self.matrix, other.matrix))
                return T_res(matrix=matrix, params=self.params)
        else:
            # Transform-NumPy/SymPy array multiplication
            matrix = sp.Matrix(np.matmul(self.matrix, other)) if self.is_symbolic else np.matmul(self.matrix, other)
            return T_res(matrix=matrix, params=self.params)

    def __call__(self, *args, **kwargs):
        """
        Transform call

        When an array of parameters of size n is passed, it will return:
            - A number n of square if the transform instance contains a square transform matrix
            - An array of length 3 with each entry an array of length n, representing x, y and z coordinates, if the
              transform instance contains a transformed vector.

        - If suppress is False, the parameters to be substituted will be printed together with their
          assigned numerical values

        :param args: Values of the symbolic parameters.
        :param kwargs:
        :return: Symbolic transform matrix or transformed vector, with all symbolic parameters replaced by arguments.
        """

        self.lambdification_info(self.params, *args)

        f = lambdify(self.params, self.matrix)
        r = f(*args)

        if self.matrix.shape == (3, 3):
            np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
            return np.array([r[0][0] if np.array(r[0][0]).squeeze().size != 1 else np.ones(len(*args))*r[0][0],
                             r[1][0] if np.array(r[1][0]).squeeze().size != 1 else np.ones(len(*args))*r[1][0],
                             r[2][0] if np.array(r[2][0]).squeeze().size != 1 else np.ones(len(*args))*r[2][0]]
                            ).squeeze()
        else:
            return r

    def __getitem__(self, item):
        return self.matrix.item(item)

    def __repr__(self):
        return sp.pretty(self.matrix) if self.is_symbolic else \
               pretty_array(self.matrix)

    def get_lambda(self):
        """
        Transform lambdification

        When an array of parameters of size n is passed to the lambdified transform, it will return:
            - A number n of square if the transform instance contains a square transform matrix
            - An array of length 3 with each entry an array of length n, representing x, y and z coordinates, if the
              transform instance contains a transformed vector.

        :return: Symbolic transform matrix or transformed vector lambda function. When called, all symbolic parameters
                 are replaced by the provided arguments.
        """

        f = lambdify(self.params, self.matrix)

        if self.matrix.shape == (3, 3):
            return f
        else:
            np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
            g = lambda x: np.array([f(x)[0][0] if np.array(f(x)[0][0]).squeeze().size != 1 else np.ones(np.array(x).squeeze().size)*f(x)[0][0],
                                    f(x)[1][0] if np.array(f(x)[1][0]).squeeze().size != 1 else np.ones(np.array(x).squeeze().size)*f(x)[1][0],
                                    f(x)[2][0] if np.array(f(x)[2][0]).squeeze().size != 1 else np.ones(np.array(x).squeeze().size)*f(x)[2][0]],
                                   ).squeeze()
            return g

    def get_params(self):
        params = tuple(list(self.matrix.atoms(sp.Symbol)))
        print("Parameters: " + str(params))
        return params

    def to_latex(self, var="T"):
        lx = "\n"\
             r"\begin{equation}"                   + \
             f"\n   {var} = {sp.latex(self.matrix)}\n" + \
             r"\end{equation}"
        print_color(lx, "blue")
        return lx

    def lambdification_info(self, params, *args):
        print_color(f"\nLambdifying rotation matrix with parameters\n", "blue")
        print("===========================================")
        for i in list(zip(params, [*args])):
            print(tuple_to_equal(str(i)))
        print("===========================================")

    def type_mismatch(self, other):
        if not self.suppress:
            print_color(f"===================================================", "red")
            print_color(f"Warning: Type mismatch between multiplied matrices:", "red")
            print_color(f"{type(self.matrix)}", "blue")
            print(f"{self.__repr__()}")
            print_color(f"{type(other.matrix)}", "blue")
            print(f"{other.matrix}")
            print_color(f"Proceeding with matrix multiplication", "red")
            print_color(f"===================================================", "red")

    """
    Operator overloading: +, -
    """
    def __add__(self, other):
        return self.fund_op(other, 1)

    def __sub__(self, other):
        return self.fund_op(other, -1)

    def fund_op(self, other, sign):
        """
        Transform addition, defined in the same way as transform multiplication
        """

        if type(other).__bases__ == type(self).__bases__:

            T_res = globals()[type(other).__name__]

            if self.matrix.shape == other.matrix.shape:
                matrix = self.matrix * other.matrix if self.is_symbolic else np.matmul(self.matrix, other.matrix)
                params = self.params + other.params
                return T_res(matrix=matrix, params=params)
            else:
                matrix = self.matrix + sign*other.matrix
                params = self.params + other.params
                return T_res(matrix=matrix, params=params)
        else:

            self.type_mismatch(other)
            T_res = globals()[type(self).__name__]

            if self.matrix.shape == other.shape:
                matrix = self.matrix * other if self.is_symbolic else np.matmul(np.squeeze(self.matrix), np.squeeze(other))
                return T_res(matrix=matrix, params=self.params)
            else:
                matrix = self.matrix + sign*sp.Matrix(other) if self.is_symbolic else self.matrix + other
                return T_res(matrix=matrix, params=self.params)


class Tx(T):

    def numerical(self):
        return np.array([[1, 0, 0],
                         [0, np.cos(self.delta), np.sin(self.delta)],
                         [0, -np.sin(self.delta), np.cos(self.delta)]])

    def symbolic(self):
        return sp.Matrix([[1,  0,                   0],
                         [0,  sp.cos(self.delta),  sp.sin(self.delta)],
                         [0, -sp.sin(self.delta),  sp.cos(self.delta)]])


class Ty(T):

    def numerical(self):
        return np.array([[np.cos(self.delta), 0, -np.sin(self.delta)],
                         [0,                  1, 1],
                         [np.sin(self.delta), 0, np.cos(self.delta)]])

    def symbolic(self):
        return sp.Matrix([[sp.cos(self.delta), 0, -sp.sin(self.delta)],
                          [0,                  1, 0],
                          [sp.sin(self.delta), 0, sp.cos(self.delta)]])


class Tz(T):

    def numerical(self):
        return np.array([[np.cos(self.delta),  np.sin(self.delta), 0],
                         [-np.sin(self.delta), np.cos(self.delta), 0],
                         [0,                  0,                  1]])

    def symbolic(self):
        return sp.Matrix([[sp.cos(self.delta),  sp.sin(self.delta), 0],
                          [-sp.sin(self.delta), sp.cos(self.delta), 0],
                          [0,                  0,                  1]])
