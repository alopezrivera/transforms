import sys
import numpy as np
import sympy as sp
from sympy.utilities.lambdify import lambdify

from Alexandria.general.console import print_color

from utilities import pretty_array, tuple_to_equal


class T:

    def __init__(self, delta=None, matrix=None, params=None, suppress=True):
        """
        :param delta: Rotation angle
        :param suppress: Suppress console output
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

        self.lambdification_info(self.params, *args)

        return lambdify(self.params, self.matrix)(*args)

    def __getitem__(self, item):
        return self.matrix.item(item)

    def __repr__(self):
        return sp.pretty(self.matrix) if self.is_symbolic else \
               pretty_array(self.matrix)

    def get_params(self):
        params = tuple(list(self.matrix.atoms(sp.Symbol)))
        print("Parameters" + params)
        return params

    def to_latex(self, var="T"):
        lx = r"\begin{equation}"                   + \
             f"\n   {var} = {sp.latex(self.matrix)}\n" + \
             r"\end{equation}"
        print_color(lx, "blue")
        return lx

    def lambdification_info(self, params, *args):
        print_color(f"\nLambdifying rotation matrix with parameters\n", "blue")
        for i in list(zip(params, [*args])):
            print(tuple_to_equal(str(i)))
        print("\n")

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
        """
        Transform addition, defined in the same way as transform multiplication
        """

        T_res = globals()[type(other).__name__]

        if isinstance(self.matrix, type(other.matrix)):
            matrix = self.matrix * other.matrix if self.is_symbolic else np.matmul(self.matrix, other.matrix)
            params = self.params + other.params
            return T_res(matrix=matrix, params=params)
        else:
            self.type_mismatch(other)
            matrix = self.matrix * other.matrix if self.is_symbolic else np.matmul(self.matrix, other.matrix)
            return T_res(matrix=matrix, params=self.params)

    def __sub__(self, other):
        """
        Transform subtraction, defined as the opposite of addition/multiplication
        """

        T_res = globals()[type(other).__name__]

        other.matrix = other.matrix.T

        if isinstance(self.matrix, type(other.matrix)):
            matrix = self.matrix * other.matrix if self.is_symbolic else np.matmul(self.matrix, other.matrix)
            params = self.params + other.params
            return T_res(matrix=matrix, params=params)
        else:
            self.type_mismatch(other)
            matrix = self.matrix * other.matrix if self.is_symbolic else np.matmul(self.matrix, other.matrix)
            return T_res(matrix=matrix, params=self.params)

    def __neg__(self):
        """
        Transform inverse
        """
        return globals()[type(self).__name__](matrix=self.matrix.T, params=self.params)


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



