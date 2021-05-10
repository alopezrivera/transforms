# Transforms: Python coordinate frame transform library

Library to ease work with 3D coordinate frame transformations, by two means:
- Easy-to-operate-with symbolic/numerical linear transformation classes
- LaTeX export and enhanced console printing of transformation matrices

Nice experiment to learn about operator overloading.

`Antonio Lopez Rivera, 2020`

---

[ 1. Install ](#1-install)

[ 2. Usage and Syntax ](#2-usage-and-syntax)

[ 3. To-do ](#3-to-do)

## 1. Install

1. Place `transforms.py` and `utilities.py` in your root directory (or another, but mind the import)
2. `from transforms import Tx, Ty, Tz`

## 2. Usage and Syntax

All code available in `demo.py`.

### `2.1 Creating a Linear Transformation`

Transformation describing the position of a rotated object (by angle `a`) from its original frame of reference:

    Ta = Tx(a)
    
The linear transformation class may be initialized with a `Sympy.Symbol`, or regular values for the rotation angle

### `2.2 Lambdifying a symbolic linear transformation`

Turning a symbolic linear transformation to a numerical one can be done by calling the transformation itself.

    T_num = Ta(<VALUE>)

### `2.3 Operating with Linear Transformations`

Transform concatenation is defined with the multiplication sign, as well as the addition sign. The multiplication notation is recommended.

    Tt = Ta*Tb*Tc
    TT = Ta+Tb+Tc
    
    Tt == TT
    
Transform multiplication with NumPy or SymPy arrays is defined with the multiplication sign alone.

    r = np.array([1, 1, 1])
    
    r_tr = Tt*r
    
### `2.4 Inspecting matrices`

#### _LaTeX_

Transformations, symbolic and numerical, can be outputed to LaTeX with the function call below. The latex equation will be visible in the terminal in light blue.

    r_tr.to_latex()

#### _Printing_

Printing a transformation will output an ASCII representation in the terminal

    print(r_tr)

## 3. To-do

1. n-dimensional transform matrix generation
