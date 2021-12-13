# Transforms

### Python coordinate frame transform library

![alt text](tests/coverage/coverage.svg ".coverage available in tests/coverage/")

Library to ease work with 3D coordinate frame transformations, by two means:
- Easy-to-operate-with symbolic/numerical linear transformation classes
- LaTeX export and enhanced console printing of transformation matrices

[API reference.](https://alopezrivera-docs.github.io/transforms/)

---

[ 1. Install ](#1-install)

[ 2. Usage and Syntax ](#2-usage-and-syntax)

[ 3. To-do ](#3-to-do)

## 1. Install

`pip install linear_transforms`

## 2. Usage and Syntax

To use the library, import it from Python with

    from transforms import Tx, Ty, Tz

The code for the following examples can be found in `demo/`. Further examples can be found in `tests/`.

#### 2.1 Creating a Linear Transformation

The three standard transformations, `Tx`, `Ty` and `Tz`, describe the **original** frame of reference of an object which undergoes a rotation from its **body reference frame**, for **right-handed coordinate systems**. 

A transformation can be created in two ways:

- Using a SymPy symbol for the rotation angle, which greatly facilitates inspecting transform combination matrices (further discussion in [ Section 2.4 ](#24-inspecting-matrices)):

        import sympy as sp
        a = sp.Symbol("alpha")

        Ta = Tx(a)

- Using a numerical variable for the rotation angle:
    
        import numpy as np
        a = np.pi/5

        Ta = Tx(a)

#### 2.2 Lambdifying a symbolic linear transformation

Turning a symbolic linear transformation to a numerical one can be done by calling the transformation itself.

    T_num = Ta(<VALUE>)

#### 2.3 Operating with Linear Transformations

Transform concatenation is defined with the multiplication sign, as well as the addition sign. The multiplication notation is recommended.

    Tt = Ta*Tb*Tc
    TT = Ta+Tb+Tc
    
    Tt == TT
    
Transform multiplication with NumPy or SymPy arrays is defined with the multiplication sign alone.

    r = np.array([1, 1, 1])
    
    r_tr = Tt*r
    
#### 2.4 Inspecting matrices

##### _LaTeX_

Transformations, symbolic and numerical, can be outputed to LaTeX with the function call below. The latex equation will be visible in the terminal in light blue.

    r_tr.to_latex()

##### _Printing_

Printing a transformation will output an ASCII representation in the terminal

    print(r_tr)

---
[Back to top](#transforms)
