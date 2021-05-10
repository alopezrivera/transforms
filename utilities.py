import numpy as np
from math import ceil


def get_representative_decimals(n, precision=1/100):
    d = -np.log10(abs(n)*precision)
    return ceil(d) if d > 0 else 0


def pretty_array(a):
    return np.array2string(a, precision=get_representative_decimals(np.min(a[np.nonzero(a)])), suppress_small=True)


def tuple_to_equal(a):
    chars = r"()"
    for c in chars:
        if c in a:
            a = a.replace(c, "")
    return a.replace(", ", " = ")
