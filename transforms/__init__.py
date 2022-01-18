import sympy as sp

from transforms.transforms import M, R
from transforms.transforms import Rx as _Rx, \
                                  Ry as _Ry, \
                                  Rz as _Rz,\
                                  Tx as _Tx, \
                                  Ty as _Ty, \
                                  Tz as _Tz


def _M(f, delta):

    M = globals()[f'_{f}']

    if 'sympy' in str(type(delta)):
        return M(delta)
    else:
        return M(sp.Symbol('aux'))(delta)


def Rx(delta):
    return _M('Rx', delta)


def Ry(delta):
    return _M('Ry', delta)


def Rz(delta):
    return _M('Rz', delta)


def Tx(delta):
    return _M('Tx', delta)


def Ty(delta):
    return _M('Ty', delta)


def Tz(delta):
    return _M('Tz', delta)
