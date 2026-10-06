from __future__ import annotations

from typing import Dict
from .spherical_harmonics import YL, gam
import numpy as np

# from gpaw.typing import Array3D

_gaunt: Dict[int, np.ndarray] = {}
_nabla: Dict[int, np.ndarray] = {}
_gaunt_pair: Dict[tuple, np.ndarray] = {}


def gaunt(lmax: int = 2): # -> Array3D:
    r"""Gaunt coefficients

    :::

         ^      ^     -- L      ^
      Y (r)  Y (r) =  > G    Y (r)
       L      L       -- L L  L
        1      2      L   1 2
    """

    if lmax in _gaunt:
        return _gaunt[lmax]

    # Lmax = (lmax + 1)**2
    # L2max = (2 * lmax + 1)**2
    Lmax = (2 * lmax + 1)**2
    L2max = (lmax + 1)**2

    G_LLL = np.zeros((Lmax, L2max, L2max))
    for L1 in range(Lmax):
        for L2 in range(L2max):
            for L in range(L2max):
                r = 0.0
                for c1, n1 in YL[L1]:
                    for c2, n2 in YL[L2]:
                        for c, n in YL[L]:
                            nx = n1[0] + n2[0] + n[0]
                            ny = n1[1] + n2[1] + n[1]
                            nz = n1[2] + n2[2] + n[2]
                            r += c * c1 * c2 * gam(nx, ny, nz)
                G_LLL[L1, L2, L] = r
    _gaunt[lmax] = G_LLL
    return G_LLL


def gaunt_pair(l1: int, l2: int):
    key = (l1, l2)
    if key in _gaunt_pair:
        return _gaunt_pair[key]

    lmax = l1 + l2
    if (lmax + 1)**2 > len(YL):
        raise ValueError(
            f'Gaunt table supports output harmonics only through '
            f'l={int(np.sqrt(len(YL))) - 1}, but l1={l1}, l2={l2} '
            f'require l={lmax}.'
        )

    G_Lmm = np.zeros(((lmax + 1)**2, 2*l1 + 1, 2*l2 + 1))
    for L in range((lmax + 1)**2):
        for im1, L1 in enumerate(range(l1**2, (l1 + 1)**2)):
            for im2, L2 in enumerate(range(l2**2, (l2 + 1)**2)):
                r = 0.0
                for c, n in YL[L]:
                    for c1, n1 in YL[L1]:
                        for c2, n2 in YL[L2]:
                            nx = n[0] + n1[0] + n2[0]
                            ny = n[1] + n1[1] + n2[1]
                            nz = n[2] + n1[2] + n2[2]
                            r += c * c1 * c2 * gam(nx, ny, nz)
                G_Lmm[L, im1, im2] = r
    _gaunt_pair[key] = G_Lmm
    return G_Lmm


def nabla(lmax: int = 2): # -> Array3D:
    """Create the array of derivative intergrals.

    :::

      /  ^    ^   1-l' d   l'    ^
      | dr Y (r) r     --[r  Y  (r)]
      /     L           _     L'
                       dr
    """

    if lmax in _nabla:
        return _nabla[lmax]

    Lmax = (lmax + 1)**2
    Y_LLv = np.zeros((Lmax, Lmax, 3))
    # Insert new values
    for L1 in range(Lmax):
        for L2 in range(Lmax):
            for v in range(3):
                r = 0.0
                for c1, n1 in YL[L1]:
                    for c2, n2 in YL[L2]:
                        n = [0, 0, 0]
                        n[0] = n1[0] + n2[0]
                        n[1] = n1[1] + n2[1]
                        n[2] = n1[2] + n2[2]
                        if n2[v] > 0:
                            # apply derivative
                            n[v] -= 1
                            # add integral
                            r += n2[v] * c1 * c2 * gam(n[0], n[1], n[2])
                Y_LLv[L1, L2, v] = r
    _nabla[lmax] = Y_LLv
    return Y_LLv
