import numpy as np
import scipy.special as sp
from math import factorial, sqrt, pi
from .from_gpaw.spherical_harmonics import Y

'''
Real spherical harmonics follow the definition:
    https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#Real_spherical_harmonics
Work functions for generating spherical harmonics are adapted from GPAW
'''
    
def spharm(r, l):
    rshape = r.shape
    r = r.reshape((-1, 3))
    _, x, y, z = r_to_xyz(r)
    spharm = spharm_xyz(l, x, y, z)
    return spharm.reshape(rshape[:-1]+(2*l+1,))
    
def r_to_xyz(r):
    '''
    Returns:
        rnorm, x, y, z: x, y, z are normalized
    '''
    eps = 1e-8
    rnorm = np.linalg.norm(r, axis=-1)
    x = r[..., 0] / (rnorm + eps/10)
    y = r[..., 1] / (rnorm + eps/10)
    z = r[..., 2] / (rnorm + eps/10)
    
    # r=0
    r_is_zero = rnorm < eps
    x[r_is_zero] = 0.0
    y[r_is_zero] = 0.0
    z[r_is_zero] = 1.0
    
    return rnorm, x, y, z

def spharm_xyz(l, x, y, z):
    # x, y, z must be normalized
    assert x.shape[0] == y.shape[0] == z.shape[0]
    assert len(x.shape) == len(y.shape) == len(z.shape) == 1
    spharm = np.zeros((x.shape[0], 2*l+1), dtype='f8')
    for m in range(-l, l+1):
        spharm[:, m+l] = Y(l**2+m+l, x, y, z)
    return spharm

def _C_lm(l, m):
    """
    the real normalization factor of Y_lm in SIESTA 
    when m != 0, multiplied by sqrt(2)
    when m is negative, the same as positive
    """
    a = (2*l + 1) / (4.0 * pi)
    b = factorial(l - abs(m)) / factorial(l + abs(m))
    c = sqrt(a * b)
    if m != 0:
        c *= sqrt(2.0)
    return c

def rly_grly_single(l, r, x, y, z, tiny=1e-14):
    """
    Calculate rly and grly of single point (X, Y, Z) (the same as spher_harm.f of SIESTA)
    Returns:
        rly : (2l+1,)    —— r^l * Y_{lm}, where m = -l..l
        grly: (2l+1, 3)  —— ∇(r^l * Y_{lm}), x, y, z components
    """
    # special case: r = 0
    if r <= tiny:
        rly  = np.zeros((2*l + 1,), dtype='f8')
        grly = np.zeros((2*l + 1, 3), dtype='f8')
        if l == 0:
            rly[l + 0] = _C_lm(0, 0)
        elif l == 1:
            c1 = _C_lm(1, -1)
            c2 = _C_lm(1, 0)
            c3 = _C_lm(1, 1)
            # m = -1: -c1 * y
            grly[l - 1, 1] = -c1
            # m = 0:   c2 * z
            grly[l + 0, 2] =  c2
            # m = +1: -c3 * x
            grly[l + 1, 0] = -c3
        return rly, grly
    
    # explicit formula: l <= 2
    if l == 0:
        rly  = np.zeros((1,), dtype='f8')
        grly = np.zeros((1, 3), dtype='f8')
        rly[0] = _C_lm(0, 0)
        return rly, grly

    if l == 1:
        rly  = np.zeros((3,), dtype='f8')
        grly = np.zeros((3, 3), dtype='f8')
        c1 = _C_lm(1, -1)
        c2 = _C_lm(1, 0)
        c3 = _C_lm(1, 1)
        # m = -1: -c1 * y
        rly[0]     = -c1 * y
        grly[0, 1] = -c1
        # m = 0:   c2 * z
        rly[1]     =  c2 * z
        grly[1, 2] =  c2
        # m = +1: -c3 * x
        rly[2]     = -c3 * x
        grly[2, 0] = -c3
        return rly, grly

    if l == 2:
        rly  = np.zeros((5,), dtype='f8')
        grly = np.zeros((5, 3), dtype='f8')
        # m = -2: +C4 * 6 x y
        C4 = _C_lm(2, -2)
        rly[0]     =  C4 * 6.0 * x * y
        grly[0, 0] =  C4 * 6.0 * y
        grly[0, 1] =  C4 * 6.0 * x
        # m = -1: -C5 * 3 y z
        C5 = _C_lm(2, -1)
        rly[1]     = -C5 * 3.0 * y * z
        grly[1, 1] = -C5 * 3.0 * z
        grly[1, 2] = -C5 * 3.0 * y
        # m = 0: +C6 * 0.5*(2 z^2 - x^2 - y^2)
        C6 = _C_lm(2, 0)
        rly[2]     =  C6 * 0.5 * (2.0*z*z - x*x - y*y)
        grly[2, 0] = -C6 * x
        grly[2, 1] = -C6 * y
        grly[2, 2] =  C6 * 2.0 * z
        # m = +1: -C7 * 3 x z
        C7 = _C_lm(2, 1)
        rly[3]     = -C7 * 3.0 * x * z
        grly[3, 0] = -C7 * 3.0 * z
        grly[3, 2] = -C7 * 3.0 * x
        # m = +2: +C8 * 3 (x^2 - y^2)
        C8 = _C_lm(2, 2)
        rly[4]     =  C8 * 3.0 * (x*x - y*y)
        grly[4, 0] =  C8 * 6.0 * x
        grly[4, 1] = -C8 * 6.0 * y
        return rly, grly
    
    # general recursive case: l >= 3 
    xhat, yhat, zhat = x/r, y/r, z/r
    xyhat = np.hypot(xhat, yhat)
    if xyhat < tiny:
        xhat = tiny
        xyhat = np.hypot(xhat, yhat)
    cosphi, sinphi = xhat / xyhat, yhat / xyhat
    # recursively calculate P(l,m) and its angular derivatives ZP(l,m)
    P  = np.zeros((l+2, l+2), dtype='f8')
    ZP = np.zeros((l+1, l+1), dtype='f8')
    for M in range(l, -1, -1):
        P[M, M] = 1.0
        fac = 1.0
        for _ in range(1, M+1):
            P[M, M] = -(P[M, M] * fac * xyhat)
            fac += 2.0
        P[M+1, M] = zhat * (2*M + 1) * P[M, M]
        for L in range(M+2, l+1):
            P[L, M] = (zhat*(2*L - 1)*P[L-1, M] - (L+M-1)*P[L-2, M])/(L - M)
        ZP[l, M] = -((M * P[l, M] * zhat / xyhat + P[l, M+1]) / xyhat)
    RL   = r**l
    RLm1 = r**(l-1)
    rly  = np.zeros((2*l + 1,), dtype='f8')
    grly = np.zeros((2*l + 1, 3), dtype='f8')
    cosm, sinm = 1.0, 0.0
    for M in range(0, l+1):
        Plm, ZPlm = P[l, M], ZP[l, M]
        # m = -M (sin component)
        if M > 0:
            Cmm = _C_lm(l, -M)
            YY  = Cmm * Plm * sinm
            rly[l - M] = RL * YY
            GY1 = -ZPlm * xhat * zhat * sinm - Plm * M * cosm * sinphi / xyhat
            GY2 = -ZPlm * yhat * zhat * sinm + Plm * M * cosm * cosphi / xyhat
            GY3 =  ZPlm * xyhat * xyhat * sinm
            grly[l - M, 0] = xhat * l * RLm1 * YY + RL * GY1 * Cmm / r
            grly[l - M, 1] = yhat * l * RLm1 * YY + RL * GY2 * Cmm / r
            grly[l - M, 2] = zhat * l * RLm1 * YY + RL * GY3 * Cmm / r
        # m = +M (cos component; M = 0 is also here)
        Cpm = _C_lm(l, +M)
        YY  = Cpm * Plm * cosm
        rly[l + M] = RL * YY
        GY1 = -ZPlm * xhat * zhat * cosm + Plm * M * sinm * sinphi / xyhat
        GY2 = -ZPlm * yhat * zhat * cosm - Plm * M * sinm * cosphi / xyhat
        GY3 =  ZPlm * xyhat * xyhat * cosm
        grly[l + M, 0] = xhat * l * RLm1 * YY + RL * GY1 * Cpm / r
        grly[l + M, 1] = yhat * l * RLm1 * YY + RL * GY2 * Cpm / r
        grly[l + M, 2] = zhat * l * RLm1 * YY + RL * GY3 * Cpm / r

        cosm, sinm = cosm * cosphi - sinm * sinphi, cosm * sinphi + sinm * cosphi
    
    return rly, grly

_PHASE_CACHE = {}  # { l : phase[2l+1] }, each element is ±1

def _phase_map_to_match_Y(l, trials=7, seed=42):
    """
    estimate the phase to keep SIESTA spherical harmonics basis and GPAW - Y() aligned (±1), length is 2l+1.
    Use multiple random directions for majority voting, discarding values close to 0.
    """
    rng = np.random.default_rng(seed)
    votes = []

    for _ in range(trials):
        # generate a random non-axial unit vector
        u = rng.normal(size=3)
        u /= np.linalg.norm(u)
        x, y, z = float(u[0]), float(u[1]), float(u[2])
        R = 1.2345
        # r^l Y: from Y()
        rly_Y = np.array([(R**l) * Y(l*l+l+m, x, y, z) for m in range(-l, l+1)], dtype='f8')
        # r^l Y: from SIESTA
        rly_S, _ = rly_grly_single(l, R, R*x, R*y, R*z)
        eps = 1e-12
        # vote to 0 for components close to 0, others by sign ratio
        s = np.zeros((2*l+1,), dtype=np.int8)
        mask = np.abs(rly_S) > eps
        s[mask] = np.sign(rly_Y[mask] / rly_S[mask]).astype(np.int8)
        # treat 0 as +1 (conservative); can also keep 0 for majority
        s[s == 0] = 1
        votes.append(s)

    votes = np.stack(votes, axis=0).astype(np.int32)
    phase = np.sign(np.sum(votes, axis=0)).astype(np.int8)
    phase[phase == 0] = 1
    return phase.astype(np.float64)

def get_phase(l):
    """
    get the phase the same as Y(), if not cached, compute and cache it
    """
    if l not in _PHASE_CACHE:
        _PHASE_CACHE[l] = _phase_map_to_match_Y(l)
    return _PHASE_CACHE[l]

def grad_spharm_xyz(l, R, x, y, z):
    # x, y, z must be normalized
    assert x.shape[0] == y.shape[0] == z.shape[0] == R.shape[0]
    assert len(x.shape) == len(y.shape) == len(z.shape) == len(R.shape) == 1
    spharm = np.zeros((x.shape[0], 2*l+1), dtype='f8')
    grad_spharm = np.zeros((x.shape[0], 2*l+1, 3), dtype='f8')
    # calculate rly and grly following siesta
    for i in range(x.shape[0]):
        rly, grly = rly_grly_single(l, R[i], R[i]*x[i], R[i]*y[i], R[i]*z[i], tiny=1e-14)
        spharm[i, :] = rly
        grad_spharm[i, :, :] = grly
    # keep the phase of spharm and grad_spharm the same as Y()
    phase = get_phase(l)
    spharm *= phase[None, :]
    grad_spharm *= phase[None, :, None]
    return spharm, grad_spharm

def spharm_old(r, l):
    raise DeprecationWarning()
    '''
    Real spherical harmonics:
    https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#Real_spherical_harmonics
    
    Parameters:
    ---------
    r(..., xyz): unnormalized real vectors
    l: integer
    
    Returns:
    ---------
    spharm(..., 2*l+1): spherical harmonics Y_lm(r)
    '''
    
    rshape = r.shape
    r = r.reshape((-1, 3))
    
    eps = 1e-8
    rnorm = np.linalg.norm(r, axis=-1)
    theta = np.arccos(r[..., 2] / (rnorm + eps/10))
    phi = np.arctan2(r[..., 1], r[..., 0])
    # r=0
    r_is_zero = rnorm < eps
    theta[r_is_zero] = 0.0
    phi[r_is_zero] = 0.0
    
    spharm = np.zeros((r.shape[0], 2*l+1), dtype=np.complex128)
    for m in range(-l, l+1):
        if m < 0:
            # note that scipy uses different names for theta, phi
            out = 1.j/np.sqrt(2.0)*(         sp.sph_harm( m, l, phi, theta)
                                    -(-1)**m*sp.sph_harm(-m, l, phi, theta))
        elif m == 0:
            out = sp.sph_harm(0, l, phi, theta)
        else:  # m>0
            out = 1.0/np.sqrt(2.0)*(         sp.sph_harm(-m, l, phi, theta)
                                    +(-1)**m*sp.sph_harm( m, l, phi, theta))
        spharm[..., m+l] = out
    
    assert np.max(np.abs(spharm.imag)) < 1e-8
    spharm = spharm.real

    return spharm.reshape(rshape[:-1]+(2*l+1,))


'''
Utility functions related to FT of atomic orbitals
'''

def spbessel_transfrorm(l, k, rgd, R_r, norm='forward'):
    r'''
    Calculate spherical bessel transformation.
    If norm='forward', then calculate
    F_l(k) = 4 \pi (-i)^l \int dr r^2 R_l(r) j_l(kr)
    If norm='backward', then calculate
    R_l(r) = \frac{1}{(2\pi)^3} 4 \pi i^l \int dk k^2 F_l(k) j_l(kr)
    
    Parameters:
    ---------
    l: integer
    k: float
    R_r(ngrid): function R_l(r) on a radial grid
    rgrid: RadialGrid object
    norm: 'forward' or 'backward'
    
    Returns:
    ---------
    sbt_real: float, F_l(k)
    cplx_phase: phase part of spherical bessel transform
    '''
    
    r = rgd.rfunc
    kr = k * r
    j_l = sp.spherical_jn(l, kr)
    sbt_real = rgd.sips(j_l*R_r)
    if norm=='forward':
        sbt_real *= 4.0*np.pi
        cplx_phase = (-1j)**l
    elif norm=='backward':
        sbt_real *= 1. / (2*np.pi**2)
        cplx_phase = (1j)**l
    else:
        raise ValueError(f'Norm must be "forward" or "backward", not {norm}')
    return sbt_real, cplx_phase


'''
Utility functions related to k-points and g-vecs
'''

class kGsphere:
    '''
    Find all g-vectors in the sphere of given cutoff energy
    '''
    def __init__(self, rprim, ecut):
        maxgnorm = np.sqrt(2 * ecut)
        maxgidx = np.floor(maxgnorm * np.linalg.norm(rprim, axis=-1) / (2*np.pi) + 1).astype(int)
        gk_g_all = np.stack(np.meshgrid(np.linspace(-maxgidx[0], maxgidx[0], 2*maxgidx[0]+1, dtype=int),
                                        np.linspace(-maxgidx[1], maxgidx[1], 2*maxgidx[1]+1, dtype=int),
                                        np.linspace(-maxgidx[2], maxgidx[2], 2*maxgidx[2]+1, dtype=int)),
                            axis=-1).reshape(-1, 3)
        
        self.rprim = rprim
        self.gprim = np.linalg.inv(rprim.T)
        self.maxgnorm = maxgnorm
        self.maxgidx = maxgidx
        self.FFTgrid = 2*(1+maxgidx)
        self.gk_g_all = gk_g_all
        
    def get_gk_g(self, kpt):
        '''
        Only works for -1<=kpt<=1
        kpt(abc): k point vector in reduced coordinate
        '''
        kgcart_all = 2 * np.pi * (kpt[None, :] + self.gk_g_all) @ self.gprim
        within_cutoff = np.linalg.norm(kgcart_all, axis=-1) <= self.maxgnorm
        gk_g = self.gk_g_all[within_cutoff]
        ngk_g = gk_g.shape[0]
        kgcart = kgcart_all[within_cutoff]
        
        return ngk_g, gk_g, kgcart


def same_kpt(kpt1, kpt2):
    eps = 1.0e-5
    return np.all(np.abs(kpt1 - kpt2) < eps, axis=-1)


def diff_by_G(kpt1, kpt2):
    '''
    Check if two kpts are only different by reciprocal lattice vector G

    Parameters:
    ---------
      kpt1: [(extra_dimensions), 3]
      kpt2: [(extra_dimensions), 3]
    
    Returns:
    ---------
      same_kpt [(extra_dimensions)] containing boolean values
    '''
    eps = 1.0e-5
    kdiff = kpt1 - kpt2
    kdiff1BZ = firstBZ(kdiff)
    samekpt = np.all(np.abs(kdiff1BZ) < eps, axis=-1)
    return samekpt


# def same_kpt_sym(kpt1, kpt2, symopt=0):
#     '''
#     Parameters:
#     ---------
#       kpt1: [(extra_dimensions), 3]
#       kpt2: [(extra_dimensions), 3]
#       symopt: 0 - no symmetry; 1 - only time-reversal symmetry
    
#     Returns:
#     ---------
#       same_kpt [(extra_dimensions)] containing boolean values
#     '''
#     assert symopt in [0,  1]
#     if symopt == 0:
#         samekpt = same_kpt(kpt1, kpt2)
#     elif symopt == 1:
#         samekpt = same_kpt(kpt1, kpt2)
#         samekpt += same_kpt(-kpt1, kpt2)
#     return samekpt


def firstBZ(kpt):
    '''
    find kpt in 1BZ (-0.5, 0.5]
    '''
    return -np.divmod(-kpt+0.5, 1)[1] + 0.5


def find_kidx(kpt, kqpt, allow_multiple=True):
    '''
    Find the index of kqpt (one point) in the kpt (all points).
    If multiple match exists, only return the index of the first match.
    If there's no match, then return -1
    '''

    kidx = np.where(same_kpt(kpt, kqpt))[0]
    if not ((kidx.ndim==1) and (kidx.shape[0]>=1)):
        msg = f'\nCannot find kpt {kqpt} among k points:\n'
        msg += str(kpt)
        raise ValueError(msg)
    if (not allow_multiple) and (kidx.shape[0]>1):
        msg = f'Multiple kpt {kqpt} found among k points:\n'
        msg += str(kpt)
        raise ValueError(msg)
    return kidx[0].item()


def make_kkmap(kpt1, kpt2):
    '''
    maps every point in kpt1 to kpt2
    
    Returns:
    ---------
    kkmap[nk2]: kkmap[i] is the index of the k-point in kpt1 corresponding to the ith k-point in kpt2
    '''
    # nk = kpt1.shape[0]
    # assert kpt2.shape[0] == nk
    nk2 = kpt2.shape[0]
    
    kkmap = np.zeros(nk2, dtype=int)
    for ik, kqpt in enumerate(kpt2):
        kkmap[ik] = find_kidx(kpt1, kqpt, allow_multiple=True)
      
    return kkmap


def kgrid_with_tr(gridsize):
    '''
    Generate k grid with time-reversal symmetry
    '''
    if len(gridsize) == 3:
        kgrid = np.array(gridsize)
        shift = np.zeros(3)
    elif len(gridsize) == 6:
        kgrid = np.array(gridsize[:3])
        shift = np.array(gridsize[3:])
    else:
        raise ValueError()
    kpts, kptwts = [], []
    grid_interval = 1 / kgrid
    for n1 in range(kgrid[0]):
        for n2 in range(kgrid[1]):
            for n3 in range(kgrid[2]):
                ns = np.array([n1, n2, n3])
                kpt = (ns + shift) * grid_interval
                found = False
                for ikold in range(len(kpts)):
                    if diff_by_G(-kpt, kpts[ikold]):
                        kptwts[ikold] += 1
                        found = True
                if not found:
                    kpts.append(firstBZ(kpt))
                    kptwts.append(1)
    kptwts = np.array(kptwts)
    kpts = np.stack(kpts)
    total = np.sum(kptwts)
    assert total == np.prod(kgrid)
    kptwts = kptwts / total
    return kpts, kptwts