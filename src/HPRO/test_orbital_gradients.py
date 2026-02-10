import numpy as np
from .mathutils import r_to_xyz

def _sample_points_in_ball(npts, radius, avoid_zero=True, seed=None):
    """
    random sampling within rcut 
    """
    rng = np.random.default_rng(seed)
    # uniform sampling in the direction 
    u = rng.normal(size=(npts, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    # uniform sampling of r -> r = radius * U^(1/3)
    R = radius * rng.random(npts)**(1/3)
    if avoid_zero:
        R = np.maximum(R, 1e-3)
    return u * R[:, None]

def _central_diff_along_axes(orb, Rvec, h):
    """
    axis-wise central differences of orbitals
    """
    N = Rvec.shape[0]
    grad_fd = np.zeros((N, 2*orb.l + 1, 3), dtype=np.float64)
    # x
    Rp = Rvec.copy(); Rp[:,0] += h
    Rm = Rvec.copy(); Rm[:,0] -= h
    fp = orb.generate3D_noselect(Rp)
    fm = orb.generate3D_noselect(Rm)
    grad_fd[:,:,0] = (fp - fm) / (2*h)
    # y
    Rp = Rvec.copy(); Rp[:,1] += h
    Rm = Rvec.copy(); Rm[:,1] -= h
    fp = orb.generate3D_noselect(Rp)
    fm = orb.generate3D_noselect(Rm)
    grad_fd[:,:,1] = (fp - fm) / (2*h)
    # z
    Rp = Rvec.copy(); Rp[:,2] += h
    Rm = Rvec.copy(); Rm[:,2] -= h
    fp = orb.generate3D_noselect(Rp)
    fm = orb.generate3D_noselect(Rm)
    grad_fd[:,:,2] = (fp - fm) / (2*h)
    return grad_fd

def _central_diff_directional(orb, Rvec, vhat, h):
    """
    central difference for directional derivatives
    """
    Rp = Rvec + h * vhat
    Rm = Rvec - h * vhat
    fp = orb.generate3D_noselect(Rp)
    fm = orb.generate3D_noselect(Rm)
    return (fp - fm) / (2*h)

def _error_stats(a, b, rel_floor=1e-12):
    """
    return (max, rms) of absolute error and relative error
    denominator of relative error is max(|b|, rel_floor) to avoid amplification by 0
    """
    diff = a - b
    abs_err = np.abs(diff) 
    rel_err = abs_err / np.maximum(np.abs(b), rel_floor)
    def stats(x):
        return float(np.max(x)), float(np.sqrt(np.mean(x*x)))
    return stats(abs_err), stats(rel_err)

def _test_orbital_gradients(orb, grad_orb, npts=100, h_list=(1e-3, 3e-4, 1e-4, 3e-5),
                          seed=2025, verbose=True):
    """
    test generate3D_grad() by finite differences:
      - axis-wise central differences  vs analytical gradients
      - random directional derivatives vs inner product of analytical gradients
    """
    # sample points
    rcut = max(0.5, orb.rcut - 20 * max(h_list))
    Rvec = _sample_points_in_ball(npts, rcut, seed=seed)
    Rnorm, x, y, z = r_to_xyz(Rvec)
    # analytical gradients, dimension is (N, 2l+1, 3)
    grad_an = grad_orb.generate3D_grad_norm(Rnorm, x, y, z)     

    # scan different step size for axis-wise central differences
    if verbose:
        print(f"[Axis-wise central differences]  npts={npts}, rcut≈{rcut:.2f}")
        print("  h         |  abs max     abs rms   |  rel max     rel rms")

    for h in h_list:
        grad_fd = _central_diff_along_axes(orb, Rvec, h)  # (N, 2l+1, 3)
        (amax, arms), (rmax, rrms) = _error_stats(grad_fd, grad_an)
        if verbose:
            print(f"  {h:9.2e} |  {amax:10.3e} {arms:10.3e} |  {rmax:10.3e} {rrms:10.3e}")

    # select a mediate step size for directional derivatives
    h = h_list[min(2, len(h_list)-1)]
    rng = np.random.default_rng(seed+1)
    v = rng.normal(size=Rvec.shape)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    dphi_dv_fd = _central_diff_directional(orb, Rvec, v, h)          # (N, 2l+1)
    dphi_dv_an = np.einsum('nij,nj->ni', grad_an, v)                 # (N, 2l+1)
    (amax, arms), (rmax, rrms) = _error_stats(dphi_dv_fd, dphi_dv_an)

    if verbose:
        print("\n[Directional derivative check]  (using middle h)")
        print(f"  abs max = {amax:.3e}, abs rms = {arms:.3e}")
        print(f"  rel max = {rmax:.3e}, rel rms = {rrms:.3e}")

    ok = (rrms < 5e-6) and (rmax < 1e-3)
    if verbose:
        print(f"\nResult: {'PASS' if ok else 'CHECK'}")
    return ok