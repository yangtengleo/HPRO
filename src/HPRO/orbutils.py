import xml.etree.ElementTree as ET
from math import pi
import numpy as np
import copy
import os
import sys
from scipy.integrate import simpson
from scipy.interpolate import CubicSpline
import scipy.special as sp

from .structure import Structure
from .from_gpaw.gaunt import gaunt
from .mathutils import r_to_xyz, spbessel_transfrorm, spharm_xyz, grad_spharm_xyz

# == radial grids ==

class GridFunc:
    '''
    This is the class for functions that can be separated into a radial part and angular part.
    '''
    def __init__(self, rgd, func, l=0, rcut=None):
        """
        Initializes a GridFunc object.

        Parameters:
            rgd (RadialGrid): The radial grid object.
            func (array): The function values on the radial grid.
            l (int, optional): The angular momentum quantum number. Defaults to 0.
            rcut (float, optional): The radial cutoff. If None, it will be automatically detected. Defaults to None.
        """
        self.rgd = rgd
        self.func = func
        self.generator = None
        self.l = l
        
        # get rcut
        if rcut is None and func is not None:
            rcut = self.rgd.rend
        self.rcut = rcut
    
    def calc_generator(self):
        '''
        Calculate and return a generator for the radial function.
        '''
        self.generator = self.rgd.generator(self.func)
        return self.generator

    def generate(self, r):
        '''
        Generate the atomic orbital function at a given set of radial coordinates.
        '''
        if self.generator is None:
            self.calc_generator()
        assert np.max(r) <= self.rgd.rend
        assert np.min(r) >= self.rgd.rstart
        return self.rgd.generate(self.generator, r)
    
    def generate_grad(self, r):
        '''
        Same as generate, but returns both function values and their gradients.
        '''
        if self.generator is None:
            self.calc_generator()
        assert np.max(r) <= self.rgd.rend
        assert np.min(r) >= self.rgd.rstart
        return self.rgd.generate(self.generator, r), self.rgd.generate_grad(self.generator, r)

    def generatexyz(self, Rnorm, x, y, z):
        Ylm = spharm_xyz(self.l, x, y, z)
        Rlm = self.generate(Rnorm)
        return Rlm[:, None] * Ylm[:, :]
    
    def generatexyz_grad(self, Rnorm, x, y, z):
        Rl_Ylm, grad_Rl_Ylm = grad_spharm_xyz(self.l, Rnorm, x, y, z)
        Rlm, grad_Rlm = self.generate_grad(Rnorm)
        Rhat = np.concatenate([x[:, None], y[:, None], z[:, None]], axis=1)
        return grad_Rlm[:, None, None] * Rl_Ylm[:, :, None] * Rhat[:, None, :] + Rlm[:, None, None] * grad_Rl_Ylm[:, :, :]

    def generate3D(self, Rvec):
        '''
        Generate the atomic orbital function at a given set of cartesian coordinates.
        '''
        rshape0 = Rvec.shape[:-1]
        Rvec = Rvec.reshape(-1, 3)
        Rnorm , x, y, z = r_to_xyz(Rvec)
        # deal with points within the range of the radial grid (outsiders are set to zero)
        Rwithin = Rnorm <= self.rgd.rend
        nwithin = np.sum(Rwithin)
        R_lm = np.zeros((Rvec.shape[0], 2*self.l+1))
        if nwithin > 0:
            Rnorm, x, y, z = Rnorm[Rwithin], x[Rwithin], y[Rwithin], z[Rwithin]
            R_lm[Rwithin, :] = self.generatexyz(Rnorm, x, y, z)
        # deal with points smaller than self.rgd.rstart
        Rsmall = Rnorm < self.rgd.rstart
        nsmall = np.sum(Rsmall)
        if nsmall > 0:
            R_lm[Rsmall, :] = 0. if self.l > 0 else self.func[0]
        return R_lm.reshape(rshape0 + (2*self.l+1,))

    def generate3D_noselect(self, Rvec):
        '''
        Same as `generate3D` but assume all points are already within the range of generated orbitals. 
        This can be more efficient than `generate3D` but use with care.
        '''
        rshape0 = Rvec.shape[:-1]
        Rvec = Rvec.reshape(-1, 3)
        Rnorm, x, y, z = r_to_xyz(Rvec)
        R_lm = self.generatexyz(Rnorm, x, y, z)
        return R_lm.reshape(rshape0 + (2*self.l+1,))

    def generate3D_norm(self, Rnorm, x, y, z):
        '''
        Same as `generate3D_noselect` but assume all points are normalized.
        '''
        R_lm = self.generatexyz(Rnorm, x, y, z)
        return R_lm.reshape((Rnorm.shape[0], 2*self.l+1))
    
    def generate3D_grad_norm(self, Rnorm, x, y, z):
        '''
        Returns the orbital gradients, assume all points are normalized.
        '''
        grad_R_lm = self.generatexyz_grad(Rnorm, x, y, z)
        return grad_R_lm.reshape((Rnorm.shape[0], 2*self.l+1, 3))
    
    def generate3D_norm_check(self, Rnorm, x, y, z):
        '''
        Same as `generate3D_norm` but check the range.
        '''
        Rwithin = Rnorm <= self.rgd.rend
        nwithin = np.sum(Rwithin)
        R_lm = np.zeros((Rnorm.shape[0], 2*self.l+1))
        if nwithin > 0:
            Rnorm, x, y, z = Rnorm[Rwithin], x[Rwithin], y[Rwithin], z[Rwithin]
            R_lm[Rwithin, :] = self.generatexyz(Rnorm, x, y, z)
        Rsmall = Rnorm < self.rgd.rstart
        nsmall = np.sum(Rsmall)
        if nsmall > 0:
            R_lm[Rsmall, :] = 0. if self.l > 0 else self.func[0]
        return R_lm
    
    def generate3D_grad_norm_check(self, Rnorm, x, y, z):
        '''
        Same as `generate3D_grad_norm` but check the range.
        '''
        Rwithin = Rnorm <= self.rgd.rend
        nwithin = np.sum(Rwithin)
        grad_R_lm = np.zeros((Rnorm.shape[0], 2*self.l+1, 3))
        if nwithin > 0:
            Rnorm, x, y, z = Rnorm[Rwithin], x[Rwithin], y[Rwithin], z[Rwithin] 
            grad_R_lm[Rwithin, :, :] = self.generatexyz_grad(Rnorm, x, y, z)
        Rsmall = Rnorm < self.rgd.rstart
        nsmall = np.sum(Rsmall)
        if nsmall > 0:
            grad_R_lm[Rsmall, :, :] = 0.
        return grad_R_lm


class RadialGrid:
    '''
    This is the base class for radial grids.
    '''
    def __init__(self, rfunc):
        self.rfunc = rfunc
        self.rstart = rfunc[0]
        self.rend = rfunc[-1]
        self.npoints = len(rfunc)
    
    def sips(self, func_on_grid, n=2):
        assert len(func_on_grid) == self.npoints
        return simpson(func_on_grid * self.rfunc**n, self.rfunc)

    def generator(self, func_on_grid):
        return CubicSpline(self.rfunc, func_on_grid)
    
    def generate(self, generator, r):
        return generator(r)

    def generate_grad(self, generator, r):
        return generator(r, 1)
    
    def r2i_ceil(self, r):
        return np.searchsorted(self.rfunc, r)
    
    def __eq__(self, other):
        if self is other:
            return True
        else:
            if not self.__class__ is other.__class__: return False
            if not np.all(np.abs(self.rfunc-other.rfunc)) < 1e-6: return False
            return True
    
class LinearRGD(RadialGrid):
    def __init__(self, rstart, rend, npoints):
        # includes both ends
        assert rend > rstart >= 0
        assert npoints > 1
        rfunc = np.linspace(rstart, rend, npoints)
        super().__init__(rfunc)
        self.dx = (self.rend - self.rstart) / (self.npoints - 1)
    
    def sips(self, func_on_grid, n=2):
        assert len(func_on_grid) <= self.npoints
        return simpson(func_on_grid * self.rfunc**n, dx=self.dx)

    @classmethod
    def from_explicit_grid(cls, rfunc):
        rstart = rfunc[0]
        rend = rfunc[-1]
        npoints = len(rfunc)
        obj = cls(rstart, rend, npoints)
        assert np.all(np.abs(obj.rfunc - rfunc)) < 1e-6
        return obj

class FracPolyRGD(RadialGrid):
    # r=a*i/(n-i)
    def __init__(self, a, n):
        ilist = np.array(list(range(n)))
        rfunc = a * ilist / (n - ilist)
        super().__init__(rfunc)
        self.dr_dx = a * n / (n - ilist) ** 2
        self.a = a
    
    def sips(self, func_on_grid, n=2):
        assert len(func_on_grid) == self.npoints
        # return simpson(func_on_grid * self.dr_dx)
        return simpson(func_on_grid * self.rfunc**n, self.rfunc)

class ExpRGD(RadialGrid):
    # r=a*exp(d*i) or r=a*(exp(d*i)-1)
    def __init__(self, npoints, a, d, minus1=False):
        ilist = np.arange(0, npoints, dtype=float) # istart=0
        rfunc = a * np.exp(d * ilist)
        if minus1:
            rfunc -= a
        super().__init__(rfunc)
        self.a = a
        self.d = d
        self.minus1 = minus1
    
    def sips(self, func_on_grid, n=2):
        assert len(func_on_grid) == self.npoints
        return simpson(func_on_grid * self.rfunc**(n+1), dx=self.d)
    
    @classmethod
    def from_explicit_grid(cls, rfunc):
        a = rfunc[0]
        npoints = len(rfunc) - 1
        d = np.log(rfunc[npoints-1] / rfunc[0]) / (npoints-1)
        npoints = len(rfunc)
        obj = cls(npoints, a, d)
        assert np.all(np.abs(obj.rfunc - rfunc)) < 1e-6
        return obj
    
    def generator(self, func_on_grid):
        return CubicSpline(np.arange(0, self.npoints, dtype=float), func_on_grid)
    
    def generate(self, generator, r):
        assert np.all(r>self.rstart)
        i_fromr = np.log(r/self.a) / self.d 
        return generator(i_fromr)

# grid FTs

def grid_overlap(gridfunc1, gridfunc2):
    '''
    Overlap of two functions centered at the same point
    '''
    assert gridfunc1.rgd == gridfunc2.rgd
    rgd = gridfunc1.rgd
    return rgd.sips(gridfunc1.func * gridfunc2.func)

def grid_G2R(rgrid, gridfuncG, return_real=True):
    '''
    Perform radial Fourier transformation from reciprocal space to real space.
    '''
    # todo: fast Bessel transform
    dtype = 'f8' if return_real else 'c16'
    funcR = np.empty(rgrid.npoints, dtype=dtype)
    for ir in range(rgrid.npoints):
        phi, phase = spbessel_transfrorm(gridfuncG.l, rgrid.rfunc[ir], gridfuncG.rgd, gridfuncG.func, norm='backward')
        funcR[ir] = phi if return_real else phi * phase
    return GridFunc(rgrid, funcR, l=gridfuncG.l)

def grid_R2G(Ggrid, gridfuncR, return_real=True):
    '''
    Perform radial Fourier transformation from real space to reciprocal space.
    '''
    # todo: fast Bessel transform
    dtype = 'f8' if return_real else 'c16'
    funcR = np.empty(Ggrid.npoints, dtype=dtype)
    for ir in range(Ggrid.npoints):
        phi, phase = spbessel_transfrorm(gridfuncR.l, Ggrid.rfunc[ir], gridfuncR.rgd, gridfuncR.func, norm='forward')
        funcR[ir] = phi if return_real else phi * phase
    return GridFunc(Ggrid, funcR, l=gridfuncR.l)
    
# == load orbitals ==

def parse_siesta_ion(filename):
    
    phirgrids = []
    grad_phirgrids = []
    norb = 0
    
    ionfile = open(filename, 'r')
    line = ionfile.readline()
    while line:
        if line.find('#orbital l, n, z, is_polarized, population') > 0:
            sp = line.split()
            l = int(sp[0])
            norb += 1
            
            line = ionfile.readline()
            assert line.split()[0] == '500'
            
            grad_phirgrid = np.zeros((2, 500)) # r, R(r)/r^l
            for ipt in range(500):
                grad_phirgrid[:, ipt] = list(map(float, ionfile.readline().split()))
            # found this from sisl/io/siesta/siesta_nc.py: ncSileSiesta.read_basis(self): 
            # sorb = SphericalOrbital(l, (r * Bohr2Ang, psi), orb_q0[io])
            phirgrid = copy.deepcopy(grad_phirgrid) # r, R(r)
            phirgrid[1, :] *= np.power(grad_phirgrid[0, :], l) 
            rgd = LinearRGD.from_explicit_grid(phirgrid[0])
            phirgrids.append(GridFunc(rgd, phirgrid[1], l=l))
            grad_phirgrids.append(GridFunc(rgd, grad_phirgrid[1], l=l))
        line = ionfile.readline()
    ionfile.close()

    return norb, phirgrids, grad_phirgrids


def parse_gpaw_basis(filename):
    root = ET.parse(filename).getroot()
    gridfuncs = {}
    for gridfunc in root.findall('radial_grid'):
        if gridfunc.attrib['eq'] == 'r=d*i':
            istart = int(gridfunc.attrib['istart'])
            iend = int(gridfunc.attrib['iend'])
            d = float(gridfunc.attrib['d'])
            rgd = LinearRGD(istart*d, iend*d, iend-istart+1)
            gridid = gridfunc.attrib['id']
            gridfuncs[gridid] = rgd
        else:
            raise NotImplementedError

    phirgrids = []
    for basisfunc in root.findall('basis_function'):
        l = int(basisfunc.attrib['l'])
        gridid = basisfunc.attrib['grid']
        phi = np.array(list(map(float, basisfunc.text.split())))
        gridlen = len(phi)
        rgd = LinearRGD.from_explicit_grid(gridfuncs[gridid].rfunc[:gridlen])
        phirgrids.append(GridFunc(rgd, phi, l=l))

    norb = len(phirgrids)
    
    return norb, phirgrids

def parse_deeph_orbtyps(deephsave):
    stru = Structure.from_deeph(deephsave)
    orbital_types = []
    with open(f'{deephsave}/orbital_types.dat') as f:
        line = f.readline()
        while line:
            orbital_types.append(list(map(int, line.split())))
            line = f.readline()
    orbital_types_spc = {}
    for atom_nbr, orbitals in zip(stru.atomic_numbers, orbital_types):
        if atom_nbr in orbital_types_spc:
            assert orbitals == orbital_types_spc[atom_nbr]
        else:
            orbital_types_spc[atom_nbr] = orbitals
    return orbital_types_spc, stru


class OrbPair:
    def __init__(self, rgrid1, rgrid2, rcut, index=1):
        
        grid_nR = int(rcut * 16.66)
        gridR = LinearRGD(0, rcut, grid_nR)
        gridQ = rgrid1.rgd
        assert rgrid1.rgd == rgrid2.rgd
        l1 = rgrid1.l
        l2 = rgrid2.l
        l3 = max(l1, l2)
        self.lmax = l3
        
        Sl = np.empty((2*l3+1, gridR.npoints))
        for iR in range(gridR.npoints):
            R = gridR.rfunc[iR]
            for yy in range(0, 2*l3+1):
                kr = gridQ.rfunc * R
                j_l = sp.spherical_jn(yy, kr)
                xx = (-1)**((l1-l2-yy)//2) / (2*pi**2)
                if index == 1 or index == 3:
                    Sl[yy, iR] = gridQ.sips(j_l*rgrid1.func*rgrid2.func, n=2) * xx
                elif index == 2:
                    Sl[yy, iR] = gridQ.sips(j_l*rgrid1.func*rgrid2.func, n=4) * xx/2
                
        Sl_grids = []
        for yy in range(0, 2*l3+1):
            func = GridFunc(gridR, Sl[yy], l=yy)
            func.calc_generator()
            Sl_grids.append(func)

        self.l1 = l1
        self.l2 = l2
        self.gamma = gaunt(l3)[0:(2*l3+1)**2, l1**2:(l1+1)**2, l2**2:(l2+1)**2]
        self.Sl_grids = Sl_grids
        self.grad_Sl_grids = None

    def grad_setup(self):
        self.grad_Sl_grids = []
        for yy in range(0, 2*self.lmax+1):
            grad_Sl = copy.deepcopy(self.Sl_grids[yy].func)
            if yy != 0:
                grad_Sl[1:] /= np.power(self.Sl_grids[yy].rgd.rfunc[1:], yy)
                grad_Sl[0] = ( 4.0 * grad_Sl[1] - grad_Sl[2] ) / 3.0
            func = GridFunc(self.Sl_grids[yy].rgd, grad_Sl, l=yy)
            func.calc_generator()
            self.grad_Sl_grids.append(func)

    def calc(self, Rnorm, x, y, z):
        Sl_Ylm = np.empty((Rnorm.shape[0], (2*self.lmax+1)**2))
        pos = 0
        for l in range(0, 2*self.lmax+1):
            Sl_Ylm[:, pos:pos+2*l+1] = self.Sl_grids[l].generate3D_norm_check(Rnorm, x, y, z)
            pos += 2 * l + 1
        Sl_3D = np.sum(self.gamma[None, :, :, :] * 
                          Sl_Ylm[:, :, None, None], axis=1)
        return Sl_3D.reshape((Rnorm.shape[0], 2*self.l1+1, 2*self.l2+1))

    def calc_grad(self, Rnorm, x, y, z):
        assert self.grad_Sl_grids is not None
        grad_Sl_Ylm = np.empty((Rnorm.shape[0], (2*self.lmax+1)**2, 3))
        pos = 0
        for l in range(0, 2*self.lmax+1):
            grad_Sl_Ylm[:, pos:pos+2*l+1, :] = self.grad_Sl_grids[l].generate3D_grad_norm_check(Rnorm, x, y, z)
            pos += 2 * l + 1
        grad_Sl_3D = np.sum(self.gamma[None, :, :, :, None] * 
                          grad_Sl_Ylm[:, :, None, None, :], axis=1)
        return grad_Sl_3D.reshape((Rnorm.shape[0], 2*self.l1+1, 2*self.l2+1, 3))

def read_upf(filename):
    """
    Read a QE pseudopotential file in the upf format.

    Let nproj be the number of projector functions, and nproj_full be the sum of 2*l+1 of each projector:

    Returns:
        funch_full array(nproj_full, nproj_full): A 2D array representing the funch matrix.
        projR_list (list): A list of GridFunc objects representing the projector functions.
    """
    
    root = ET.parse(filename).getroot()

    header_elem = root.find('PP_HEADER')
    nproj = int(header_elem.attrib['number_of_proj'])

    r_elem = root.find('PP_MESH').find('PP_R')
    gridsize = int(r_elem.attrib['size'])
    rgridfunc = np.fromiter(map(float, r_elem.text.split()), float, count=gridsize)
    rgrid = LinearRGD.from_explicit_grid(rgridfunc)

    nloc_elem = root.find('PP_NONLOCAL')

    funch_elem = nloc_elem.find('PP_DIJ')
    funch = np.fromiter(map(float, funch_elem.text.split()), float, count=nproj**2).reshape((nproj, nproj)) / 2. # Ry to Har
    
    projR_list = []
    l_list = []
    for iproj in range(nproj):
        projelem = nloc_elem.find(f'PP_BETA.{iproj+1}')
        l = int(projelem.attrib['angular_momentum'])
        rcut = float(projelem.attrib['cutoff_radius'])
        assert int(projelem.attrib['size']) == gridsize
        projfunc = np.fromiter(map(float, projelem.text.split()), float, count=gridsize)
        projfunc[1:] /= rgrid.rfunc[1:] # function in upf is stored as R(r)*r
        projfunc[0] = projfunc[1] if l==0 else 0.
        l_list.append(l)
        projR_list.append(GridFunc(rgrid, projfunc, l=l, rcut=rcut))
    
    funch_full = []
    for iorb in range(len(l_list)):
        funch_row = []
        l1 = l_list[iorb]
        for jorb in range(len(l_list)):
            l2 = l_list[jorb]
            if l1 == l2:
                funch_row.append(np.identity(2*l1+1) * funch[iorb, jorb])
            else:
                assert np.abs(funch[iorb, jorb]) < 1e-8
                funch_row.append(np.zeros((2*l1+1, 2*l2+1)))
        funch_full.append(funch_row)
    funch_full = np.block(funch_full)
    
    return funch_full, projR_list

def read_siesta_ion(dirpath):
    funch, l_list, projR_list = [], [], []
    ionfile = open(os.path.join(dirpath, "P.ion"), 'r')
    line = ionfile.readline()
    while line:
        if line.find('#kb l, n (sequence number), Reference energy') > 0:
            sp = line.split()
            l = int(sp[0])
            l_list.append(l)
            funch.append(float(sp[2]) * 0.5)
            line = ionfile.readline()
            assert line.split()[0] == '500'
            phirgrid = np.zeros((2, 500)) # r, R(r)
            for ipt in range(500):
                phirgrid[:, ipt] = list(map(float, ionfile.readline().split()))
            phirgrid[1, :] *= np.power(phirgrid[0, :], l) 
            rgd = LinearRGD.from_explicit_grid(phirgrid[0])
            projR_list.append(GridFunc(rgd, phirgrid[1], l=l))
        line = ionfile.readline()
    ionfile.close()
    
    funch = np.diag(funch)
    funch_full = []
    for iorb in range(len(l_list)):
        funch_row = []
        l1 = l_list[iorb]
        for jorb in range(len(l_list)):
            l2 = l_list[jorb]
            if l1 == l2:
                funch_row.append(np.identity(2*l1+1) * funch[iorb, jorb])
            else:
                assert np.abs(funch[iorb, jorb]) < 1e-8
                funch_row.append(np.zeros((2*l1+1, 2*l2+1)))
        funch_full.append(funch_row)
    funch_full = np.block(funch_full)
    return funch_full, projR_list


def read_siesta_projectors(dirpath, species_index=None, tol=1e-12, normalize=False):
    kb_file = os.path.join(dirpath, "kb_params.txt")
    proj_file = os.path.join(dirpath, "projectors_r.txt")
    
    rows = []
    with open(kb_file, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            ks, koa, kg, l, rcut, eps = s.split()[:6]
            ks = int(ks); koa = int(koa); kg = int(kg); l = int(l)
            rcut = float(rcut); eps = float(eps)
            if (species_index is None) or (ks == species_index):
                rows.append((ks, koa, kg, l, rcut, eps))
    kg_info = {kg: (l, rcut, eps) for _, _, kg, l, rcut, eps in rows}
    kg_set = set(kg_info.keys())

    blocks = {}  # kg -> dict(l, rcut, r, fr)
    with open(proj_file, "r") as f:
        ig, l, rcut, RMAX = None, None, None, None
        r_list, fr_list = [], []
        for line in f:
            s = line.strip()
            if s.startswith("BEGIN_IG"):
                ig = int(s.split()[-1])
                l = None; rcut = None; RMAX = None
                r_list = []; fr_list = []
            elif s.startswith("END_IG"):
                if ig in kg_set:
                    blocks[ig] = {
                        "l": l,
                        "rcut": rcut,
                        "RMAX": RMAX,
                        "r": np.array(r_list, float),
                        "fr": np.array(fr_list, float),
                    }
                ig = None
            elif ig is not None:
                if s.startswith("l ="):
                    l = int(s.split()[-1])
                elif "rcut" in s:
                    rcut = float(s.split()[-1])
                elif "RMAX" in s:
                    RMAX = float(s.split()[-1])
                else:
                    parts = s.split()
                    if len(parts) == 2:
                        try:
                            r_list.append(float(parts[0]))
                            fr_list.append(float(parts[1]))
                        except:
                            pass
    
    groups = []
    for _, _, kg, l, rcut, eps in sorted(rows, key=lambda x: x[2]):
        b = blocks[kg]
        r = b["r"]; fr = b["fr"]; RMAX = b["RMAX"]
        hit = None
        for g in groups:
            if g["l"] != l:
                continue
            if len(g["r"]) != len(r):
                continue
            if not np.allclose(g["r"], r, rtol=0, atol=0):
                continue
            if np.allclose(g["fr"], fr, rtol=0, atol=tol):
                hit = g; break
        if hit is None:
            groups.append({
                "l": l, "rcut": rcut, "RMAX": RMAX, "r": r, "fr": fr.copy(),
                "eps_list": [eps], "kg_list": [kg]
            })
        else:
            hit["eps_list"].append(eps)
            hit["kg_list"].append(kg)

    projR_list, l_list, eps_diag = [], [], []
    for g in groups:
        l = g["l"]; r = g["r"]; fr = g["fr"].copy()
        rcut = g["rcut"]; RMAX = g["RMAX"]
        eps = float(np.mean(g["eps_list"]))
        if normalize:
            N = np.trapz((r**2) * (fr**2), r)
            fr /= np.sqrt(N)
            eps *= N
        rgrid = LinearRGD.from_explicit_grid(r)
        projR_list.append(GridFunc(rgrid, fr, l=l, rcut=rcut))
        l_list.append(l)
        eps_diag.append(eps)

    eps_diag = np.diag(eps_diag)
    n = len(eps_diag)
    blocks_mat = []
    for i in range(n):
        row_blocks = []
        li = l_list[i]
        for j in range(n):
            lj = l_list[j]
            if li == lj:
                row_blocks.append(np.eye(2*li+1) * eps_diag[i, j])
            else:
                row_blocks.append(np.zeros((2*li+1, 2*lj+1)))
        blocks_mat.append(row_blocks)
    funch_full = np.block(blocks_mat)

    return funch_full, projR_list

def read_siesta_projectors_k(dirpath, species_index=None, tol=1e-12, normalize=False):
    kb_file  = os.path.join(dirpath, "kb_params.txt")
    kproj_file = os.path.join(dirpath, "projectors_k.txt")

    rows = []
    with open(kb_file, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            ks, koa, kg, l, rcut, eps = s.split()[:6]
            ks = int(ks); koa = int(koa); kg = int(kg); l = int(l)
            rcut = float(rcut); eps = float(eps)
            if (species_index is None) or (ks == species_index):
                rows.append((ks, koa, kg, l, rcut, eps))
    kg_info = {kg: (l, rcut, eps) for _, _, kg, l, rcut, eps in rows}
    kg_set = set(kg_info.keys())

    blocks = {}  # kg -> dict(l, rcut, q, fq, QMAX)
    with open(kproj_file, "r") as f:
        kg, l, rcut, QMAX = None, None, None, None
        q_list, fq_list = [], []
        for line in f:
            s = line.strip()
            if s.startswith("BEGIN_IG"):
                kg = int(s.split()[-1])
                l = None; rcut = None; QMAX = None
                q_list = []; fq_list = []
            elif s.startswith("END_IG"):
                if kg in kg_set:
                    blocks[kg] = {
                        "l": l,
                        "rcut": rcut,
                        "QMAX": QMAX,
                        "q": np.array(q_list, float),
                        "fq": np.array(fq_list, float),
                    }
                kg = None
            elif kg is not None:
                if s.startswith("l ="):
                    l = int(s.split()[-1])
                elif "rcut" in s:
                    rcut = float(s.split()[-1])
                elif "QMAX" in s:
                    QMAX = float(s.split()[-1])
                else:
                    parts = s.split()
                    if len(parts) == 2:
                        try:
                            q_list.append(float(parts[0]))
                            fq_list.append(float(parts[1]))
                        except:
                            pass
    
    any_kg = next(iter(blocks))
    q_ref = blocks[any_kg]["q"]
    for kg_b, b in blocks.items():
        if len(b["q"]) != len(q_ref) or not np.allclose(b["q"], q_ref, rtol=0, atol=0):
            raise ValueError("qgrids in projectors_k.txt are not consistent!")

    groups = []
    for _, _, kg, l, rcut, eps in sorted(rows, key=lambda x: x[2]):
        b = blocks[kg]
        fq = b["fq"]; QMAX = b["QMAX"]
        hit = None
        for g in groups:
            if g["l"] != l:
                continue
            if np.allclose(g["fq"], fq, rtol=0, atol=tol):
                hit = g; break
        if hit is None:
            groups.append({
                "l": l, "rcut": rcut, "QMAX": QMAX, "q": q_ref, "fq": fq.copy(),
                "eps_list": [eps], "kg_list": [kg]
            })
        else:
            hit["eps_list"].append(eps)
            hit["kg_list"].append(kg)
    
    phiQ_list = []
    qgrid = LinearRGD.from_explicit_grid(q_ref)
    for g in groups:
        l = g["l"]; fq = g["fq"].copy()
        eps = float(np.mean(g["eps_list"]))
        if normalize:
            m = np.abs(fq) > 1e-12
            k = q_ref[m]
            if k.size > 1:
                # Simpson on LinearRGD: Δk = q_ref[1]-q_ref[0]
                # N = ∫ (k^2/(2π^2)) |F(k)|^2 dk
                dk = q_ref[1] - q_ref[0]
                N = np.sum((k**2) * (np.abs(fq[m])**2)) * dk / (2*np.pi**2)
                if N > 0.0:
                    fq /= np.sqrt(N)
                    eps *= N
        phiQgrid = GridFunc(qgrid, fq, l=l)
        phiQ_list.append(phiQgrid)
    
    return phiQ_list, qgrid

def read_siesta_ao_k(dirpath, species_index=None, tol=1e-12, normalize=False):
    ao_param_file  = os.path.join(dirpath, "ao_params.txt")
    ao_k_file = os.path.join(dirpath, "ao_k.txt")
    
    rows = []
    with open(ao_param_file, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            ig, l, rcut, RMAX, QMAX, NQ = s.split()[:6]
            ig = int(ig); l = int(l); rcut = float(rcut); RMAX = float(RMAX); QMAX = float(QMAX); NQ = int(NQ)
            if (species_index is None) or (ig == species_index):
                rows.append((ig, l, rcut, RMAX, QMAX, NQ))
    ig_info = {ig: (l, rcut, RMAX, QMAX, NQ) for ig, l, rcut, RMAX, QMAX, NQ in rows}
    ig_set = set(ig_info.keys())

    blocks = {}  # ig -> dict(l, rcut, QMAX, q, fq)
    with open(ao_k_file, "r") as f:
        ig, l, rcut, QMAX = None, None, None, None
        q_list, fq_list = [], []
        for line in f:
            s = line.strip()
            if s.startswith("BEGIN_IG"):
                ig = int(s.split()[-1])
                l = None; rcut = None; QMAX = None
                q_list = []; fq_list = []
            elif s.startswith("END_IG"):
                if ig in ig_set:
                    blocks[ig] = {
                        "l": l,
                        "rcut": rcut,
                        "QMAX": QMAX,
                        "q": np.array(q_list, float),
                        "fq": np.array(fq_list, float),
                    }
                ig = None
            elif ig is not None:
                if s.startswith("l ="):
                    l = int(s.split()[-1])
                elif "rcut" in s:
                    rcut = float(s.split()[-1])
                elif "QMAX" in s:
                    QMAX = float(s.split()[-1])
                else:
                    parts = s.split()
                    if len(parts) == 2:
                        try:
                            q_list.append(float(parts[0]))
                            fq_list.append(float(parts[1]))
                        except:
                            pass

    any_ig = next(iter(blocks))
    q_ref = blocks[any_ig]["q"]
    for ig_b, b in blocks.items():
        if len(b["q"]) != len(q_ref) or not np.allclose(b["q"], q_ref, rtol=0, atol=0):
            raise ValueError("qgrids in projectors_k.txt are not consistent!")
    
    groups = []
    for ig, l, rcut, RMAX, QMAX, NQ in sorted(rows, key=lambda x: x[0]):
        b = blocks[ig]
        fq = b["fq"]
        hit = None
        for g in groups:
            if g["l"] != l:
                continue
            if g["rcut"] != rcut:
                continue
            if np.allclose(g["fq"], fq, rtol=0, atol=tol):
                hit = g; break
        if hit is None:
            groups.append({
                "l": l, "rcut": rcut, "QMAX": QMAX, "ig_list": [ig],
                "q": q_ref, "fq": fq.copy(),
            })
        else:
            hit["ig_list"].append(ig)

    phiQ_list = []
    qgrid = LinearRGD.from_explicit_grid(q_ref)
    for g in groups:
        l = g["l"]; fq = g["fq"].copy()
        if normalize:
            m = np.abs(fq) > 1e-12
            k = q_ref[m]
            if k.size > 1:
                # Simpson on LinearRGD: Δk = q_ref[1]-q_ref[0]
                # N = ∫ (k^2/(2π^2)) |F(k)|^2 dk
                dk = q_ref[1] - q_ref[0]
                N = np.sum((k**2) * (np.abs(fq[m])**2)) * dk / (2*np.pi**2)
                if N > 0.0:
                    fq /= np.sqrt(N)
        phiQgrid = GridFunc(qgrid, fq, l=l)
        phiQ_list.append(phiQgrid)
    
    return phiQ_list, qgrid

def read_siesta_ao(dirpath, species_index=None, tol=1e-12, normalize=False):
    ao_param_file  = os.path.join(dirpath, "ao_params.txt")
    ao_r_file = os.path.join(dirpath, "ao_r.txt")
    
    rows = []
    with open(ao_param_file, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            ig, l, rcut, RMAX, QMAX, NQ = s.split()[:6]
            ig = int(ig); l = int(l); rcut = float(rcut); RMAX = float(RMAX); QMAX = float(QMAX); NQ = int(NQ)
            if (species_index is None) or (ig == species_index):
                rows.append((ig, l, rcut, RMAX, QMAX, NQ))
    ig_info = {ig: (l, rcut, RMAX, QMAX, NQ) for ig, l, rcut, RMAX, QMAX, NQ in rows}
    ig_set = set(ig_info.keys())

    blocks = {}  # ig -> dict(l, rcut, QMAX, q, fq)
    with open(ao_r_file, "r") as f:
        ig, l, rcut, RMAX = None, None, None, None
        r_list, fr_list = [], []
        for line in f:
            s = line.strip()
            if s.startswith("BEGIN_IG"):
                ig = int(s.split()[-1])
                l = None; rcut = None; RMAX = None
                r_list = []; fr_list = []
            elif s.startswith("END_IG"):
                if ig in ig_set:
                    blocks[ig] = {
                        "l": l,
                        "rcut": rcut,
                        "RMAX": RMAX,
                        "r": np.array(r_list, float),
                        "fr": np.array(fr_list, float),
                    }
                ig = None
            elif ig is not None:
                if s.startswith("l ="):
                    l = int(s.split()[-1])
                elif "rcut" in s:
                    rcut = float(s.split()[-1])
                elif "RMAX" in s:
                    RMAX = float(s.split()[-1])
                else:
                    parts = s.split()
                    if len(parts) == 2:
                        try:
                            r_list.append(float(parts[0]))
                            fr_list.append(float(parts[1]))
                        except:
                            pass

    any_ig = next(iter(blocks))
    r_ref = blocks[any_ig]["r"]
    for ig_b, b in blocks.items():
        if len(b["r"]) != len(r_ref) or not np.allclose(b["r"], r_ref, rtol=0, atol=0):
            raise ValueError("rgrids in ao_r.txt are not consistent!")

    groups = []
    for ig, l, rcut, RMAX, QMAX, NQ in sorted(rows, key=lambda x: x[0]):
        b = blocks[ig]
        fr = b["fr"]
        hit = None
        for g in groups:
            if g["l"] != l:
                continue
            if g["rcut"] != rcut:
                continue
            if np.allclose(g["fr"], fr, rtol=0, atol=tol):
                hit = g; break
        if hit is None:
            groups.append({
                "l": l, "rcut": rcut, "RMAX": RMAX, "ig_list": [ig],
                "r": r_ref, "fr": fr.copy(),
            })
        else:
            hit["ig_list"].append(ig)

    phiR_list, grad_phiR_list = [], []
    rgrid = LinearRGD.from_explicit_grid(r_ref)
    for g in groups:
        l = g["l"]; rcut = g["rcut"]; fr = g["fr"].copy()
        if normalize:
            m = np.abs(fr) > 1e-12
            k = r_ref[m]
            if k.size > 1:
                # Simpson on LinearRGD: Δk = r_ref[1]-r_ref[0]
                # N = ∫ (k^2/(2π^2)) |F(k)|^2 dk
                dk = r_ref[1] - r_ref[0]
                N = np.sum((k**2) * (np.abs(fr[m])**2)) * dk / (2*np.pi**2)
                if N > 0.0:
                    fr /= np.sqrt(N)
        phiRgrid = GridFunc(rgrid, fr, l=l, rcut=rcut)
        phiR_list.append(phiRgrid)
        grad_fr = copy.deepcopy(fr)
        grad_fr[1:] = fr[1:] / np.power(rgrid.rfunc[1:], l)
        grad_phiRgrid = GridFunc(rgrid, grad_fr, l=l, rcut=rcut)
        grad_phiR_list.append(grad_phiRgrid)

    return phiR_list, grad_phiR_list