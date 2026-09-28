import numpy as np
import sys
import sisl
import h5py

from .bgwio import bgw_vsc
from .lcaodata import LCAOData
from .utils import slice_same, tqdm_mpi_tofile, distrib_vec, is_master, comm, MPI
from .matlcao import pairs_to_indices, indices_to_pairs, MatLCAO
from .gridintg import GridPoints
from .mathutils import r_to_xyz
from .test_orbital_gradients import _test_orbital_gradients
from .constants import bohr2ang, hartree2ev


'''
This module implements several functions needed by real-space construction of AO Hamiltonian.
This includes constructing Hamiltonian in real space, and constructing VKB under AO basis.
'''


def read_vloc(filename, interface):
    if interface == 'bgw':
        vscread = bgw_vsc(filename)
        vscread.read_header()
        vscread.read_data()
        vscread.close()
        FFTgrid = np.array([vscread.nr1, vscread.nr2, vscread.nr3])
        vscg_full = np.zeros(FFTgrid, dtype='c16')
        g_g_full = np.mod(vscread.g_g, FFTgrid)
        vscg_full[g_g_full[:, 0], g_g_full[:, 1], g_g_full[:, 2]] = vscread.vscg
        del g_g_full
        del vscread

        vlocr_complex = np.fft.ifftn(vscg_full, s=FFTgrid, norm='forward')
        del vscg_full
        assert np.max(np.abs(vlocr_complex.imag)) < 1e-6
        vlocr = np.array(vlocr_complex.real, dtype=np.float64, order='C', copy=True)
        del vlocr_complex
    else:
        raise NotImplementedError(f'Unknown vloc interface: {interface}')
    return vlocr


def read_vloc_shared(filename, interface):
    """
    Read the local Vsc and share it among ranks per computing node.
    Create an MPI shared-memory window so all ranks view the same buffer.  

    Returns
    -------
    vlocr : ndarray
        Node-shared real-space local Vsc.
    resource : tuple or None
        The tuple (win, node_comm) must be alive as long as vlocr is used.  
        None is returned for serial case.
    """
    if comm is None or MPI is None or comm.size == 1:
        return read_vloc(filename, interface), None

    node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED, 0, MPI.INFO_NULL)
    node_rank = node_comm.rank
    if node_rank == 0:
        vloc_data = read_vloc(filename, interface)
        meta_data = (tuple(vloc_data.shape), vloc_data.dtype.str)
    else:
        vloc_data = None
        meta_data = None
    
    vloc_shape, vloc_dtype_str = node_comm.bcast(meta_data, root=0)
    vloc_dtype = np.dtype(vloc_dtype_str)
    nbytes = int(np.prod(vloc_shape, dtype=np.int64)) * vloc_dtype.itemsize
    window = MPI.Win.Allocate_shared(
        nbytes if node_rank == 0 else 0,
        vloc_dtype.itemsize,
        info=MPI.INFO_NULL,
        comm=node_comm
    )
    buf, disp_unit = window.Shared_query(0)
    assert disp_unit == vloc_dtype.itemsize
    vloc_shared = np.ndarray(shape=vloc_shape, dtype=vloc_dtype, buffer=buf)

    if node_rank == 0:
        np.copyto(vloc_shared, vloc_data)
        del vloc_data
    node_comm.Barrier()

    return vloc_shared, (window, node_comm)


def read_vloc_siesta(vscdir):
    S = sisl.get_sile(vscdir)
    try:
        G = S.read_grid(spin=[0.5, 0.5])
    except Exception:
        G = S.read_grid(spin=0)

    vlocr_eV = np.array(G.grid, dtype=np.float64, order='C', copy=False)
    eV_to_Ha = 1.0 / 27.211386245988  
    vlocr = vlocr_eV * eV_to_Ha
    assert vlocr.ndim == 3
    
    h5file = h5py.File(f'./vlocr_siesta.h5', 'w', libver='latest')
    h5file['vlocr_siesta'] = vlocr
    h5file.close()
    return vlocr


def read_hrr(structure, pspdir, funchfile=None, interface='qe'):
    if interface == 'qe':
        assert funchfile is None
        projR = LCAOData(structure, None, basis_path_root=pspdir, aocode='qe-projR')
        funch = []
        for zatm in structure.atomic_numbers:
            funch.append(projR.funch_spc[zatm])
        funcg = None
    else:
        raise NotImplementedError(f'Unknown vnloc interface: {interface}')    

    return funch, funcg, projR


def construct_Hloc(item, vlocr, basis, FFTgrid, rprimFFT, votk, grids_site, Hloc):
    '''
    Build Hamiltonian operator in atomic orbital basis according to the formula:
    H_{i\alpha,j\beta} = \langle \phi_{i\alpha} | -\frac{1}{2}\nabla^2 | \phi_{j\beta} \rangle + \int \mathrm{d}^3r\, \phi_{i\alpha}^*(\boldsymbol{r}) V_\text{eff}(\boldsymbol{r}) \phi_{j\beta}(\boldsymbol{r}) + \sum_{a\gamma\delta} \langle \phi_{i\alpha} | p_{a\gamma} \rangle D_{a\gamma\delta} \langle p_{a\delta} | \phi_{j\beta} \rangle
    '''

    npair = Hloc.npairs
    for ipair in tqdm_mpi_tofile(range(npair), total=npair):
        atm1, atm2 = Hloc.atom_pairs[ipair]
        spc1 = item.structure.atomic_numbers[atm1]
        spc2 = item.structure.atomic_numbers[atm2]

        for iorb in range(basis.norb_spc[spc1]):
            phirgrid1 = basis.phirgrids_spc[spc1][iorb]
            grad_phirgrid1 = basis.grad_phirgrids_spc[spc1][iorb]
            grid1 = grids_site[atm1][iorb]

            for jorb in range(basis.norb_spc[spc2]):
                phirgrid2 = basis.phirgrids_spc[spc2][jorb]
                grad_phirgrid2 = basis.grad_phirgrids_spc[spc2][jorb]
                slice1 = slice(basis.orbslices_spc[spc1][iorb],
                               basis.orbslices_spc[spc1][iorb+1])
                slice2 = slice(basis.orbslices_spc[spc2][jorb],
                               basis.orbslices_spc[spc2][jorb+1])
                grid2 = grids_site[atm2][jorb].translate(Hloc.translations[ipair]*FFTgrid)

                # plsgrid stores overlapping region of grid1 and grid2
                plsgrid = GridPoints.pls(grid1, grid2)
                if plsgrid.null():
                    Hloc.mats[ipair][slice1, slice2] = 0.
                    Hloc.mats_dphiVphi[ipair][slice1, slice2, :] = 0.
                    Hloc.mats_phiVdphi[ipair][slice1, slice2, :] = 0.
                    continue

                # plslcd is (N, 3) array, where N is the number of overlapping grid points,
                # each row is the point index (ix, iy, iz) in the global grid 
                plslcd = plsgrid.lcd()
                assert plslcd.shape[0]>0
                assert len(plslcd.shape)==2
                # transform point index to cartesian coordinates
                plscrt = plslcd @ rprimFFT
                
                Rvec_atm1 = (plscrt - item.structure.atomic_positions_cart[atm1]).reshape(-1, 3)
                Rnorm, x, y, z = r_to_xyz(Rvec_atm1)
                phi1 = phirgrid1.generate3D_norm(Rnorm, x, y, z)
                grad_phi1 = grad_phirgrid1.generate3D_grad_norm(Rnorm, x, y, z)

                Rvec_atm2 = (plscrt - item.structure.atomic_positions_cart[atm2] - 
                             Hloc.translations[ipair] @ item.structure.rprim).reshape(-1, 3)
                Rnorm, x, y, z = r_to_xyz(Rvec_atm2)
                phi2 = phirgrid2.generate3D_norm(Rnorm, x, y, z)
                grad_phi2 = grad_phirgrid2.generate3D_grad_norm(Rnorm, x, y, z)

                plslcd_uc = np.mod(plslcd, FFTgrid[None, :])
                x_uc, y_uc, z_uc = plslcd_uc[:, 0], plslcd_uc[:, 1], plslcd_uc[:, 2]
                f2 = vlocr[x_uc, y_uc, z_uc]
                mat = np.einsum('n,ni,nj->ij', f2, phi1, phi2, optimize=True)
                mat_dphiVphi = np.einsum('n,nik,nj->ijk', f2, grad_phi1, phi2, optimize=True)
                mat_phiVdphi = np.einsum('n,ni,njk->ijk', f2, phi1, grad_phi2, optimize=True)
                Hloc.mats[ipair][slice1, slice2] = mat * votk
                Hloc.mats_dphiVphi[ipair][slice1, slice2, :] = mat_dphiVphi * votk
                Hloc.mats_phiVdphi[ipair][slice1, slice2, :] = mat_phiVdphi * votk


def calc_vkb(olp_proj_ao, Dij=None):
    '''
    Construct VKB in atomic orbital basis according to the formula:
    <ia| Vkb |i'a'> = \sum_{jbb'} <ia|jb> D_{jbb'} <jb'|i'a'>
    where i, i', j are atom indices, a, a', b, b' are orbital indices

    Matrix D is optional. If not given, output will be all zero.
    '''
    if Dij is not None:
        for iD in range(len(Dij)):
            D = Dij[iD]
            if not np.isrealobj(D):
                # Future: D is complex
                assert np.max(np.abs(D.imag)) < 1e-8
                Dij[iD] = D.real
    olp_proj_ao.sort_atom1()
    translations = olp_proj_ao.translations
    atom_pairs = olp_proj_ao.atom_pairs
    trans1, atms2, mats3 = [], [], []
    
    # do matrix multiplications
    slice_jatm = slice_same(atom_pairs[:, 0])
    njatm = len(slice_jatm) - 1
    for ix_atm in range(njatm):
        startj = slice_jatm[ix_atm]
        endj = slice_jatm[ix_atm + 1]
        atomj = atom_pairs[startj, 0]
        ix_js, ix_jps = np.tril_indices(endj - startj) # lower-triangle indices
        ix_js += startj; ix_jps += startj
        # ix_js, ix_jps = np.meshgrid(np.arange(startj, endj, 1), np.arange(startj, endj, 1), indexing='ij')
        # ix_js = ix_js.reshape(-1); ix_jps = ix_jps.reshape(-1)
        trans1.append(translations[ix_jps] - translations[ix_js])
        atms2.append(np.stack((atom_pairs[ix_js, 1], atom_pairs[ix_jps, 1]), axis=1))
        for ix_j, ix_jp in zip(ix_js, ix_jps):
            mat = olp_proj_ao.mats[ix_j]
            matp = olp_proj_ao.mats[ix_jp]
            if Dij is not None:
                D = Dij[atomj]
                mats3.append(mat @ D @ matp.T)
            else:
                # mats_new.append(mat.T @ matp)
                mats3.append(np.zeros((mat.shape[1], matp.shape[1])))
    trans1 = np.concatenate(trans1, axis=0)
    atms2 = np.concatenate(atms2, axis=0)
    
    # collect terms with the same translations and atom pairs and sum them up
    xds1 = pairs_to_indices(olp_proj_ao.structure, trans1, atms2)
    slice3 = slice_same(xds1)
    y = len(slice3) - 1
    mats2 = []
    for ipair in range(y):
        slice2 = slice(slice3[ipair], slice3[ipair+1])
        mats2.append(np.sum([mats3[i] for i in slice2], axis=0))
    indx1 = np.unique(xds1)
    trans2, atm1 = indices_to_pairs(olp_proj_ao.structure.natom, indx1)
    
    vkb = MatLCAO(olp_proj_ao.structure, trans2, atm1, mats2, olp_proj_ao.lcaodata2)
    
    return vkb