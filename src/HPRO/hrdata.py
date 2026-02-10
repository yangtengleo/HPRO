import numpy as np
import sys
import sisl
import h5py

from .bgwio import bgw_vsc
from .lcaodata import LCAOData
from .utils import slice_same, tqdm_mpi_tofile
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
        # if is_master:
        #     print('Reading self-consistent potential from VSC')
        vscread.read_data()
        vscread.close()
        FFTgrid = np.array([vscread.nr1, vscread.nr2, vscread.nr3])
        vscg_full = np.zeros(FFTgrid, dtype='c16')
        _, g_g_full = np.divmod(vscread.g_g, FFTgrid)
        vscg_full[g_g_full[:, 0], g_g_full[:, 1], g_g_full[:, 2]] = vscread.vscg
        vlocr = np.fft.ifftn(vscg_full, s=FFTgrid, norm='forward')
        assert np.max(np.abs(vlocr.imag)) < 1e-6
        vlocr = vlocr.real
    else:
        raise NotImplementedError(f'Unknown vloc interface: {interface}')
    print('Real space grid dimensions: (' + ' '.join(f'{vlocr.shape[i]:5d}' for i in range(3)) + ')\n')
    '''
    h5file = h5py.File(f'./vlocr_qe.h5', 'w', libver='latest')
    h5file['vlocr_qe'] = vlocr
    h5file.close()
    '''
    return vlocr

def read_vloc_siesta(vscdir):
    # Implement the reading of vloc for Siesta here
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

def constructH(item, vlocr, basis, FFTgrid, rprimFFT, votk, grids_site_orb, Hmain):
    '''
    Build Hamiltonian operator in atomic orbital basis according to the formula:
    H_{i\alpha,j\beta} = \langle \phi_{i\alpha} | -\frac{1}{2}\nabla^2 | \phi_{j\beta} \rangle + \int \mathrm{d}^3r\, \phi_{i\alpha}^*(\boldsymbol{r}) V_\text{eff}(\boldsymbol{r}) \phi_{j\beta}(\boldsymbol{r}) + \sum_{a\gamma\delta} \langle \phi_{i\alpha} | p_{a\gamma} \rangle D_{a\gamma\delta} \langle p_{a\delta} | \phi_{j\beta} \rangle
    '''
    print(f'\nConstructing Hamiltonian operator with {Hmain.npairs} blocks')
    '''
    # ======== test paramters ============
    TEST_T = np.array([-1, -1, 0], dtype=int) 
    TEST_ATOM_PAIRS = (1, 1)
    TEST_CHUNK = 100000
    Nx, Ny, Nz = map(int, FFTgrid)
    Ntot = Nx * Ny * Nz
    a1 = Nx * rprimFFT[0, :]
    a2 = Ny * rprimFFT[1, :]
    a3 = Nz * rprimFFT[2, :]
    L  = np.column_stack([a1, a2, a3])
    Linv = np.linalg.inv(L)
    # ====================================

    def cell_aabb(origin, a1, a2, a3):
        lo = origin + np.minimum(a1, 0.0) + np.minimum(a2, 0.0) + np.minimum(a3, 0.0)
        hi = origin + np.maximum(a1, 0.0) + np.maximum(a2, 0.0) + np.maximum(a3, 0.0)
        return lo, hi

    def dist_point_to_aabb(p, lo, hi):
        d = np.maximum(0.0, np.maximum(lo - p, p - hi))
        return float(np.linalg.norm(d))
    '''
    for ipair in tqdm_mpi_tofile(range(Hmain.npairs)):
        atm1, atm2 = Hmain.atom_pairs[ipair]
        spc1, spc2 = item.structure.atomic_numbers[atm1], item.structure.atomic_numbers[atm2]
        '''
        for i in range(7):
            print(f'test gradient for orbital {i}', flush=True)
            test_phirgrid = basis.phirgrids_spc[spc1][i]
            test_grad_phirgrid = basis.grad_phirgrids_spc[spc1][i]
            _test_orbital_gradients(test_phirgrid, test_grad_phirgrid)
            print('\n')
        '''
        '''
        # ====== test SIESTA-like per-grid integration =======
        if np.all(Hmain.translations[ipair] == TEST_T) and (atm1+1, atm2+1) == TEST_ATOM_PAIRS: 
            Rpos1 = item.structure.atomic_positions_cart[atm1]
            Rpos2 = item.structure.atomic_positions_cart[atm2] + TEST_T @ item.structure.rprim
            rcut2_max = max(basis.phirgrids_spc[spc2][j].rgd.rend for j in range(basis.norb_spc[spc2]))
            for iorb in range(basis.norb_spc[spc1]):
                phirgrid1 = basis.phirgrids_spc[spc1][iorb]
                grad_phirgrid1 = basis.grad_phirgrids_spc[spc1][iorb]
                slice1 = slice(basis.orbslices_spc[spc1][iorb], basis.orbslices_spc[spc1][iorb+1])
                norb1 = slice1.stop - slice1.start
                rcut1 = phirgrid1.rgd.rend
                rcut1_sq = rcut1 * rcut1
                # estimate neighbor cells that need to be looped over
                Rmax_envelope = rcut1 + rcut2_max
                n1 = int(np.ceil(Rmax_envelope / np.linalg.norm(a1))) + 1
                n2 = int(np.ceil(Rmax_envelope / np.linalg.norm(a2))) + 1
                n3 = int(np.ceil(Rmax_envelope / np.linalg.norm(a3))) + 1
                # build accumulator for each jorb
                jorb_info = []
                for jorb in range(basis.norb_spc[spc2]):
                    slice2 = slice(basis.orbslices_spc[spc2][jorb], basis.orbslices_spc[spc2][jorb+1])
                    norb2 = slice2.stop - slice2.start
                    jorb_info.append(dict(
                        j=jorb, slice2=slice2, norb2=norb2,
                        rcut2=basis.phirgrids_spc[spc2][jorb].rgd.rend,
                        rcut2_sq=basis.phirgrids_spc[spc2][jorb].rgd.rend**2,
                        dphiVphi=np.zeros((norb1, norb2, 3), dtype=float),
                        phiVdphi=np.zeros((norb1, norb2, 3), dtype=float),
                    ))
                # loop over global grids chunk by chunk
                for start in range(0, Ntot, TEST_CHUNK):
                    stop = min(start + TEST_CHUNK, Ntot)
                    idx = np.arange(start, stop, dtype=np.int64)
                    ix = (idx % Nx).astype(np.int64)
                    iy = ((idx // Nx) % Ny).astype(np.int64)
                    iz = (idx // (Nx*Ny)).astype(np.int64)
                    lcd = np.stack([ix, iy, iz], axis=1).astype(float)
                    base_r = lcd @ rprimFFT
                    f2 = vlocr[ix, iy, iz]
                    # loop over translational neighbor cells
                    for sx in range(-n1, n1+1):
                        for sy in range(-n2, n2+1):
                            for sz in range(-n3, n3+1):
                                origin_S = sx*a1 + sy*a2 + sz*a3
                                loS, hiS = cell_aabb(origin_S, a1, a2, a3)
                                if dist_point_to_aabb(Rpos1, loS, hiS) > rcut1:
                                    continue
                                r = base_r + origin_S
                                # iorb cache: compute mask and phi1, grad_phi1
                                dr1 = r - Rpos1
                                dr1_sq = np.einsum('ni,ni->n', dr1, dr1)
                                mask1 = dr1_sq <= rcut1_sq
                                if not np.any(mask1):
                                    continue
                                Rnorm1, x1, y1, z1 = r_to_xyz(dr1[mask1])
                                phi1_mask = phirgrid1.generate3D_norm_check(Rnorm1, x1, y1, z1)
                                grad_phi1_mask = grad_phirgrid1.generate3D_grad_norm_check(Rnorm1, x1, y1, z1)
                                # share dr2 for all jorb
                                dr2 = r - Rpos2
                                dr2_sq = np.einsum('ni,ni->n', dr2, dr2)
                                dcell2 = dist_point_to_aabb(Rpos2, loS, hiS)
                                # second mask for all jorb and compute phi2, grad_phi2
                                for jorb in jorb_info:
                                    if dcell2 > jorb['rcut2']:
                                        continue
                                    mask2 = dr2_sq[mask1] <= jorb['rcut2_sq']
                                    if not np.any(mask2):
                                        continue
                                    dr2_mask_mask = dr2[mask1][mask2]
                                    f2_mask_mask = f2[mask1][mask2]
                                    phi1_mask_mask = phi1_mask[mask2, :]
                                    grad_phi1_mask_mask = grad_phi1_mask[mask2, :, :]
                                    Rnorm2, x2, y2, z2 = r_to_xyz(dr2_mask_mask)
                                    phirgrid2 = basis.phirgrids_spc[spc2][jorb['j']]
                                    grad_phirgrid2 = basis.grad_phirgrids_spc[spc2][jorb['j']]
                                    phi2_mask_mask = phirgrid2.generate3D_norm_check(Rnorm2, x2, y2, z2)
                                    grad_phi2_mask_mask = grad_phirgrid2.generate3D_grad_norm_check(Rnorm2, x2, y2, z2)
                                    jorb["dphiVphi"] += np.einsum('n,nik,nj->ijk', f2_mask_mask, grad_phi1_mask_mask, phi2_mask_mask, optimize=True)
                                    jorb["phiVdphi"] += np.einsum('n,ni,njk->ijk', f2_mask_mask, phi1_mask_mask, grad_phi2_mask_mask, optimize=True)
                for jorb in jorb_info:
                    Hmain.mats_dphiVphi[ipair][slice1, jorb["slice2"], :] = jorb["dphiVphi"] * votk
                    Hmain.mats_phiVdphi[ipair][slice1, jorb["slice2"], :] = jorb["phiVdphi"] * votk
            print("Finished processing test atom pairs", flush=True)
            continue
        else:
            Hmain.mats_dphiVphi[ipair][:, :, :] = 0.
            Hmain.mats_phiVdphi[ipair][:, :, :] = 0.
            continue
        # ====================================================
        '''
        for iorb in range(basis.norb_spc[spc1]):
            phirgrid1 = basis.phirgrids_spc[spc1][iorb]
            grad_phirgrid1 = basis.grad_phirgrids_spc[spc1][iorb]
            grid1 = grids_site_orb[atm1][iorb]
            for jorb in range(basis.norb_spc[spc2]):
                phirgrid2 = basis.phirgrids_spc[spc2][jorb]
                grad_phirgrid2 = basis.grad_phirgrids_spc[spc2][jorb]
                slice1 = slice(basis.orbslices_spc[spc1][iorb],
                            basis.orbslices_spc[spc1][iorb+1])
                slice2 = slice(basis.orbslices_spc[spc2][jorb],
                            basis.orbslices_spc[spc2][jorb+1])
                grid2 = grids_site_orb[atm2][jorb].translate(Hmain.translations[ipair]*FFTgrid)
                # plsgrid stores overlapping region of grid1 and grid2
                plsgrid = GridPoints.pls(grid1, grid2)
                if plsgrid.null():
                    Hmain.mats[ipair][slice1, slice2] = 0.
                    Hmain.mats_dphiVphi[ipair][slice1, slice2, :] = 0.
                    Hmain.mats_phiVdphi[ipair][slice1, slice2, :] = 0.
                    continue
                # plslcd is (N, 3) array, where N is the number of overlapping grid points,
                # each row is the point index (ix, iy, iz) in the global grid 
                plslcd = plsgrid.lcd()
                assert plslcd.shape[0]>0
                assert len(plslcd.shape)==2
                # transform point index to cartesian coordinates
                plscrt = plslcd @ rprimFFT
                # t.start()
                Rvec_atm1 = (plscrt - item.structure.atomic_positions_cart[atm1]).reshape(-1, 3)
                Rnorm, x, y, z = r_to_xyz(Rvec_atm1)
                phi1 = phirgrid1.generate3D_norm(Rnorm, x, y, z)
                grad_phi1 = grad_phirgrid1.generate3D_grad_norm(Rnorm, x, y, z)
                Rvec_atm2 = (plscrt - item.structure.atomic_positions_cart[atm2] - 
                             Hmain.translations[ipair] @ item.structure.rprim).reshape(-1, 3)
                Rnorm, x, y, z = r_to_xyz(Rvec_atm2)
                phi2 = phirgrid2.generate3D_norm(Rnorm, x, y, z)
                grad_phi2 = grad_phirgrid2.generate3D_grad_norm(Rnorm, x, y, z)
                # t.stop()
                plslcd_uc = np.divmod(plslcd, FFTgrid[None, :])[1]
                x_uc, y_uc, z_uc = plslcd_uc[:, 0], plslcd_uc[:, 1], plslcd_uc[:, 2]
                f2 = vlocr[x_uc, y_uc, z_uc]
                mat = np.einsum('n,ni,nj->ij', f2, phi1, phi2, optimize=True)
                mat_dphiVphi = np.einsum('n,nik,nj->ijk', f2, grad_phi1, phi2, optimize=True)
                mat_phiVdphi = np.einsum('n,ni,njk->ijk', f2, phi1, grad_phi2, optimize=True)
                Hmain.mats[ipair][slice1, slice2] = mat * votk
                Hmain.mats_dphiVphi[ipair][slice1, slice2, :] = mat_dphiVphi * votk
                Hmain.mats_phiVdphi[ipair][slice1, slice2, :] = mat_phiVdphi * votk

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