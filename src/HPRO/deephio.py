import os
import json
import numpy as np
import h5py
from scipy.linalg import block_diag

from .constants import bohr2ang, hartree2ev
from .structure import Structure
from .matlcao import MatLCAO, pairs_to_indices
from .lcaodata import LCAOData # this might be unsafe?

'''
This module implemements several functions for reading and writing files in deeph format
'''

def save_structure_deeph(structure, savedir):

    os.makedirs(savedir, exist_ok=True)

    rprim = structure.rprim
    gprim = structure.gprim
    atom_nbrs = structure.atomic_numbers
    atm_pos_cart = structure.atomic_positions_cart
    efermi = structure.efermi
    
    np.savetxt(f'{savedir}/lat.dat', rprim.T * bohr2ang)
    np.savetxt(f'{savedir}/element.dat', atom_nbrs, fmt='%-3i')
    np.savetxt(f'{savedir}/site_positions.dat', atm_pos_cart.T * bohr2ang)
    np.savetxt(f'{savedir}/rlat.dat', 2*np.pi*gprim.T / bohr2ang)

    info = {"isspinful": False, "fermi_level": efermi * hartree2ev}
    with open(f'{savedir}/info.json', 'w') as f:
        json.dump(info, f)


Us_openmx2wiki = {
    0: np.eye(1),
    1: np.eye(3)[[1, 2, 0]],
    2: np.eye(5)[[2, 4, 0, 3, 1]],
    3: np.eye(7)[[6, 4, 2, 0, 1, 3, 5]]
}


def get_Us_openmx2wiki(ls_spc):
    '''
    DeepH follows the OpenMX definition of spherical harmonics, but this software follows Wikipedia's convention.
    So we need to convert them.
    '''
    orbitals_Us_openmx2wiki = {}
    for spc, orbital_types in ls_spc.items():
        U2deeph = [Us_openmx2wiki[l] for l in orbital_types]
        orbitals_Us_openmx2wiki[spc] = block_diag(*U2deeph)
    return orbitals_Us_openmx2wiki


def save_mat_deeph(savedir, matlcao, filename='hamiltonians.h5', energy_unit=True):
    lcaodata = matlcao.lcaodata1
    os.makedirs(savedir, exist_ok=True)
    atom_nbrs = lcaodata.structure.atomic_numbers
    ls_spc = lcaodata.ls_spc
    with open(f'{savedir}/orbital_types.dat', 'w') as f:
        for nspc in atom_nbrs:
            f.write(' '.join(map(str, ls_spc[nspc])))
            f.write('\n')
    
    # here real spherical harmonics follow wikipedia convention
    # need to convert to openmx convension
    orbitals_Us_openmx2wiki = get_Us_openmx2wiki(ls_spc)
    h5file = h5py.File(f'{savedir}/{filename}', 'w', libver='latest')

    for ipair in range(matlcao.npairs):
        spc1 = atom_nbrs[matlcao.atom_pairs[ipair, 0]]
        spc2 = atom_nbrs[matlcao.atom_pairs[ipair, 1]]
        key = matlcao.get_keystr(ipair)
        U1 = orbitals_Us_openmx2wiki[spc1]
        U2 = orbitals_Us_openmx2wiki[spc2]
        mat = U1.T @ matlcao.mats[ipair] @ U2
        if energy_unit:
            mat *= hartree2ev
        h5file[key] = mat
    h5file.close()


def save_pos_deeph(savedir, matlcao, filename='positions.h5'):
    lcaodata = matlcao.lcaodata1
    os.makedirs(savedir, exist_ok=True)
    atom_nbrs = lcaodata.structure.atomic_numbers
    ls_spc = lcaodata.ls_spc

    orbitals_Us_openmx2wiki = get_Us_openmx2wiki(ls_spc)
    h5file = h5py.File(f'{savedir}/{filename}', 'w', libver='latest')

    for ipair in range(matlcao.npairs):
        spc1 = atom_nbrs[matlcao.atom_pairs[ipair, 0]]
        spc2 = atom_nbrs[matlcao.atom_pairs[ipair, 1]]
        U1 = orbitals_Us_openmx2wiki[spc1]
        U2 = orbitals_Us_openmx2wiki[spc2]
        key = matlcao.get_keystr(ipair)
        mat = np.einsum(
            'ia,abk,bj->ijk', 
            U1.T, matlcao.mats[ipair], U2, 
            optimize=True
        )
        h5file[key] = mat
    h5file.close()


def save_phiVdphi_deeph(savedir, matlcao, filename='phiVdphi.h5', energy_unit=True):
    lcaodata = matlcao.lcaodata1
    os.makedirs(savedir, exist_ok=True)
    atom_nbrs = lcaodata.structure.atomic_numbers
    ls_spc = lcaodata.ls_spc
    
    orbitals_Us_openmx2wiki = get_Us_openmx2wiki(ls_spc)
    h5file = h5py.File(f'{savedir}/{filename}', 'w', libver='latest')

    for ipair in range(matlcao.npairs):
        spc1 = atom_nbrs[matlcao.atom_pairs[ipair, 0]]
        spc2 = atom_nbrs[matlcao.atom_pairs[ipair, 1]]
        U1 = orbitals_Us_openmx2wiki[spc1]
        U2 = orbitals_Us_openmx2wiki[spc2]
        key = matlcao.get_keystr(ipair)
        mat = np.einsum(
            'ia,abk,bj->ijk', 
            U1.T, matlcao.mats_phiVdphi[ipair], U2,
            optimize=True
        )
        if energy_unit:
            mat *= hartree2ev / bohr2ang
        h5file[key] = mat
    h5file.close()


def load_deeph_HS(folder, filename, energy_unit=True):
    stru = Structure.from_deeph(folder)
    lcaodata = LCAOData(stru, None, basis_path_root=folder, aocode='deeph')

    orbitals_Us_openmx2wiki = get_Us_openmx2wiki(lcaodata.ls_spc)
    
    hoppings = []
    translations = []
    atom_pairs = []
    npairs = 0
    with h5py.File(f'{folder}/{filename}') as f:
        for k, v in f.items():
            npairs += 1
            Rijab = eval(k)
            translations.append(Rijab[:3])
            atom_pairs.append(np.array(Rijab[3:5]) - 1)

            spc1 = stru.atomic_numbers[atom_pairs[-1][0]]
            spc2 = stru.atomic_numbers[atom_pairs[-1][1]]
            hmat = np.array(v)
            hmat = orbitals_Us_openmx2wiki[spc1] @ hmat @ orbitals_Us_openmx2wiki[spc2].T
            if energy_unit: hmat /= hartree2ev
            hoppings.append(hmat)
    translations = np.array(translations)
    atom_pairs = np.array(atom_pairs)
    npairs = len(atom_pairs)
    return MatLCAO(stru, translations, atom_pairs, hoppings, lcaodata)


def transform_mat_block(mat, U1, U2, energy_unit=True):
    out = U1.T @ mat @ U2
    if energy_unit:
        out *= hartree2ev
    return out


def transform_grad_block(mat, U1, U2, energy_unit=True):
    out = np.einsum('ia,abk,bj->ijk', U1.T, mat, U2, optimize=True)
    if energy_unit:
        out *= hartree2ev / bohr2ang
    return out


def save_Hloc_shard(savedir, Hloc, pairs_idx, rank):
    """
    Write one rank-local Hloc shard without gathering dense blocks.
    """
    save_path = os.path.join(
        savedir, '.ham_tmp', f'Hloc_rank{int(rank):04d}.h5'
    )

    lcaodata = Hloc.lcaodata1
    stru = Hloc.structure
    atom_nbrs = stru.atomic_numbers
    Us = get_Us_openmx2wiki(lcaodata.ls_spc)
    pairs_set = set(int(x) for x in np.asarray(pairs_idx).ravel())

    translations_inv = -Hloc.translations
    atom_pairs_inv = Hloc.atom_pairs[:, [1, 0]]
    indices_inv = pairs_to_indices(stru, translations_inv, atom_pairs_inv)

    with h5py.File(save_path, 'w', libver='latest') as f:
        gh = f.create_group('Hloc')
        gg = f.create_group('phiVdphi')

        for ipair in range(Hloc.npairs):
            ia, ja = Hloc.atom_pairs[ipair]
            spc1, spc2 = atom_nbrs[ia], atom_nbrs[ja]
            U1, U2 = Us[spc1], Us[spc2]
            key = Hloc.get_keystr(ipair)
            gh[key] = transform_mat_block(
                Hloc.mats[ipair], U1, U2, 
                energy_unit=True
            )
            gg[key] = transform_grad_block(
                Hloc.mats_phiVdphi[ipair], U1, U2, 
                energy_unit=True
            )

            # duplicate Hermitian partners without allocating blocks in memory
            if int(indices_inv[ipair]) not in pairs_set:
                Rijab = (
                    translations_inv[ipair].tolist()
                    + (atom_pairs_inv[ipair] + 1).tolist()
                )
                key_inv = str(Rijab)
                gh[key_inv] = transform_mat_block(
                    Hloc.mats[ipair].T, U2, U1, 
                    energy_unit=True
                )
                grad_inv = np.swapaxes(
                    Hloc.mats_dphiVphi[ipair], 0, 1
                )
                gg[key_inv] = transform_grad_block(
                    grad_inv, U2, U1, 
                    energy_unit=True
                )


def merge_Hloc_shard(savedir, nranks):
    """
    Merge rank-local Hloc shards using only one dense block at a time.
    Returns a temporary full Hloc file in DeepH convention.
    """
    tmpdir = os.path.join(savedir, '.ham_tmp')
    Hloc_path = os.path.join(tmpdir, 'Hloc.h5')
    grad_path = os.path.join(savedir, 'phiVdphi.h5')

    with h5py.File(Hloc_path, 'w', libver='latest') as hout, \
         h5py.File(grad_path, 'w', libver='latest') as gout:
        for rank in range(nranks):
            shard = os.path.join(tmpdir, f'Hloc_rank{rank:04d}.h5')
            if not os.path.exists(shard):
                raise FileNotFoundError(f'Missing Hloc shard: {shard}')
            with h5py.File(shard, 'r') as src:
                for key in src['Hloc'].keys():
                    if key in hout:
                        raise RuntimeError(f'Duplicate distributed Hloc key: {key}')
                    hout[key] = src['Hloc'][key][...]
                for key in src['phiVdphi'].keys():
                    if key in gout:
                        raise RuntimeError(f'Duplicate distributed grad key: {key}')
                    gout[key] = src['phiVdphi'][key][...]
            os.remove(shard)


def assemble_ham(savedir, lcaodata, final_pairs=None, filename='hamiltonians.h5'):
    """Stream the final H = Hmain + Hkin + Hkb directly between HDF5 files.

    All three source files are already in eV and OpenMX/DeepH orbital
    convention.  Only one AO block is resident while summing.  If
    ``final_pairs`` is supplied (the user cutoff case), exactly that pair set is
    written and missing source blocks remain zero, matching MatLCAO.convert_to.
    """
    Hloc_path = os.path.join(savedir, '.ham_tmp', 'Hloc.h5')
    Hkin_path = os.path.join(savedir, 'kinetics.h5')
    Hkb_path = os.path.join(savedir, 'Hkb.h5')
    out_path = os.path.join(savedir, filename)

    atom_nbrs = lcaodata.structure.atomic_numbers
    with open(os.path.join(savedir, 'orbital_types.dat'), 'w') as f_orb:
        for nspc in atom_nbrs:
            f_orb.write(' '.join(map(str, lcaodata.ls_spc[nspc])))
            f_orb.write('\n')

    with h5py.File(Hloc_path, 'r') as floc, \
         h5py.File(Hkin_path, 'r') as fkin, \
         h5py.File(Hkb_path, 'r') as fkb, \
         h5py.File(out_path, 'w', libver='latest') as fout:
        
        if final_pairs is None:
            keys = sorted(set(floc.keys()) | set(fkin.keys()) | set(fkb.keys()))
            pair_meta = None
        else:
            keys = [final_pairs.get_keystr(i) for i in range(final_pairs.npairs)]
            pair_meta = final_pairs

        for kk, key in enumerate(keys):
            total = None
            for src in (floc, fkin, fkb):
                if key in src:
                    block = src[key][...]
                    if total is None:
                        total = block
                    else:
                        total += block
            
            if total is None:
                assert pair_meta is not None
                ia, ja = pair_meta.atom_pairs[kk]
                spc1, spc2 = atom_nbrs[ia], atom_nbrs[ja]
                norb1 = lcaodata.orbslices_spc[spc1][-1]
                norb2 = lcaodata.orbslices_spc[spc2][-1]
                total = np.zeros((norb1, norb2), dtype=np.float64)

            fout[key] = total

    os.remove(Hloc_path)
    os.rmdir(os.path.dirname(Hloc_path))