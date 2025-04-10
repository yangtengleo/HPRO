import time
import numpy as np

from .structure import Structure, load_structure
from .lcaodata import LCAOData
from .hrdata import read_hrr, read_vloc, constructH
from .deephio import save_structure_deeph, save_mat_deeph, get_mat0
from .utils import mpi_watch, simple_timer, is_master, comm, slice_same
from .twocenter import calc_overlap
from .orbutils import OrbPair
from .matlcao import pwc, MatLCAO, pairs_to_indices, indices_to_pairs
from .gridintg import GridPoints


class PW2AOkernel:
    
    @mpi_watch
    @simple_timer('IO done, total wall time = {t}')
    def __init__(self, 
                 lcao_interface=None, lcaodata_root=None,
                 wfn_interface=None, wfndata_root=None, wfn_max_band=None, 
                 hgdata_interface=None, hrdata_interface=None,
                 vscdir=None, vkbdir=None, vasp_workdir=None, gpawsave=None, gpaw_datadir=None, upfdir=None,
                 structure_interface=None, structure_path=None,
                 overlaps_only=False,
                 gw_root=None, eqp_option='eqp1',
                 kgrid=None, ecutwfn=None):
        '''
        Parameters:
        ---------

        lcao_interface:       Currently supports 'siesta' and 'gpaw'. LCAO data is the information for atomic orbital basis.
        lcaodata_root:        If lcao_interface='siesta', then this should be folder containing x.ion where x is the atomic symbol.
                              If lcao_interface='gpaw', then this should contain x.(...).basis, where x is the atomic symbol, and 
                              the contents in the bracket is optional.
        
        hrdata_interface:     Currently supports 'qe-bgw', 'vasp', 'gpaw'. HrData is the information for the Hamiltonian in real space.
                              ecutwfn will be needed.

            If hrdata_interface='qe-bgw':   you should specify vscdir and upfdir, where upfdir is the folder containing the QE
                                         pseudopotential file x.upf.
            If hrdata_interface='vasp':  you should specify vasp_workdir containing POSCAR, POTCAR, LOCPOT and VNLOC.
            If hrdata_interface='gpaw':  you should specify gpawsave which is the binary xxx.gpw file generated by a gpaw run, and 
                                         gpaw_datadir which contains the GPAW pseudopotential x.PBE(.gz) where x is the atomic symbol, 
                                         and .gz is optional.
            If hrdata_interface='qe-deephr: you should specify vscdir (Vtot.h5) and upfdir, where upfdir is the folder containing the QE
                                         pseudopotential file x.upf. You should also provide structure information separately.

        overlaps_only:        If this is True, only the overlap will be calculated but not the Hamiltonian. The user must also provide 
                              hrdata_interface and corresponding data. 

            If hrdata_interface='qe-bgw':   only upfdir is needed, where upfdir is the folder containing the QE
                                         pseudopotential file x.upf.
            If hrdata_interface='qe-deephr': same as 'qe-bgw'.
            If hrdata_interface='vasp':  you should specify vasp_workdir, but only POTCAR is needed.
            If hrdata_interface='gpaw':  Only gpaw_datadir is needed, which contains the GPAW pseudopotential x.PBE(.gz) where 
                                         x is the atomic symbol, and .gz is optional.
        
        structure_interface:  Usually, the structure information can be automatically determined if hgdata or hrdata is provided. However,
                              if neither of these are provided, or using overlaps_only, then you should specify an additional way for the
                              code to determine the material structure. Currently supports 'qe', 'bgw', 'vasp', 'gpaw', 'deeph'.
        structure_path:       If structure_interface='qe': the folder containing data-file-schema.xml
                              If structure_interface='bgw': could be WFN, VSC, VKB
                              If structure_interface='vasp': the POSCAR file
                              If structure_interface='gpaw': the binary xxx.gpw file
                              If structrue_interface='deeph': the folder containing lat.dat, site_positions.dat, element.dat, info.json
        
        gw_root:        Folder containing sigma_hp.log.
        eqp_option:     'eqp0' or 'eqp1'.

        kgrid:          use None to read k-grid and k-weights from input data (qe-xml or bgw binary header); 
                        or specify a grid (e.g. [3, 3, 1]) to overwrite the original k-grid.
        '''
        
        self.start_time = time.time()
        
        if is_master():
            print()
            print('==============================================================================')
            print('Program HPRO')
            print('Author: Xiaoxun Gong (xiaoxun.gong@gmail.com)')
            print('==============================================================================')
            print()
            
        if lcao_interface is not None:
            lcao_interface = lcao_interface.lower()
        
        # determine structure, and check corresopnding input
        structure = None
        def check_structure(structure, stru1):
            if structure is None:
                structure = stru1
            elif structure != stru1:
                raise ValueError('Structure mismatch')
            return structure
        if hrdata_interface is not None and not overlaps_only:
            if hrdata_interface == 'qe-bgw':
                assert vscdir is not None
                stru1 = Structure.from_bgw(vscdir)
            elif hrdata_interface == 'qe-deephr':
                assert vscdir is not None
                stru1 = None # later we read xml file to determine structure
            else:
                raise NotImplementedError(f'Unknown hrdata_interface {hrdata_interface}')
            structure = check_structure(structure, stru1)
        if structure_interface is not None:
            assert structure_path is not None
            stru1 = load_structure(structure_path, structure_interface)
            structure = check_structure(structure, stru1)
        
        assert structure is not None, 'Must provide structure information'
        if is_master():
            structure.echo_info()
            print()
            
        # get lcaodata
        if lcao_interface is not None:
            assert structure is not None, 'Must provide structure information for lcao data to be initialized'
            lcaodata = LCAOData(structure, basis_path_root=lcaodata_root, aocode=lcao_interface)
            lcaodata.check_rstart()
        else:
            raise NotImplementedError()

        if is_master():
            print('Atomic orbital basis:')
            lcaodata.echo_info()
            print()
        
        # get wfndata
        assert wfn_interface is None
        wfndata = None
        assert hgdata_interface is None
        hgdata = None
        assert not overlaps_only
        
        # get hrdata
        if hrdata_interface is not None:
            if not overlaps_only:
                if (hrdata_interface == 'qe-bgw') or (hrdata_interface == 'qe-deephr'):
                    assert vscdir is not None and upfdir is not None
                    vlocr = read_vloc(vscdir, hrdata_interface.split('-')[1]) # bgw or deephr
                    funch, funcg, projR = read_hrr(structure, upfdir, interface='qe')
                else:
                    raise NotImplementedError(f'Unknown hrdata_interface {hrdata_interface}')
            hrdata = (vlocr, funch, funcg, projR)
            if is_master():
                if vlocr is not None:
                    print('Real space grid dimensions: (' + ' '.join(f'{vlocr.shape[i]:5d}' for i in range(3)) + ')\n')
                print('Pseudopotential projectors:')
                projR.echo_info()
                print()
        else:
            hrdata = None

        self.structure = structure
        self.lcaodata = lcaodata
        self.wfndata = wfndata
        self.hgdata = hgdata
        self.ecutwfn = ecutwfn #! from file?
        self.hrdata = hrdata
        self.overlaps_only = overlaps_only
    
    @mpi_watch
    @simple_timer('\nJob done, total wall time = {t}\n')
    def run_pw2ao_rs(self, savedir, cutoffs=None):
        '''
        Convert plane-wave Hamiltonian to atomic orbital basis by integration in real space.

        Parameters:
        ---------
        savedir:         place where result will be saved
        cutoffs:         Dict[str -> float], cutoff radius for each atomic species, in bohr. This parameter is not required. If
                         not provided, then the code will decide which hoppings are nonzero by considering the cutoff of the
                         pseudopotential region and the cutoff of atomic orbitals.
        '''
        
        if is_master():
            if self.overlaps_only:
                print('\n============================')
                print('Calculating overlap matrices')
                print('============================\n')
                if (comm is not None) and (comm.size > 1):
                    print('WARNING: overlaps_only does not support parallelism yet, please run on only one CPU!\n')
            else:
                print('\n===============================================')
                print('Reconstructing PW Hamiltonian to AOs in real space')
                print('===============================================\n')

        assert self.ecutwfn is not None, 'Must provide cutoff energy ecutwfn'
        assert self.hrdata is not None
        vlocr, funch, funcg, projR = self.hrdata

        if is_master():
            save_structure_deeph(self.structure, savedir)

        ecut = self.ecutwfn
        basis = self.lcaodata
        basis.calc_phiQ(ecut * 1.1)
        projR.calc_phiQ(ecut * 1.1)

        # Get orbital pairs
        # orbpairs1 saves pairs of AO basis and AO basis
        # orbpairs2 saves pairs of projector basis and AO basis
        # orbpairs3 saves pairs of AO basis and AO basis, for kinetic energy
        orbpairs1, orbpairs2, orbpairs3 = {}, {}, {}
        stru = self.structure
        for ispc in range(stru.nspc):
            for jspc in range(stru.nspc):
                spc1 = stru.atomic_species[ispc]
                spc2 = stru.atomic_species[jspc]
                orbpairs_thisij1, orbpairs_thisij2, orbpairs_thisij3 = [], [], []
                for jorb in range(basis.norb_spc[spc2]):
                    r2 = basis.phirgrids_spc[spc2][jorb].rcut
                    for iorb in range(basis.norb_spc[spc1]):
                        r1 = basis.phirgrids_spc[spc1][iorb].rcut
                        thispair = OrbPair(basis.phiQlist_spc[spc1][iorb],
                                            basis.phiQlist_spc[spc2][jorb], r1 + r2, 1)
                        orbpairs_thisij1.append(thispair)
                        thispair = OrbPair(basis.phiQlist_spc[spc1][iorb],
                                            basis.phiQlist_spc[spc2][jorb], r1 + r2, 2)
                        orbpairs_thisij3.append(thispair)
                    for iorb in range(projR.norb_spc[spc1]):
                        r1 = projR.phirgrids_spc[spc1][iorb].rcut
                        thispair = OrbPair(projR.phiQlist_spc[spc1][iorb],
                                            basis.phiQlist_spc[spc2][jorb], r1 + r2, 1)
                        orbpairs_thisij2.append(thispair)
                orbpairs1[(spc1, spc2)] = orbpairs_thisij1
                orbpairs2[(spc1, spc2)] = orbpairs_thisij2
                orbpairs3[(spc1, spc2)] = orbpairs_thisij3
        
        if is_master(): print('Calculating overlap')
        olp_basis = calc_overlap(basis, orbpairs1, Ecut=ecut)

        if not (self.overlaps_only and funcg is None):
            # skip this step if overlaps_only and using NCPP
            olp_proj_ao = calc_overlap(projR, orbpairs2, basis, Ecut=ecut)
        
        # mats0 stores the non-local potential of each AO basis, with multiple repeated atom pairs
        trans, atoms, mats0 = get_mat0(olp_proj_ao, funch)

        assert funcg is None
        overlaps = olp_basis

        if cutoffs is not None:
            pairs_cut = pwc(self.structure, cutoffs)
            # cutoff overlaps
            overlaps_cut = MatLCAO.setc(pairs_cut, self.lcaodata, filling_value=0., dtype='f8')
            overlaps_cut.convert_to(overlaps)
            overlaps = overlaps_cut
        
        if is_master():
            print('\nWriting overlap matrices to disk')
            save_mat_deeph(savedir, overlaps, filename='overlaps.h5', energy_unit=False)
        
        if self.overlaps_only:
            return
        
        # Now deal with Hamiltonians
        
        FFTgrid = np.array(vlocr.shape)
        assert np.isrealobj(vlocr), 'Complex array not implemented'
        rprimFFT = self.structure.rprim / FFTgrid[:, None]
        dvol = self.structure.cell_volume / np.prod(FFTgrid)
        
        grids_site_orb = []
        for iatom in range(self.structure.natom):
            spc = self.structure.atomic_numbers[iatom]
            poscart = self.structure.atomic_positions_cart[iatom]
            grids_site = []
            for iorb in range(basis.norb_spc[spc]):
                obr = basis.phirgrids_spc[spc][iorb].rcut
                # grids_site stores the non-zero grid points index for each atomic orbital
                grids_site.append(GridPoints.find(rprimFFT, obr, poscart))
            grids_site_orb.append(grids_site)

        # create the index for each atom pair (i, j) + translational vectors
        # in order to process atom pairs of the same kind in batch
        # npairs2 means how many different kinds of atom pairs in total 
        xs2 = pairs_to_indices(olp_proj_ao.structure, trans, atoms)
        argsort = np.argsort(xs2, kind='stable')
        xs2 = xs2[argsort]
        slice2 = slice_same(xs2)
        npairs2 = len(slice2) - 1
        
        Hmain = MatLCAO.setc(olp_basis.get_pairs_ij(), basis, filling_value=0., dtype='f8')
        Hmain.shuffle()

        # Calculate kinetic energy
        Hkin = calc_overlap(basis, orbpairs3, Ecut=ecut)

        # Construct Hamiltonian under atomic orbital basis in real space
        constructH(self, vlocr, basis, FFTgrid, rprimFFT, dvol, grids_site_orb, Hmain)

        # Collect matrices into one object
        # mats2 sums contribution from the same atom pairs together
        mats2 = []
        for ipair in range(npairs2):
            slice_thispair = slice(slice2[ipair], slice2[ipair+1])
            mats2.append(np.sum([mats0[i] for i in argsort[slice_thispair]], axis=0))
        xs3 = np.unique(xs2)
        trans3, atms3 = indices_to_pairs(olp_proj_ao.structure.natom, xs3)
        Hmain.duplicate()
        
        Hcorr = MatLCAO(olp_proj_ao.structure, trans3, atms3, mats2, olp_proj_ao.lcaodata2)
        Hcorr.duplicate()
        
        # Sum up all the terms
        # addition and subtraction are overloaded in MatLCAO._add_sub() 
        hamiltonians = Hkin + Hmain + Hcorr

        if cutoffs is not None:
            # cutoff hamiltonians
            hamiltonians_cut = MatLCAO.setc(pairs_cut, self.lcaodata, filling_value=0., dtype='f8')
            hamiltonians_cut.convert_to(hamiltonians)
            hamiltonians = hamiltonians_cut

        if is_master():
            print('Writing Hamiltonian matrices to disk')
            save_mat_deeph(savedir, hamiltonians, filename='hamiltonians.h5', energy_unit=True)

        if comm is not None:
            comm.Barrier()
