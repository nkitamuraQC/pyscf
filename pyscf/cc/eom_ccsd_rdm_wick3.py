from numpy import einsum
import numpy as np
from pyscf.cc.ccsd_rdm import make_rdm1

def trans_rdm1(mycc, t1, t2, l1, l2, r1, r2):
    nocc = mycc._scf.mol.nelectron // 2
    nmo = mycc._scf.mo_coeff.shape[1]
    t1a  = t1
    t2ab = np.copy(t2)
    t2aa = np.copy(t2) \
         - t2.transpose(0,1,3,2)

    l1a  = l1
    l2ab = 2*np.copy(l2)
    l2aa = np.copy(l2) \
         - l2.transpose(0,1,3,2)
    
    r1a  = r1
    r2ab = 2*np.copy(r2)
    r2aa = np.copy(r2) \
         - r2.transpose(0,1,3,2)
    
    r1 = r1a.T
    r2ab = r2ab.transpose(2,3,0,1)
    r2aa = r2aa.transpose(2,3,0,1)
    t1 = t1a.T
    t2ab = t2ab.transpose(2,3,0,1)
    t2aa = t2aa.transpose(2,3,0,1)

    t2 = [t2aa, t2ab]
    l2 = [l2aa, l2ab]
    r2 = [r2aa, r2ab]

    nvir = nmo - nocc 
    ov = get_ov(mycc, t1, t2, l1, l2, r1, r2)
    vv = get_vv(mycc, t1, t2, l1, l2, r1, r2)
    oo = get_oo(mycc, t1, t2, l1, l2, r1, r2)
    vo = get_vo(mycc, t1, t2, l1, l2, r1, r2)
    dm1 = np.zeros((nmo, nmo), dtype=float)
    dm1[:nocc, :nocc] = oo + oo.T
    dm1[:nocc, nocc:] = ov + vo.T
    dm1[nocc:, :nocc] = vo + ov.T
    dm1[nocc:, nocc:] = vv + vv.T
    #dm1 += make_rdm1(mycc, t1, t2, l1, l2)
    return dm1

def get_ov(mycc, t1, t2, l1, l2, r1, r2):
    t2aa, t2ab = t2
    l2aa, l2ab = l2
    r2aa, r2ab = r2
    nocc = mycc._scf.mol.nelectron // 2
    nmo = mycc._scf.mo_coeff.shape[1]
    nvir = nmo - nocc 
    ov = np.zeros((nocc, nvir), dtype=float)
    ov += -1.0*einsum('ijab,aj->ib', l2, r1)
    return ov


def get_vv(mycc, t1, t2, l1, l2, r1, r2):
    t2aa, t2ab = t2
    l2aa, l2ab = l2
    r2aa, r2ab = r2
    nocc = mycc._scf.mol.nelectron // 2
    nmo = mycc._scf.mo_coeff.shape[1]
    nvir = nmo - nocc 
    vv = np.zeros((nvir, nvir), dtype=float)
    delta = np.identity(nvir)
    vv += 1.0*einsum('ib,ai->ab', l1, t1)
    vv += -0.5*einsum('ijcb,caji->ab', l2, r2)
    vv += -1.0*einsum('ijcb,ai,cj->ab', l2, t1, r1)
    
    return vv


def get_oo(mycc, t1, t2, l1, l2, r1, r2):
    t2aa, t2ab = t2
    l2aa, l2ab = l2
    r2aa, r2ab = r2
    nocc = mycc._scf.mol.nelectron // 2
    nmo = mycc._scf.mo_coeff.shape[1]
    nvir = nmo - nocc 
    oo = np.zeros((nocc, nocc), dtype=float)
    delta = np.identity(nocc)
    oo += -1.0*einsum('ia,aj->ij', l1, r1)
    oo += 0.5*einsum('ikab,bajk->ij', l2, r2)
    oo += 1.0*einsum('ikab,bj,ak->ij', l2, t1, r1)
    return oo


def get_vo(mycc, t1, t2, l1, l2, r1, r2):
    t2aa, t2ab = t2
    l2aa, l2ab = l2
    r2aa, r2ab = r2
    nocc = mycc._scf.mol.nelectron // 2
    nmo = mycc._scf.mo_coeff.shape[1]
    nvir = nmo - nocc 
    vo = np.zeros((nvir, nocc), dtype=float)
    delta_o = np.identity(nocc)
    delta_v = np.identity(nvir)
    vo += 1.0*einsum('ai->ai', r1)
    vo += -1.0*einsum('jb,baij->ai', l1, r2)
    vo += -1.0*einsum('jb,bi,aj->ai', l1, t1, r1)
    vo += -1.0*einsum('jb,aj,bi->ai', l1, t1, r1)
    vo += 1.0*einsum('jb,ai,bj->ai', l1, t1, r1)
    vo += 0.5*einsum('jkbc,ci,bakj->ai', l2, t1, r2)
    vo += 0.5*einsum('jkbc,aj,cbik->ai', l2, t1, r2)
    vo += 0.25*einsum('jkbc,ai,cbkj->ai', l2, t1, r2)
    vo += -0.5*einsum('jkbc,cbij,ak->ai', l2, t2, r1)
    vo += -0.5*einsum('jkbc,cakj,bi->ai', l2, t2, r1)
    vo += 1.0*einsum('jkbc,caij,bk->ai', l2, t2, r1)
    vo += 1.0*einsum('jkbc,aj,ci,bk->ai', l2, t1, t1, r1)
    
    
    return vo


if __name__ == "__main__":
    from pyscf import gto, scf, ci, cc, tdscf, fci

    def benchmark(mol, do_fci=False, root=2):
        mf = scf.RHF(mol)
        mf.verbose = 0
        mf.scf()
    
        mytd = tdscf.TDRHF(mf)
        # mytd.singlet = False
        mytd.kernel()
        mytd.verbose = 4
        trdip_td = mytd.transition_dipole()[0]
        mytd.analyze()
    
        myci = ci.CISD(mf)
        myci.nroots = 5
        es, cs = myci.kernel()

        if do_fci:
            myfci = fci.FCI(mf)
            myfci.nroots = 5
            es_fci, cs_fci = myfci.kernel()

            dm1_fci = fci.direct_spin1.trans_rdm1(cs_fci[0], cs_fci[root], mf.mo_coeff.shape[1], mf.mol.nelectron)
            t_dm1_fci = np.einsum('pi,ij,qj->pq', mf.mo_coeff, dm1_fci, mf.mo_coeff.conj())
    
        dm1 = ci.cisd.trans_rdm1(myci, cs[0], cs[root])
        t_dm1 = np.einsum('pi,ij,qj->pq', mf.mo_coeff, dm1, mf.mo_coeff.conj())

        charge_center = (np.einsum('z,zx->x', mol.atom_charges(), mol.atom_coords())
                         / mol.atom_charges().sum())
        with mol.with_common_origin(charge_center):
            trdip_ci = np.einsum('xij,ji->x', mol.intor_symmetric('int1e_r'), t_dm1)
            if do_fci:
                trdip_fci = np.einsum('xij,ji->x', mol.intor('int1e_r'), t_dm1_fci)
            else:
                trdip_fci = [None, None, None]
        return trdip_td, trdip_ci, trdip_fci


    def run_eomee(mol, root=0):
        mf = scf.RHF(mol)
        mf.verbose = 0
        mf.scf()
        mycc = cc.RCCSD(mf)
        mycc.verbose = 0
        mycc.ccsd()
        t1, t2 = mycc.t1, mycc.t2
        l1, l2 = mycc.solve_lambda(t1=t1, t2=t2)
        eom_cc = cc.eom_rccsd.EOMEESinglet(mycc)
        e, c = cc.eom_rccsd.eomee_ccsd_singlet(eom_cc, nroots=5)
        r1, r2 = eom_cc.vector_to_amplitudes(c[root])
        dipole = mol.intor_symmetric("int1e_r", comp=3)
        dipole = np.einsum("xij,ia,jb->xab", dipole, mf.mo_coeff, mf.mo_coeff)
        
        return mycc, dipole, mf, t1, t2, l1, l2, r1, r2

    mol = gto.Mole()
    mol.verbose = 0
    mol.atom = 'O 0 0 0; H 0.958 0.0 0.0; H 0.240 0.927 0.0;'
    mol.atom = 'H 0 0 0; Cl 0 0 1.0'
    #mol.atom = 'H 0 0 0; H 0 0 1.0; H 0 0 2; H 0 0 3; H 0 0 4; H 0 0 5;'
    #mol.atom = 'Kr 0 0 0;'
    mol.basis = 'def2-svp'
    mol.build()

    mycc, dip, mf, t1, t2, l1, l2, r1, r2 = run_eomee(mol, root=2)
    t_dm1 = trans_rdm1(mycc, t1, t2, l1, l2, r1, r2)
    t_dm1 = np.einsum('pi,ij,qj->pq', mf.mo_coeff, t_dm1, mf.mo_coeff.conj())

    charge_center = (np.einsum('z,zx->x', mol.atom_charges(), mol.atom_coords())
                     / mol.atom_charges().sum())
    with mol.with_common_origin(charge_center):
        trdip_cc = np.einsum('xij,ji->x', mol.intor_symmetric('int1e_r'), t_dm1)

    trdip_td, trdip_ci, trdip_fci = benchmark(mol, do_fci=False, root=3)
    print("######################")
    for dir in [0, 1, 2]:
        print(f"CCSD: {dir}", trdip_cc[dir])
        print(f"CISD: {dir}", trdip_ci[dir])
        print(f"TDDFT: {dir}", trdip_td[dir])
        print(f"FCI: {dir}", trdip_fci[dir])
        print("######################")
    print(np.argsort(trdip_cc))
    print(np.argsort(trdip_ci))
    print(np.argsort(trdip_td))
    print("######################")
    print(np.argsort(np.abs(trdip_cc)))
    print(np.argsort(np.abs(trdip_ci)))
    print(np.argsort(np.abs(trdip_td)))
