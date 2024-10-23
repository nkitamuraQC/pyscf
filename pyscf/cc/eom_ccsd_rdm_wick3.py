from numpy import einsum
import numpy as np
from pyscf.cc.ccsd_rdm import make_rdm1, _make_rdm1

def trans_rdm1(mycc, t1, t2, l1, l2, r1, r2):
    nocc = mycc._scf.mol.nelectron // 2
    nmo = mycc._scf.mo_coeff.shape[1]
    t1a  = t1
    t2ab = np.copy(t2)
    t2aa = np.copy(t2) - t2.transpose(0,1,3,2)

    l1a  = l1
    l2ab = np.copy(l2) * 2
    l2aa = np.copy(l2) - l2.transpose(0,1,3,2)
    
    r1a  = r1
    r2ab = np.copy(r2) * 2
    r2aa = np.copy(r2) - r2.transpose(0,1,3,2)
    
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
    ov = get_ov(mycc, t1, t2, l1, l2, r1, r2) * 0.5
    vv = get_vv(mycc, t1, t2, l1, l2, r1, r2) * 0.5 
    oo = get_oo(mycc, t1, t2, l1, l2, r1, r2) * 0.5 
    vo = get_vo(mycc, t1, t2, l1, l2, r1, r2) * 0.5 
    dm1 = np.zeros((nmo, nmo), dtype=float)
    dm1[:nocc, :nocc] = oo + oo.T
    dm1[:nocc, nocc:] = ov + vo.T
    dm1[nocc:, :nocc] = vo + ov.T
    dm1[nocc:, nocc:] = vv + vv.T
    dm1 = _make_rdm1(mycc, [oo, ov, vo, vv])
    return dm1

def get_ov(mycc, t1, t2, l1, l2, r1, r2):
    T, U = t2
    L, M = l2
    R, S = r2
    t = t1
    l = l1
    r = r1
    nocc = mycc._scf.mol.nelectron // 2
    nmo = mycc._scf.mo_coeff.shape[1]
    nvir = nmo - nocc 
    ov = np.zeros((nocc, nvir), dtype=float)
    ov += -9.0*einsum('ijab,aj->ib', L, r)
    ov += -1.0*einsum('ijab,aj->ib', M, r)
    return ov


def get_vv(mycc, t1, t2, l1, l2, r1, r2):
    T, U = t2
    L, M = l2
    R, S = r2
    t = t1
    l = l1
    r = r1
    nocc = mycc._scf.mol.nelectron // 2
    nmo = mycc._scf.mo_coeff.shape[1]
    nvir = nmo - nocc 
    vv = np.zeros((nvir, nvir), dtype=float)
    vv += 1.0*einsum('ib,ai->ab', l, r)
    vv += -40.5*einsum('ijcb,caji->ab', L, R)
    vv += -4.5*einsum('ijcb,caji->ab', L, S)
    vv += -4.5*einsum('ijcb,caji->ab', M, R)
    vv += -0.5*einsum('ijcb,caji->ab', M, S)
    vv += -0.5*einsum('ijbA,aAji->ab', M, S)
    vv += 1.0*einsum('iIcb,caiI->ab', M, S)
    vv += 1.0*einsum('iIbA,aAiI->ab', M, S)
    vv += -0.5*einsum('IJcb,caJI->ab', M, S)
    vv += -0.5*einsum('IJbA,aAJI->ab', M, S)
    vv += -9.0*einsum('ijcb,ai,cj->ab', L, t, r)
    vv += -1.0*einsum('ijcb,ai,cj->ab', M, t, r)
    
    return vv


def get_oo(mycc, t1, t2, l1, l2, r1, r2):
    T, U = t2
    L, M = l2
    R, S = r2
    t = t1
    l = l1
    r = r1
    nocc = mycc._scf.mol.nelectron // 2
    nmo = mycc._scf.mo_coeff.shape[1]
    nvir = nmo - nocc 
    oo = np.zeros((nocc, nocc), dtype=float)
    oo += -1.0*einsum('ia,aj->ij', l, r)
    oo += 40.5*einsum('ikab,bajk->ij', L, R)
    oo += 4.5*einsum('ikab,bajk->ij', L, S)
    oo += 4.5*einsum('ikab,bajk->ij', M, R)
    oo += 0.5*einsum('ikab,bajk->ij', M, S)
    oo += -1.0*einsum('ikaA,aAjk->ij', M, S)
    oo += 0.5*einsum('ikAB,BAjk->ij', M, S)
    oo += 0.5*einsum('iIab,bajI->ij', M, S)
    oo += -1.0*einsum('iIaA,aAjI->ij', M, S)
    oo += 0.5*einsum('iIAB,BAjI->ij', M, S)
    oo += 9.0*einsum('ikab,bj,ak->ij', L, t, r)
    oo += 1.0*einsum('ikab,bj,ak->ij', M, t, r)
    return oo


def get_vo(mycc, t1, t2, l1, l2, r1, r2):
    T, U = t2
    L, M = l2
    R, S = r2
    t = t1
    l = l1
    r = r1
    nocc = mycc._scf.mol.nelectron // 2
    nmo = mycc._scf.mo_coeff.shape[1]
    nvir = nmo - nocc 
    vo = np.zeros((nvir, nocc), dtype=float)
    vo += 1.0*einsum('ai->ai', r)
    vo += -9.0*einsum('jb,baij->ai', l, R)
    vo += -1.0*einsum('jb,baij->ai', l, S)
    vo += -1.0*einsum('jb,bi,aj->ai', l, t, r)
    vo += -1.0*einsum('jb,aj,bi->ai', l, t, r)
    vo += 1.0*einsum('jb,ai,bj->ai', l, t, r)
    vo += 40.5*einsum('jkbc,ci,bakj->ai', L, t, R)
    vo += 4.5*einsum('jkbc,ci,bakj->ai', L, t, S)
    vo += 40.5*einsum('jkbc,aj,cbik->ai', L, t, R)
    vo += 4.5*einsum('jkbc,aj,cbik->ai', L, t, S)
    vo += 20.25*einsum('jkbc,ai,cbkj->ai', L, t, R)
    vo += 2.25*einsum('jkbc,ai,cbkj->ai', L, t, S)
    vo += -40.5*einsum('jkbc,cbij,ak->ai', L, T, r)
    vo += -40.5*einsum('jkbc,cakj,bi->ai', L, T, r)
    vo += 81.0*einsum('jkbc,caij,bk->ai', L, T, r)
    vo += -4.5*einsum('jkbc,cbij,ak->ai', L, U, r)
    vo += -4.5*einsum('jkbc,cakj,bi->ai', L, U, r)
    vo += 9.0*einsum('jkbc,caij,bk->ai', L, U, r)
    vo += 4.5*einsum('jkbc,ci,bakj->ai', M, t, R)
    vo += 0.5*einsum('jkbc,ci,bakj->ai', M, t, S)
    vo += 4.5*einsum('jkbc,aj,cbik->ai', M, t, R)
    vo += 0.5*einsum('jkbc,aj,cbik->ai', M, t, S)
    vo += 2.25*einsum('jkbc,ai,cbkj->ai', M, t, R)
    vo += 0.25*einsum('jkbc,ai,cbkj->ai', M, t, S)
    vo += -4.5*einsum('jkbc,cbij,ak->ai', M, T, r)
    vo += -4.5*einsum('jkbc,cakj,bi->ai', M, T, r)
    vo += 9.0*einsum('jkbc,caij,bk->ai', M, T, r)
    vo += -0.5*einsum('jkbc,cbij,ak->ai', M, U, r)
    vo += -0.5*einsum('jkbc,cakj,bi->ai', M, U, r)
    vo += 1.0*einsum('jkbc,caij,bk->ai', M, U, r)
    vo += 0.5*einsum('jkbA,bi,aAkj->ai', M, t, S)
    vo += -1.0*einsum('jkbA,aj,bAik->ai', M, t, S)
    vo += -0.5*einsum('jkbA,ai,bAkj->ai', M, t, S)
    vo += 1.0*einsum('jkbA,bAij,ak->ai', M, U, r)
    vo += 0.5*einsum('jkbA,aAkj,bi->ai', M, U, r)
    vo += -1.0*einsum('jkbA,aAij,bk->ai', M, U, r)
    vo += 0.5*einsum('jkAB,aj,BAik->ai', M, t, S)
    vo += 0.25*einsum('jkAB,ai,BAkj->ai', M, t, S)
    vo += -0.5*einsum('jkAB,BAij,ak->ai', M, U, r)
    vo += -1.0*einsum('jIbc,ci,bajI->ai', M, t, S)
    vo += 0.5*einsum('jIbc,aj,cbiI->ai', M, t, S)
    vo += -0.5*einsum('jIbc,ai,cbjI->ai', M, t, S)
    vo += 0.5*einsum('jIbc,cbiI,aj->ai', M, U, r)
    vo += 1.0*einsum('jIbc,cajI,bi->ai', M, U, r)
    vo += -1.0*einsum('jIbc,caiI,bj->ai', M, U, r)
    vo += -1.0*einsum('jIbA,bi,aAjI->ai', M, t, S)
    vo += -1.0*einsum('jIbA,aj,bAiI->ai', M, t, S)
    vo += 1.0*einsum('jIbA,ai,bAjI->ai', M, t, S)
    vo += -1.0*einsum('jIbA,bAiI,aj->ai', M, U, r)
    vo += -1.0*einsum('jIbA,aAjI,bi->ai', M, U, r)
    vo += 1.0*einsum('jIbA,aAiI,bj->ai', M, U, r)
    vo += 0.5*einsum('jIAB,aj,BAiI->ai', M, t, S)
    vo += -0.5*einsum('jIAB,ai,BAjI->ai', M, t, S)
    vo += 0.5*einsum('jIAB,BAiI,aj->ai', M, U, r)
    vo += 0.5*einsum('IJbc,ci,baJI->ai', M, t, S)
    vo += 0.25*einsum('IJbc,ai,cbJI->ai', M, t, S)
    vo += -0.5*einsum('IJbc,caJI,bi->ai', M, U, r)
    vo += 0.5*einsum('IJbA,bi,aAJI->ai', M, t, S)
    vo += -0.5*einsum('IJbA,ai,bAJI->ai', M, t, S)
    vo += 0.5*einsum('IJbA,aAJI,bi->ai', M, U, r)
    vo += 0.25*einsum('IJAB,ai,BAJI->ai', M, t, S)
    vo += 9.0*einsum('jkbc,aj,ci,bk->ai', L, t, t, r)
    
    
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
    #mol.atom = 'O 0 0 0; H 0.958 0.0 0.0; H 0.240 0.927 0.0;'
    mol.atom = 'H 0 0 0; Cl 0 0 1.0'
    #mol.atom = 'Li 0 0 0; H 0 0 1.0;'
    #mol.atom = 'Kr 0 0 0;'
    mol.basis = 'def2-svp'
    mol.build()

    mycc, dip, mf, t1, t2, l1, l2, r1, r2 = run_eomee(mol, root=0)
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
