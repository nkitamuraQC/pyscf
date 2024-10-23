from numpy import einsum
import numpy as np

def make_rdm1(mycc, t1, t2, l1, l2):
    nocc = mycc._scf.mol.nelectron // 2
    nmo = mycc._scf.mo_coeff.shape[1]
    t1 = t1.T

    t2ab = np.copy(t2)
    t2aa = np.copy(t2) - t2.transpose(0,1,3,2)

    l2ab = np.copy(l2) * 2
    l2aa = np.copy(l2) - l2.transpose(0,1,3,2)

    t2ab = t2ab.transpose(2,3,0,1)
    t2aa = t2aa.transpose(2,3,0,1)

    t2 = [t2aa, t2ab]
    l2 = [l2aa, l2ab]

    nvir = nmo - nocc 
    ov = get_ov(mycc, t1, t2, l1, l2)
    vv = get_vv(mycc, t1, t2, l1, l2)
    oo = get_oo(mycc, t1, t2, l1, l2)
    vo = get_vo(mycc, t1, t2, l1, l2)
    dm1 = np.zeros((nmo, nmo), dtype=float)
    dm1[:nocc, :nocc] = oo
    dm1[:nocc, nocc:] = ov
    dm1[nocc:, :nocc] = vo
    dm1[nocc:, nocc:] = vv
    #dm1 += make_rdm1(mycc, t1, t2, l1, l2)
    return dm1

def get_ov(mycc, t1, t2, l1, l2):
    T, U = t2
    L, M = l2
    nocc = mycc._scf.mol.nelectron // 2
    nmo = mycc._scf.mo_coeff.shape[1]
    nvir = nmo - nocc 
    ov = np.zeros((nocc, nvir), dtype=float)
    ov += 1.0*einsum('ia->ia', l1)
    return ov


def get_vv(mycc, t1, t2, l1, l2):
    T, U = t2
    L, M = l2
    t = t1
    l = l1
    nocc = mycc._scf.mol.nelectron // 2
    nmo = mycc._scf.mo_coeff.shape[1]
    nvir = nmo - nocc 
    vv = np.zeros((nvir, nvir), dtype=float)
    delta = np.identity(nvir)
    vv += 1.0*einsum('ib,ai->ab', l, t)
    vv += -40.5*einsum('ijcb,caji->ab', L, T)
    vv += -4.5*einsum('ijcb,caji->ab', L, U)
    vv += -4.5*einsum('ijcb,caji->ab', M, T)
    vv += -0.5*einsum('ijcb,caji->ab', M, U)
    vv += -0.5*einsum('ijbA,aAji->ab', M, U)
    vv += 1.0*einsum('iIcb,caiI->ab', M, U)
    vv += 1.0*einsum('iIbA,aAiI->ab', M, U)
    vv += -0.5*einsum('IJcb,caJI->ab', M, U)
    vv += -0.5*einsum('IJbA,aAJI->ab', M, U)
    
    return vv


def get_oo(mycc, t1, t2, l1, l2):
    T, U = t2
    L, M = l2
    t = t1
    l = l1
    nocc = mycc._scf.mol.nelectron // 2
    nmo = mycc._scf.mo_coeff.shape[1]
    nvir = nmo - nocc 
    oo = np.zeros((nocc, nocc), dtype=float)
    delta = np.identity(nocc)
    oo += -1.0*einsum('ia,aj->ij', l, t)
    oo += 40.5*einsum('ikab,bajk->ij', L, T)
    oo += 4.5*einsum('ikab,bajk->ij', L, U)
    oo += 4.5*einsum('ikab,bajk->ij', M, T)
    oo += 0.5*einsum('ikab,bajk->ij', M, U)
    oo += -1.0*einsum('ikaA,aAjk->ij', M, U)
    oo += 0.5*einsum('ikAB,BAjk->ij', M, U)
    oo += 0.5*einsum('iIab,bajI->ij', M, U)
    oo += -1.0*einsum('iIaA,aAjI->ij', M, U)
    oo += 0.5*einsum('iIAB,BAjI->ij', M, U)
    return oo


def get_vo(mycc, t1, t2, l1, l2):
    T, U = t2
    L, M = l2
    t = t1
    l = l1
    nocc = mycc._scf.mol.nelectron // 2
    nmo = mycc._scf.mo_coeff.shape[1]
    nvir = nmo - nocc 
    vo = np.zeros((nvir, nocc), dtype=float)
    delta_o = np.identity(nocc)
    delta_v = np.identity(nvir)
    vo += 1.0*einsum('ai->ai', t)
    vo += -9.0*einsum('jb,baij->ai', l, T)
    vo += -1.0*einsum('jb,baij->ai', l, U)
    vo += -1.0*einsum('jb,aj,bi->ai', l, t, t)
    vo += 40.5*einsum('jkbc,ci,bakj->ai', L, t, T)
    vo += 4.5*einsum('jkbc,ci,bakj->ai', L, t, U)
    vo += 40.5*einsum('jkbc,aj,cbik->ai', L, t, T)
    vo += 4.5*einsum('jkbc,aj,cbik->ai', L, t, U)
    vo += 4.5*einsum('jkbc,ci,bakj->ai', M, t, T)
    vo += 0.5*einsum('jkbc,ci,bakj->ai', M, t, U)
    vo += 4.5*einsum('jkbc,aj,cbik->ai', M, t, T)
    vo += 0.5*einsum('jkbc,aj,cbik->ai', M, t, U)
    vo += 0.5*einsum('jkbA,bi,aAkj->ai', M, t, U)
    vo += -1.0*einsum('jkbA,aj,bAik->ai', M, t, U)
    vo += 0.5*einsum('jkAB,aj,BAik->ai', M, t, U)
    vo += -1.0*einsum('jIbc,ci,bajI->ai', M, t, U)
    vo += 0.5*einsum('jIbc,aj,cbiI->ai', M, t, U)
    vo += -1.0*einsum('jIbA,bi,aAjI->ai', M, t, U)
    vo += -1.0*einsum('jIbA,aj,bAiI->ai', M, t, U)
    vo += 0.5*einsum('jIAB,aj,BAiI->ai', M, t, U)
    vo += 0.5*einsum('IJbc,ci,baJI->ai', M, t, U)
    vo += 0.5*einsum('IJbA,bi,aAJI->ai', M, t, U)
    
    
    return vo


if __name__ == "__main__":
    from pyscf import gto, scf, ci, cc, tdscf, fci
    from pyscf.cc.ccsd_rdm_slow import _gamma1_intermediates
    def run_ccsd(mol, root=0):
        mf = scf.RHF(mol)
        mf.verbose = 0
        mf.scf()
        mycc = cc.RCCSD(mf)
        mycc.verbose = 0
        mycc.ccsd()
        t1, t2 = mycc.t1, mycc.t2
        l1, l2 = mycc.solve_lambda(t1=t1, t2=t2)
        doo, dov, dvo, dvv = _gamma1_intermediates(mycc, t1, t2, l1, l2)
        nocc = mycc._scf.mol.nelectron // 2
        nmo = mycc._scf.mo_coeff.shape[1]
        dm1 = np.zeros((nmo, nmo), dtype=float)
        dm1[:nocc, :nocc] = doo
        dm1[:nocc, nocc:] = dov
        dm1[nocc:, :nocc] = dvo
        dm1[nocc:, nocc:] = dvv

        dm1_my = make_rdm1(mycc, t1, t2, l1, l2)

        print(np.max(dm1), np.max(dm1_my))
        print(np.min(dm1), np.min(dm1_my))

    mol = gto.Mole()
    mol.verbose = 0
    mol.atom = 'O 0 0 0; H 0.958 0.0 0.0; H 0.240 0.927 0.0;'
    mol.atom = 'H 0 0 0; Cl 0 0 1.0'
    #mol.atom = 'H 0 0 0; H 0 0 1.0; H 0 0 2; H 0 0 3; H 0 0 4; H 0 0 5;'
    #mol.atom = 'Kr 0 0 0;'
    mol.basis = 'def2-svp'
    mol.build()


    run_ccsd(mol)