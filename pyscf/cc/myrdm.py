from numpy import einsum
import numpy as np

def make_rdm1(mycc, t1, t2, l1, l2):
    nocc = mycc._scf.mol.nelectron // 2
    nmo = mycc._scf.mo_coeff.shape[1]
    t1 = t1.T
    t2 = t2.transpose(2,3,0,1)

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
    return oo, ov, vo, vv

def get_ov(mycc, t1, t2, l1, l2):
    nocc = mycc._scf.mol.nelectron // 2
    nmo = mycc._scf.mo_coeff.shape[1]
    nvir = nmo - nocc 
    ov = np.zeros((nocc, nvir), dtype=float)
    ov += 1.0*einsum('ia->ia', l1)
    return ov


def get_vv(mycc, t1, t2, l1, l2):
    nocc = mycc._scf.mol.nelectron // 2
    nmo = mycc._scf.mo_coeff.shape[1]
    nvir = nmo - nocc 
    vv = np.zeros((nvir, nvir), dtype=float)
    delta = np.identity(nvir)
    vv += 1.0*einsum('ib,ai->ab', l1, t1)
    vv += -0.5*einsum('ijcb,caji->ab', l2, t2)
    
    return vv


def get_oo(mycc, t1, t2, l1, l2):
    nocc = mycc._scf.mol.nelectron // 2
    nmo = mycc._scf.mo_coeff.shape[1]
    nvir = nmo - nocc 
    oo = np.zeros((nocc, nocc), dtype=float)
    delta = np.identity(nocc)
    oo += -1.0*einsum('ia,aj->ij', l1, t1)
    oo += 0.5*einsum('ikab,bajk->ij', l2, t2)
    return oo


def get_vo(mycc, t1, t2, l1, l2):
    nocc = mycc._scf.mol.nelectron // 2
    nmo = mycc._scf.mo_coeff.shape[1]
    nvir = nmo - nocc 
    vo = np.zeros((nvir, nocc), dtype=float)
    delta_o = np.identity(nocc)
    delta_v = np.identity(nvir)
    vo += 1.0*einsum('ai->ai', t1)
    vo += -1.0*einsum('jb,baij->ai', l1, t2)
    vo += -1.0*einsum('jb,aj,bi->ai', l1, t1, t1)
    vo += 0.5*einsum('jkbc,ci,bakj->ai', l2, t1, t2)
    vo += 0.5*einsum('jkbc,aj,cbik->ai', l2, t1, t2)
    
    
    return vo


if __name__ == "__main__":
    from pyscf import gto, scf, ci, cc, tdscf, fci
    from pyscf.cc.ccsd_rdm import _gamma1_intermediates, _make_rdm1
    def run_ccsd(mol, root=0):
        mf = scf.RHF(mol)
        mf.verbose = 0
        mf.scf()
        mycc = cc.RCCSD(mf)
        mycc.verbose = 0
        mycc.ccsd()
        t1, t2 = mycc.t1, mycc.t2
        l1, l2 = mycc.solve_lambda(t1=t1, t2=t2)
        dm1 = mycc.make_rdm1()
        doo, dov, dvo, dvv = _gamma1_intermediates(mycc, t1, t2, l1, l2)
        dm1 = _make_rdm1(mycc, [doo, dov, dvo, dvv])

        oo, ov, vo, vv = make_rdm1(mycc, t1, t2, l1, l2)
        mydm1 = _make_rdm1(mycc, [oo, ov, vo, vv])

        print(np.max(dm1), np.max(mydm1))
        print(np.min(dm1), np.min(mydm1))

    mol = gto.Mole()
    mol.verbose = 0
    mol.atom = 'O 0 0 0; H 0.958 0.0 0.0; H 0.240 0.927 0.0;'
    mol.atom = 'H 0 0 0; Cl 0 0 1.0'
    #mol.atom = 'H 0 0 0; H 0 0 1.0; H 0 0 2; H 0 0 3; H 0 0 4; H 0 0 5;'
    #mol.atom = 'Kr 0 0 0;'
    mol.basis = 'def2-svp'
    mol.build()


    run_ccsd(mol)