from numpy import einsum
import numpy as np

def trans_rdm1(mycc, t1, t2, l1, l2, r1, r2):
    nocc = mycc._scf.mol.nelectron // 2
    nmo = mycc._scf.mo_coeff.shape[1]
    nvir = nmo - nocc 
    ov = get_ov(mycc, t1, t2, l1, l2, r1, r2)
    vv = get_vv(mycc, t1, t2, l1, l2, r1, r2)
    oo = get_oo(mycc, t1, t2, l1, l2, r1, r2)
    vo = get_vo(mycc, t1, t2, l1, l2, r1, r2)
    dm1 = np.zeros((nmo, nmo), dtype=float)
    dm1[:nocc, :nocc] = oo
    dm1[:nocc, nocc:] = ov
    dm1[nocc:, :nocc] = vo
    dm1[nocc:, nocc:] = vv
    return dm1

def get_ov(mycc, t1, t2, l1, l2, r1, r2):
    nocc = mycc._scf.mol.nelectron // 2
    nmo = mycc._scf.mo_coeff.shape[1]
    nvir = nmo - nocc 
    ov = np.zeros((nocc, nvir), dtype=float)
    ov += -1.0*einsum('ijab,jb->ia', l2, r1)
    return ov


def get_vv(mycc, t1, t2, l1, l2, r1, r2):
    nocc = mycc._scf.mol.nelectron // 2
    nmo = mycc._scf.mo_coeff.shape[1]
    nvir = nmo - nocc 
    vv = np.zeros((nvir, nvir), dtype=float)
    delta = np.identity(nvir)
    vv += 1.0*einsum('ib,ac,ic->ab', l1, delta, r1)
    vv += -1.0*einsum('aj,jicb,ic->ab', t1.T, l2, r1)

    vv += -1.0*einsum('jicb,ad,ijcd->ab', l2, delta, r2)
    vv += 1.0*einsum('jidb,ac,ijcd->ab', l2, delta, r2)
    
    return vv


def get_oo(mycc, t1, t2, l1, l2, r1, r2):
    nocc = mycc._scf.mol.nelectron // 2
    nmo = mycc._scf.mo_coeff.shape[1]
    nvir = nmo - nocc 
    oo = np.zeros((nocc, nocc), dtype=float)
    delta = np.identity(nocc)
    oo += -1.0*einsum('ia,jk,ka->ij', l1, delta, r1)
    oo += 1.0*einsum('bj,ikab,ka->ij', t1.T, l2, r1)
    oo += -1.0*einsum('ikba,jl,klab->ij', l2, delta, r2)
    oo += 1.0*einsum('ilba,jk,klab->ij', l2, delta, r2)
    return oo


def get_vo(mycc, t1, t2, l1, l2, r1, r2):
    nocc = mycc._scf.mol.nelectron // 2
    nmo = mycc._scf.mo_coeff.shape[1]
    nvir = nmo - nocc 
    vo = np.zeros((nvir, nocc), dtype=float)
    delta_o = np.identity(nocc)
    delta_v = np.identity(nvir)
    t2 = t2.transpose(2,3,0,1)
    
    vo += 1.0*einsum('ab,ij,jb->ai', delta_v, delta_o, r1)
    vo += 1.0*einsum('ai,jb,jb->ai', t1.T, l1, r1)
    vo += 1.0*einsum('caik,kjbc,jb->ai', t2, l2, r1)
    vo += -1.0*einsum('ak,kb,ij,jb->ai', t1.T, l1, delta_o, r1)
    vo += -1.0*einsum('ci,jc,ab,jb->ai', t1.T, l1, delta_v, r1)
    vo += 1.0*einsum('ak,ci,kjbc,jb->ai', t1.T, t1.T, l2, r1)
    vo += -0.5*einsum('cakl,lkbc,ij,jb->ai', t2, l2, delta_o, r1)
    vo += -0.5*einsum('cdik,kjdc,ab,jb->ai', t2, l2, delta_v, r1)

    vo += 1.0*einsum('ai,kjcb,jkbc->ai', t1.T, l2, r2)
    vo += 1.0*einsum('jb,ik,ac,jkbc->ai', l1, delta_o, delta_v, r2)
    vo += -1.0*einsum('jc,ik,ab,jkbc->ai', l1, delta_o, delta_v, r2)
    vo += -1.0*einsum('kb,ac,ij,jkbc->ai', l1, delta_v, delta_o, r2)
    vo += 1.0*einsum('kc,ij,ab,jkbc->ai', l1, delta_o, delta_v, r2)
    vo += -1.0*einsum('al,ljcb,ik,jkbc->ai', t1.T, l2, delta_o, r2)
    vo += 1.0*einsum('al,lkcb,ij,jkbc->ai', t1.T, l2, delta_o, r2)
    vo += 1.0*einsum('di,kjbd,ac,jkbc->ai', t1.T, l2, delta_v, r2)
    vo += -1.0*einsum('di,kjcd,ab,jkbc->ai', t1.T, l2, delta_v, r2)
    return vo


if __name__ == "__main__":
    from pyscf import gto, scf, ci
    from myadd.trdipole_exp import cisd, run_eomee2
    mol = gto.Mole()
    mol.verbose = 0
    mol.unit = 'A'
    mol.atom = 'S 0 0 0; H 0.958 0.0 0.0; H 0.240 0.927 0.0;'
    #mol.atom = 'Li 0 0 0; Li 0 0 1.0'
    #mol.atom = 'H 0 0 0; H 0 0 1.0; H 0 0 2; H 0 0 3;'
    #mol.atom = 'Kr 0 0 0;'
    mol.basis = '6-31g'
    mol.build()
    for dir in [0, 1, 2]:
        mycc, dip, t1, t2, l1, l2, r1, r2 = run_eomee2(mol, dir=dir)
        dm1 = trans_rdm1(mycc, t1, t2, l1, l2, r1, r2)
        trdip = np.einsum("ij,ij->", dip, dm1) * 2

        cisd(mol, dir=dir)

        print(f"CCSD: {dir}", trdip)
