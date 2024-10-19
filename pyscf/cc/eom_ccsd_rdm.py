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
    from pyscf import gto, scf, ci, cc, tdscf

    def benchmark(mol):
        mf = scf.RHF(mol)
        mf.verbose = 0
        mf.scf()
    
        mytd = tdscf.TDRHF(mf)
        mytd.kernel()
        trdip_td = mytd.transition_dipole()[0]
    
        myci = ci.CISD(mf)
        myci.nroots = 2
        es, cs = myci.kernel()
    
        dm1 = ci.cisd.trans_rdm1(myci, cs[0], cs[1])
        dipole = mol.intor_symmetric("int1e_r", comp=3)
        dipole = np.einsum("xij,ia,jb->xab", dipole, mf.mo_coeff, mf.mo_coeff)
        trdip_ci = np.einsum("ij,xij->x", dm1, dipole) * 2
        return trdip_td, trdip_ci


    def run_eomee(mol):
        mf = scf.RHF(mol)
        mf.verbose = 0
        mf.scf()
        mycc = cc.RCCSD(mf)
        mycc.verbose = 0
        mycc.ccsd()
        t1, t2 = mycc.t1, mycc.t2
        l1, l2 = mycc.solve_lambda(t1=t1, t2=t2)
        eee, cee = mycc.eeccsd(nroots=1)
        eom_cc = cc.eom_rccsd.EOMEETriplet(mycc)
        r1, r2 = eom_cc.vector_to_amplitudes(cee)
        r2 = (r2[0] + r2[1]) / 2
        dipole = mol.intor_symmetric("int1e_r", comp=3)
        dipole = np.einsum("xij,ia,jb->xab", dipole, mf.mo_coeff, mf.mo_coeff)
        return mycc, dipole, t1, t2, l1, l2, r1, r2

    mol = gto.Mole()
    mol.verbose = 0
    mol.unit = 'A'
    mol.atom = 'O 0 0 0; H 0.958 0.0 0.0; H 0.240 0.927 0.0;'
    #mol.atom = 'Li 0 0 0; Li 0 0 1.0'
    #mol.atom = 'H 0 0 0; H 0 0 1.0; H 0 0 2; H 0 0 3;'
    #mol.atom = 'Kr 0 0 0;'
    mol.basis = '6-31g'
    mol.build()

    mycc, dip, t1, t2, l1, l2, r1, r2 = run_eomee(mol)
    dm1 = trans_rdm1(mycc, t1, t2, l1, l2, r1, r2)
    trdip_cc = np.einsum("xij,ij->x", dip, dm1) * 2

    trdip_td, trdip_ci = benchmark(mol)
    print("######################")
    for dir in [0, 1, 2]:
        print(f"CCSD: {dir}", trdip_cc[dir])
        print(f"CISD: {dir}", trdip_ci[dir])
        print(f"TDDFT: {dir}", trdip_td[dir])
        print("######################")
    print(np.argsort(trdip_cc))
    print(np.argsort(trdip_ci))
    print(np.argsort(trdip_td))
