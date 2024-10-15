from pyscf.ci.cisd import overlap, amplitudes_to_cisdvec, trans_rdm1, CISD
from pyscf.cc.eom_rccsd import _IMDS, EOMEESinglet, EOMEETriplet, EOMEE
from pyscf.cc.ccsd import _ChemistsERIs
import numpy as np


def get_transition_dipole(mycc, imds, t1, t2, l1, l2, r1, r2):
    l0, r0 = np.zeros((1)), np.zeros((1))
    lamda_cisd = amplitudes_to_cisdvec(l0, l1, l2)
    eom_cc = EOMEESinglet(mycc)
    matvec = eom_cc.gen_matvec(imds=imds)
    vec = eom_cc.amplitudes_to_vector(r1, r2)
    vec2 = eom_cc.amplitudes_to_vector(r1, r2)
    vec = matvec(vec)
    hr1, hr2 = eom_cc.vector_to_amplitudes(vec)
    hr_cisd = amplitudes_to_cisdvec(r0, hr1, hr2)
    nmo = mycc._scf.mo_coeff.shape[1]
    nocc = mycc._scf.mol.nelectron // 2
    trdip = overlap(lamda_cisd, hr_cisd, nmo, nocc)

    myci = CISD(mycc._scf)
    dm1 = trans_rdm1(myci, lamda_cisd, vec2)
    fov, foo, fvv = imds.Fov, imds.Foo, imds.Fvv
    fvo = fov.T
    dipole1 = np.concatenate([foo, fov], axis=1)
    dipole2 = np.concatenate([fvo, fvv], axis=1)
    dipole = np.concatenate([dipole1, dipole2], axis=0)
    trdm = np.einsum("ij,ij->", dm1, dipole)
    print("CCSD: ", trdm)
    return trdip


def run_eomee():
    from pyscf import gto, scf, cc
    mol = gto.Mole()
    mol.verbose = 5
    mol.unit = 'A'
    mol.atom = 'O 0 0 0; O 0 0 1.2'
    mol.basis = 'sto-3g'
    mol.build()
    
    mf = scf.RHF(mol)
    mf.verbose = 7
    mf.scf()
    
    mycc = cc.RCCSD(mf)
    mycc.verbose = 7
    mycc.ccsd()
    t1, t2 = mycc.t1, mycc.t2

    l1, l2 = mycc.solve_lambda(t1=t1, t2=t2)
    
    eip,cip = mycc.ipccsd(nroots=1)
    eea,cea = mycc.eaccsd(nroots=1)
    eee,cee = mycc.eeccsd(nroots=1)

    eom_cc = EOMEETriplet(mycc)
    r1, r2 = eom_cc.vector_to_amplitudes(cee)

    dipole = mol.intor("int1e_r", comp=3)[0]

    nmo = mycc._scf.mo_coeff.shape[1]
    nocc = mycc._scf.mol.nelectron // 2

    eris = mycc.ao2mo(mo_coeff=np.zeros((nmo, nmo)))
    eris.fock = np.einsum("ij,ia,jb->ab", dipole, mf.mo_coeff, mf.mo_coeff)
    #eris = fill_zero(eris, nocc, nmo)
    imds = eom_cc.make_imds(eris)
    return mycc, imds, t1, t2, l1, l2, r1, r2


def cisd():
    from pyscf import gto, scf, ci
    mol = gto.Mole()
    mol.verbose = 5
    mol.unit = 'A'
    mol.atom = 'O 0 0 0; O 0 0 1.2'
    mol.basis = 'sto-3g'
    mol.build()
    
    mf = scf.RHF(mol)
    mf.verbose = 7
    mf.scf()

    myci = ci.CISD(mf)
    myci.nroots = 2
    es, cs = myci.kernel()

    dm1 = trans_rdm1(myci, cs[0], cs[1])
    dipole = mol.intor("int1e_r", comp=3)[0]
    trdm = np.einsum("ij,ij->", dm1, dipole)
    print("CISD: ", trdm)
    return


def fill_zero(eris, o, nmo):
    v = nmo - o
    eris.oooo = np.zeros((o, o, o, o))
    eris.ovoo = np.zeros((o, v, o, o))
    eris.oovv = np.zeros((o, o, v, v))
    eris.ovvo = np.zeros((o, v, v, o))
    eris.ovov = np.zeros((o, v, o, v))
    eris.ovvv = np.zeros((o, v, v, v))
    eris.vvvv = np.zeros((v, v, v, v))
    return eris


if __name__ == "__main__":
    mycc, imds, t1, t2, l1, l2, r1, r2 = run_eomee()
    get_transition_dipole(mycc, imds, t1, t2, l1, l2, r1, r2)

    cisd()