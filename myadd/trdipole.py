from pyscf.ci.cisd import overlap, amplitudes_to_cisdvec
from pyscf.cc.eom_rccsd import _IMDS, EOMEESinglet
from pyscf.cc.ccsd import _ChemistsERIs, ao2mo
import numpy as np


def get_transition_dipole(mycc, imds, t1, t2, l1, l2, r1, r2):
    l0, r0 = np.zeros((1)), np.zeros((1))
    lamda_cisd = amplitudes_to_cisdvec(l0, l1, l2)
    eom_cc = EOMEESinglet(mycc)
    matvec = eom_cc.gen_matvec(imds=imds)
    vec = eom_cc.amplitudes_to_vector(r1, r2)
    vec = matvec(vec)
    hr1, hr2 = eom_cc.vector_to_amplitudes(vec)
    hr_cisd = amplitudes_to_cisdvec(r0, hr1, hr2)
    trdip = overlap(lamda_cisd, hr_cisd)
    return trdip


def run_eomee():
    from pyscf import gto, scf, cc
    mol = gto.Mole()
    mol.verbose = 5
    mol.unit = 'A'
    mol.atom = 'O 0 0 0; O 0 0 1.2'
    mol.basis = 'ccpvdz'
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

    eom_cc = EOMEESinglet(mycc)
    r1, r2 = eom_cc.vector_to_amplitudes(cee)

    dipole = mol.intor("int1e_r", comp=3)[0]

    eris = mycc.ao2mo()
    eris.fock = np.einsum("ij,ia,jb->ab", dipole, mf.mo_coeff, mf.mo_coeff)
    eris = fill_zero(eris)
    imds = _IMDS(mycc, eris)
    imds = imds.make_ee()
    return mycc, imds, t1, t2, l1, l2, r1, r2


def fill_zero(eris):
    return