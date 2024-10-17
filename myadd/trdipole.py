from pyscf.ci.cisd import overlap, amplitudes_to_cisdvec, trans_rdm1, CISD, cisdvec_to_amplitudes
from pyscf.cc.eom_rccsd import _IMDS, EOMEESinglet, EOMEETriplet, EOMEE, eeccsd_matvec_triplet, eeccsd_matvec_singlet
from pyscf.cc.ccsd import _ChemistsERIs
from pyscf.fci import FCI
from pyscf.mcscf import CASCI
from pyscf.fci.direct_spin1 import trans_rdm1 as fci_trans_rdm1
import numpy as np


def get_transition_dipole(mycc, imds_dip, t1, t2, l1, l2, r1, r2):
    l0, r0 = np.zeros((1)), np.zeros((1))
    lamda_cisd = amplitudes_to_cisdvec(l0, l1, l2)
    eom_cc = EOMEETriplet(mycc)
    #matvec, _ = eom_cc.gen_matvec(imds=imds)
    vec = eom_cc.amplitudes_to_vector(r1, r2)
    vec = eeccsd_matvec_triplet(eom_cc, vec, imds)
    hr1, hr2 = eom_cc.vector_to_amplitudes(vec)
    vec1, vec2 = eom_cc.vector_to_amplitudes(vec)
    hr2 = (hr2[0] + hr2[1]) / 2
    vec2 = (vec2[0] + vec2[1]) / 2
    hr_cisd = amplitudes_to_cisdvec(r0, hr1, hr2)
    vec3 = amplitudes_to_cisdvec(r0, vec1, vec2)
    nmo = mycc._scf.mo_coeff.shape[1]
    nocc = mycc._scf.mol.nelectron // 2
    print(lamda_cisd.shape, hr_cisd.shape)
    cisdvec_to_amplitudes(hr_cisd, nmo, nocc)
    trdip = overlap(lamda_cisd, hr_cisd, nmo, nocc)


    #### こちらが有望
    myci = CISD(mycc._scf)
    dm1 = trans_rdm1(myci, lamda_cisd, vec3, nmo=nmo, nocc=nocc)
    fov, foo, fvv = imds_dip.Fov, imds_dip.Foo, imds_dip.Fvv
    fvo = fov.T
    dipole1 = np.concatenate([foo, fov], axis=1)
    dipole2 = np.concatenate([fvo, fvv], axis=1)
    dipole = np.concatenate([dipole1, dipole2], axis=0)
    trdip2 = np.einsum("ij,ij->", dm1, dipole)
    return trdip, trdip2


def run_eomee(mol):
    from pyscf import gto, scf, cc
    
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
    eee,cee = mycc.eeccsd(nroots=3)

    eom_cc = EOMEETriplet(mycc)
    print(cee)
    r1, r2 = eom_cc.vector_to_amplitudes(cee[0])
    #r1, r2 = eom_cc.vector_to_amplitudes(cee[2])

    dipole = mol.intor("int1e_r", comp=3)[2]

    nmo = mycc._scf.mo_coeff.shape[1]
    nocc = mycc._scf.mol.nelectron // 2

    eris = mycc.ao2mo(mo_coeff=np.zeros((nmo, nmo)))
    #eris = mycc.ao2mo()
    eris.fock = np.einsum("ij,ia,jb->ab", dipole, mf.mo_coeff, mf.mo_coeff)
    #eris = fill_zero(eris, nocc, nmo)
    imds_dip = eom_cc.make_imds(eris)
    print(r1, len(r2))
    #r2 = (r2[0] + r2[1]) / 2
    return mycc, imds_dip, t1, t2, l1, l2, r1, r2


def cisd(mol):
    
    mf = scf.RHF(mol)
    mf.verbose = 7
    mf.scf()

    myci = ci.CISD(mf)
    myci.nroots = 2
    es, cs = myci.kernel()

    dm1 = trans_rdm1(myci, cs[0], cs[1])
    dipole = mol.intor("int1e_r", comp=3)[2]
    dipole = np.einsum("ij,ia,jb->ab", dipole, mf.mo_coeff, mf.mo_coeff)
    trdip = np.einsum("ij,ij->", dm1, dipole)
    print("CISD: ", trdip)

    f = FCI(mf)
    #f = CASCI(mf, 4, 4)
    #f.fcisolver.nroots = 2
    f.nroots = 2
    e, c = f.kernel()
    dm1 = fci_trans_rdm1(c[0], c[1], mf.mo_coeff.shape[1], mf.mol.nelectron)

    dipole = mol.intor("int1e_r", comp=3)[2]
    dipole = np.einsum("ij,ia,jb->ab", dipole, mf.mo_coeff, mf.mo_coeff)
    trdip = np.einsum("ij,ij->", dm1, dipole)
    print("FCI: ", trdip)
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
    from pyscf import gto, scf, ci
    mol = gto.Mole()
    mol.verbose = 5
    mol.unit = 'A'
    mol.atom = 'N 0 0 0; N 0 0 1.2'
    mol.basis = 'sto-3g'
    mol.build()
    mycc, imds, t1, t2, l1, l2, r1, r2 = run_eomee(mol)
    trdip = get_transition_dipole(mycc, imds, t1, t2, l1, l2, r1, r2)

    cisd(mol)

    print("CCSD: ", trdip)