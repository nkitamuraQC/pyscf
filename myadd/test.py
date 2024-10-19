# test_calc.py

import unittest
from trdm_wick import trans_rdm1
from pyscf import gto, scf, cc, tdscf, ci
import numpy as np

def benchmark(mol):
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.scf()

    mytd = tdscf.TDRHF(mf)
    mytd.kernel()
    trdip_tddft = mytd.transition_dipole()[0]

    myci = ci.CISD(mf)
    myci.nroots = 2
    es, cs = myci.kernel()

    dm1 = ci.cisd.trans_rdm1(myci, cs[0], cs[1])
    dipole = mol.intor("int1e_r", comp=3)
    dipole = np.einsum("xij,ia,jb->xab", dipole, mf.mo_coeff, mf.mo_coeff)
    trdip_cisd = np.einsum("ij,xij->x", dm1, dipole) * 2
    return trdip_tddft, trdip_cisd


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


def template(mol):
    dip_td, dip_ci = benchmark(mol)
    mycc, dipole, t1, t2, l1, l2, r1, r2 = run_eomee(mol)
    dm1 = trans_rdm1(mycc, t1, t2, l1, l2, r1, r2)
    trdip_eom_cc = np.einsum("ji,xij->x", dm1, dipole) * 2
    td_sort = np.argsort(dip_td)
    ci_sort = np.argsort(dip_ci)
    cc_sort = np.argsort(trdip_eom_cc)

    print("cc: ", trdip_eom_cc)
    print("ci: ", dip_ci)
    print("tddft: ", dip_td)
    
    assert td_sort[0] == cc_sort[0]
    assert td_sort[1] == cc_sort[1]
    return



class TestCalc(unittest.TestCase):

    def test_Li2(self):
        mol = gto.Mole()
        mol.verbose = 0
        mol.unit = 'A'
        mol.atom = 'Li 0 0 0; Li 0 0 1.0'
        mol.basis = 'def2-svp'
        mol.build()
        template(mol)

    def test_LiH(self):
        mol = gto.Mole()
        mol.verbose = 0
        mol.unit = 'A'
        mol.atom = 'Li 0 0 0; H 0 0 1.0'
        mol.basis = 'def2-svp'
        mol.build()
        template(mol)

    def test_HCl(self):
        mol = gto.Mole()
        mol.verbose = 0
        mol.unit = 'A'
        mol.atom = 'H 0 0 0; Cl 0 0 1.0'
        mol.basis = 'def2-svp'
        mol.build()
        template(mol)

    def test_H2O(self):
        mol = gto.Mole()
        mol.verbose = 0
        mol.unit = 'A'
        mol.atom = 'O 0 0 0; H 0.958 0.0 0.0; H 0.240 0.927 0.0;'
        mol.basis = 'def2-svp'
        mol.build()
        template(mol)

    def test_HF(self):
        mol = gto.Mole()
        mol.verbose = 0
        mol.unit = 'A'
        mol.atom = 'H 0 0 0; F 0 0 1.0'
        mol.basis = 'def2-svp'
        mol.build()
        template(mol)

    def test_CO(self):
        mol = gto.Mole()
        mol.verbose = 0
        mol.unit = 'A'
        mol.atom = 'C 0 0 0; O 0 0 1.0'
        mol.basis = '6-31g'
        mol.build()
        template(mol)

    def test_H2S(self):
        mol = gto.Mole()
        mol.verbose = 0
        mol.unit = 'A'
        mol.atom = 'S 0 0 0; H 0.958 0.0 0.0; H 0.240 0.927 0.0;'
        mol.basis = '6-31g'
        mol.build()
        template(mol)

if __name__ == '__main__':
    unittest.main()
