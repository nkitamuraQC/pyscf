from numpy import einsum
import numpy as np

def transform_int1e(mycc, t1, t2, int1e_seps):
    nelec = mycc._scf.mol.nelectron
    f_oo, f_vo, f_ov, f_vv = int1e_seps
    f_oo_ = f_oo
    f_vo_ = f_vo
    
    f_vo_ = 1.0*einsum('ia->ia', f_ov).T

    f_vv_ = 1.0*einsum('ab->ab', f_vv)
    f_vv_ += -1.0*einsum('ai,ib->ab', t1.T, f_ov)

    f_oo_ = 1.0*einsum('ij->ij', f_oo)
    f_oo_ += 1.0*einsum('aj,ia->ij', t1.T, f_ov)

    nmo = mycc._scf.mo_coeff.shape[1]
    int1e = np.zeros((nmo, nmo), dtype=float)
    
    int1e[:nelec, :nelec] = f_oo_
    int1e[nelec:, :nelec] = f_vo_
    int1e[:nelec, nelec:] = f_vo_.T
    int1e[nelec:, nelec:] = f_vv_
    return int1e
