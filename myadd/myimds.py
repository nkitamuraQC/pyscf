from pyscf.lib import logger, module_method
from pyscf.cc import rintermediates as imd
from pyscf import lib
import numpy as np
from pyscf.cc import ccsd

class _IMDS:
    def __init__(self, cc, eris=None):
        self.verbose = cc.verbose
        self.stdout = cc.stdout
        self.max_memory = cc.max_memory
        self.t1 = cc.t1
        self.t2 = cc.t2
        if eris is None:
            eris = cc.ao2mo()
        self.eris = eris
        self._made_shared_2e = False

    def _make_shared_1e(self):
        cput0 = (logger.process_clock(), logger.perf_counter())

        t1, t2, eris = self.t1, self.t2, self.eris
        self.Loo = imd.Loo(t1, t2, eris)
        self.Lvv = imd.Lvv(t1, t2, eris)
        self.Fov = imd.cc_Fov(t1, t2, eris)

        logger.timer_debug1(self, 'EOM-CCSD shared one-electron '
                            'intermediates', *cput0)
        return self

    def _make_shared_2e(self):
        cput0 = (logger.process_clock(), logger.perf_counter())
        log = logger.Logger(self.stdout, self.verbose)

        t1, t2, eris = self.t1, self.t2, self.eris
        # 2 virtuals
        self.Wovov = imd.Wovov(t1, t2, eris)
        self.Wovvo = imd.Wovvo(t1, t2, eris)
        self.Woovv = np.asarray(eris.ovov).transpose(0,2,1,3)

        self._made_shared_2e = True
        log.timer_debug1('EOM-CCSD shared two-electron intermediates', *cput0)
        return self

    def make_ip(self, ip_partition=None):
        self._make_shared_1e()
        if not self._made_shared_2e and ip_partition != 'mp':
            self._make_shared_2e()

        cput0 = (logger.process_clock(), logger.perf_counter())
        log = logger.Logger(self.stdout, self.verbose)

        t1, t2, eris = self.t1, self.t2, self.eris

        # 0 or 1 virtuals
        if ip_partition != 'mp':
            self.Woooo = imd.Woooo(t1, t2, eris)
        self.Wooov = imd.Wooov(t1, t2, eris)
        self.Wovoo = imd.Wovoo(t1, t2, eris)
        log.timer_debug1('EOM-CCSD IP intermediates', *cput0)
        return self

    def make_t3p2_ip(self, cc, ip_partition=None):
        assert (ip_partition is None)
        cput0 = (logger.process_clock(), logger.perf_counter())

        t1, t2, eris = cc.t1, cc.t2, self.eris
        delta_E_corr, pt1, pt2, Wovoo, Wvvvo = \
            imd.get_t3p2_imds_slow(cc, t1, t2, eris)
        self.t1 = pt1
        self.t2 = pt2

        self._made_shared_2e = False  # Force update
        self.make_ip()  # Make after t1/t2 updated
        self.Wovoo = self.Wovoo + Wovoo

        logger.timer_debug1(self, 'EOM-CCSD(T)a IP intermediates', *cput0)
        return self


    def make_ea(self, ea_partition=None):
        self._make_shared_1e()
        if not self._made_shared_2e and ea_partition != 'mp':
            self._make_shared_2e()

        cput0 = (logger.process_clock(), logger.perf_counter())
        log = logger.Logger(self.stdout, self.verbose)

        t1, t2, eris = self.t1, self.t2, self.eris

        # 3 or 4 virtuals
        self.Wvovv = imd.Wvovv(t1, t2, eris)
        if ea_partition == 'mp':
            self.Wvvvo = imd.Wvvvo(t1, t2, eris)
        else:
            self.Wvvvv = imd.Wvvvv(t1, t2, eris)
            self.Wvvvo = imd.Wvvvo(t1, t2, eris, self.Wvvvv)
        log.timer_debug1('EOM-CCSD EA intermediates', *cput0)
        return self

    def make_t3p2_ea(self, cc, ea_partition=None):
        assert (ea_partition is None)
        cput0 = (logger.process_clock(), logger.perf_counter())

        t1, t2, eris = cc.t1, cc.t2, self.eris
        delta_E_corr, pt1, pt2, Wovoo, Wvvvo = \
            imd.get_t3p2_imds_slow(cc, t1, t2, eris)
        self.t1 = pt1
        self.t2 = pt2

        self._made_shared_2e = False  # Force update
        self.make_ea()  # Make after t1/t2 updated
        self.Wvvvo = self.Wvvvo + Wvvvo

        logger.timer_debug1(self, 'EOM-CCSD(T)a EA intermediates', *cput0)
        return self


    def make_ee(self):
        cput0 = (logger.process_clock(), logger.perf_counter())
        log = logger.Logger(self.stdout, self.verbose)

        t1, t2, eris = self.t1, self.t2, self.eris
        dtype = np.result_type(t1, t2)
        if np.iscomplexobj(t2):
            raise NotImplementedError('Complex integrals are not supported in EOM-EE-CCSD')

        nocc, nvir = t1.shape

        fswap = lib.H5TmpFile()
        self.saved = lib.H5TmpFile()
        self.wvOvV = self.saved.create_dataset('wvOvV', (nvir,nocc,nvir,nvir), dtype.char)
        self.woVvO = self.saved.create_dataset('woVvO', (nocc,nvir,nvir,nocc), dtype.char)
        self.woVVo = self.saved.create_dataset('woVVo', (nocc,nvir,nvir,nocc), dtype.char)
        self.woOoV = self.saved.create_dataset('woOoV', (nocc,nocc,nocc,nvir), dtype.char)

        foo = eris.fock[:nocc,:nocc]
        fov = eris.fock[:nocc,nocc:]
        fvv = eris.fock[nocc:,nocc:]

        self.Fov = np.zeros((nocc,nvir), dtype=dtype)
        self.Foo = np.zeros((nocc,nocc), dtype=dtype)
        self.Fvv = np.zeros((nvir,nvir), dtype=dtype)

        #:eris_ovvv = lib.unpack_tril(np.asarray(eris.ovvv).reshape(nocc*nvir,nvir**2)).reshape(nocc,nvir,nvir,nvir)
        #:self.Fvv  = np.einsum('mf,mfae->ae', t1, eris_ovvv) * 2
        #:self.Fvv -= np.einsum('mf,meaf->ae', t1, eris_ovvv)
        #:self.woVvO = lib.einsum('jf,mebf->mbej', t1, eris_ovvv)
        #:self.woVVo = lib.einsum('jf,mfbe->mbej',-t1, eris_ovvv)
        #:tau = _make_tau(t2, t1, t1)
        #:self.woVoO  = 0.5 * lib.einsum('mebf,ijef->mbij', eris_ovvv, tau)
        #:self.woVoO += 0.5 * lib.einsum('mfbe,ijfe->mbij', eris_ovvv, tau)
        eris_ovoo = np.asarray(eris.ovoo)
        woVoO = np.empty((nocc,nvir,nocc,nocc), dtype=dtype)
        tau = _make_tau(t2, t1, t1)
        theta = t2*2 - t2.transpose(0,1,3,2)

        mem_now = lib.current_memory()[0]
        max_memory = max(0, self.max_memory - mem_now)
        blksize = min(nocc, max(ccsd.BLKMIN, int(max_memory*1e6/8/(nvir**3*3))))
        for seg, (p0,p1) in enumerate(lib.prange(0, nocc, blksize)):
            ovvv = eris.get_ovvv(slice(p0,p1))  # ovvv = eris.ovvv[p0:p1]
            # transform integrals (ia|bc) -> (ac|ib)
            fswap['ebmf/%d'%seg] = np.einsum('mebf->ebmf', ovvv)

            self.Fvv += np.einsum('mf,mfae->ae', t1[p0:p1], ovvv) * 2
            self.Fvv -= np.einsum('mf,meaf->ae', t1[p0:p1], ovvv)
            woVoO[p0:p1] = lib.einsum('mebf,ijef->mbij', ovvv, tau)
            woVvO = lib.einsum('jf,mebf->mbej', t1, ovvv)
            woVVo = lib.einsum('jf,mfbe->mbej',-t1, ovvv)
            ovvv = None

            eris_ovov = np.asarray(eris.ovov[p0:p1])
            woOoV = lib.einsum('if,mfne->mnie', t1, eris_ovov)
            woOoV+= eris_ovoo[:,:,p0:p1].transpose(2,0,3,1)
            self.woOoV[p0:p1] = woOoV
            woOoV = None

            tmp = lib.einsum('njbf,mfne->mbej', t2, eris_ovov)
            woVvO -= tmp * .5
            woVVo += tmp

            ovoo = lib.einsum('menf,jf->menj', eris_ovov, t1)
            woVvO -= lib.einsum('nb,menj->mbej', t1, ovoo)
            ovoo = lib.einsum('mfne,jf->menj', eris_ovov, t1)
            woVVo += lib.einsum('nb,menj->mbej', t1, ovoo)
            ovoo = None

            ovov = eris_ovov * 2 - eris_ovov.transpose(0,3,2,1)
            woVvO += lib.einsum('njfb,menf->mbej', theta, ovov) * .5

            self.Fov[p0:p1] = np.einsum('nf,menf->me', t1, ovov)
            tilab = np.einsum('ia,jb->ijab', t1[p0:p1], t1) * .5
            tilab += t2[p0:p1]
            self.Foo += lib.einsum('mief,menf->ni', tilab, ovov)
            self.Fvv -= lib.einsum('mnaf,menf->ae', tilab, ovov)
            eris_ovov = ovov = tilab = None

            woVvO -= lib.einsum('nb,menj->mbej', t1, eris_ovoo[p0:p1,:,:])
            woVVo += lib.einsum('nb,nemj->mbej', t1, eris_ovoo[:,:,p0:p1])

            woVvO += np.asarray(eris.ovvo[p0:p1]).transpose(0,2,1,3)
            woVVo -= np.asarray(eris.oovv[p0:p1]).transpose(0,2,3,1)

            self.woVvO[p0:p1] = woVvO
            self.woVVo[p0:p1] = woVVo

        self.Fov += fov
        self.Foo += foo + 0.5*np.einsum('me,ie->mi', self.Fov+fov, t1)
        self.Fvv += fvv - 0.5*np.einsum('me,ma->ae', self.Fov+fov, t1)

        # 0 or 1 virtuals
        woOoO = lib.einsum('je,nemi->mnij', t1, eris_ovoo)
        woOoO = woOoO + woOoO.transpose(1,0,3,2)
        woOoO += np.asarray(eris.oooo).transpose(0,2,1,3)

        tmp = lib.einsum('meni,jneb->mbji', eris_ovoo, t2)
        woVoO -= tmp.transpose(0,1,3,2) * .5
        woVoO -= tmp
        tmp = None
        ovoo = eris_ovoo*2 - eris_ovoo.transpose(2,1,0,3)
        woVoO += lib.einsum('nemi,njeb->mbij', ovoo, theta) * .5
        self.Foo += np.einsum('ne,nemi->mi', t1, ovoo)
        ovoo = None

        eris_ovov = np.asarray(eris.ovov)
        woOoO += lib.einsum('ijef,menf->mnij', tau, eris_ovov)
        self.woOoO = self.saved['woOoO'] = woOoO
        woVoO -= lib.einsum('nb,mnij->mbij', t1, woOoO)
        woOoO = None

        tmpoovv = lib.einsum('njbf,nemf->ejmb', t2, eris_ovov)
        ovov = eris_ovov*2 - eris_ovov.transpose(0,3,2,1)
        eris_ovov = None

        tmpovvo = lib.einsum('nifb,menf->eimb', theta, ovov)
        ovov = None

        tmpovvo *= -.5
        tmpovvo += tmpoovv * .5
        woVoO -= lib.einsum('ie,ejmb->mbij', t1, tmpovvo)
        woVoO -= lib.einsum('ie,ejmb->mbji', t1, tmpoovv)
        woVoO += eris_ovoo.transpose(3,1,2,0)

        # 3 or 4 virtuals
        eris_ovvo = np.asarray(eris.ovvo)
        tmpovvo -= eris_ovvo.transpose(1,3,0,2)
        fswap['ovvo'] = tmpovvo
        tmpovvo = None

        eris_oovv = np.asarray(eris.oovv)
        tmpoovv -= eris_oovv.transpose(3,1,0,2)
        fswap['oovv'] = tmpoovv
        tmpoovv = None

        woVoO += lib.einsum('mebj,ie->mbij', eris_ovvo, t1)
        woVoO += lib.einsum('mjbe,ie->mbji', eris_oovv, t1)
        woVoO += lib.einsum('me,ijeb->mbij', self.Fov, t2)
        self.woVoO = self.saved['woVoO'] = woVoO
        woVoO = eris_ovvo = eris_oovv = None

        #:theta = t2*2 - t2.transpose(0,1,3,2)
        #:eris_ovvv = lib.unpack_tril(np.asarray(eris.ovvv).reshape(nocc*nvir,nvir**2)).reshape(nocc,nvir,nvir,nvir)
        #:ovvv = eris_ovvv*2 - eris_ovvv.transpose(0,3,2,1)
        #:tmpab = lib.einsum('mebf,miaf->eiab', eris_ovvv, t2)
        #:tmpab = tmpab + tmpab.transpose(0,1,3,2) * .5
        #:tmpab-= lib.einsum('mfbe,mifa->eiba', ovvv, theta) * .5
        #:self.wvOvV += eris_ovvv.transpose(2,0,3,1).conj()
        #:self.wvOvV -= tmpab
        nsegs = len(fswap['ebmf'])
        def load_ebmf(slice):
            dat = [fswap['ebmf/%d'%i][slice] for i in range(nsegs)]
            return np.concatenate(dat, axis=2)

        mem_now = lib.current_memory()[0]
        max_memory = max(0, self.max_memory - mem_now)
        blksize = min(nocc, max(ccsd.BLKMIN, int(max_memory*1e6/8/(nocc*nvir**2*4))))
        for p0, p1 in lib.prange(0, nvir, blksize):
            #:wvOvV  = lib.einsum('mebf,miaf->eiab', ovvv, t2)
            #:wvOvV += lib.einsum('mfbe,miaf->eiba', ovvv, t2)
            #:wvOvV -= lib.einsum('mfbe,mifa->eiba', ovvv, t2)*2
            #:wvOvV += lib.einsum('mebf,mifa->eiba', ovvv, t2)

            ebmf = load_ebmf(slice(p0, p1))
            wvOvV = lib.einsum('ebmf,miaf->eiab', ebmf, t2)
            wvOvV = -.5 * wvOvV.transpose(0,1,3,2) - wvOvV

            # Using the permutation symmetry (em|fb) = (em|bf)
            efmb = load_ebmf((slice(None), slice(p0, p1)))
            wvOvV += np.einsum('ebmf->bmfe', efmb.conj())

            # tmp = (mf|be) - (me|bf)*.5
            tmp = -.5 * ebmf
            tmp += efmb.transpose(1,0,2,3)
            ebmf = None
            wvOvV += lib.einsum('efmb,mifa->eiba', tmp, theta)
            tmp = None

            wvOvV += lib.einsum('meni,mnab->eiab', eris_ovoo[:,p0:p1], tau)
            wvOvV -= lib.einsum('me,miab->eiab', self.Fov[:,p0:p1], t2)
            wvOvV += lib.einsum('ma,eimb->eiab', t1, fswap['ovvo'][p0:p1])
            wvOvV += lib.einsum('ma,eimb->eiba', t1, fswap['oovv'][p0:p1])

            self.wvOvV[p0:p1] = wvOvV

        self.made_ee_imds = True
        log.timer('EOM-CCSD EE intermediates', *cput0)
        return self
    

def _make_tau(t2, t1, r1, fac=1, out=None):
    tau = np.einsum('ia,jb->ijab', t1, r1)
    tau = tau + tau.transpose(1,0,3,2)
    tau *= fac * .5
    tau += t2
    return tau