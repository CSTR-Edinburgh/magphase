#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

@author: Felipe Espic
"""

import sys, os
this_dir = os.getcwd()
sys.path.append(os.path.realpath(this_dir + '/../src'))
import numpy as np
import libutils as lu
import libaudio as la
from libplot import lp
import magphase as mp


if __name__ == '__main__':

    # INPUT:====================================================================
    magfile_ref = '/home/s1373426/Dropbox/Education/UoE/Projects/fft_feats_DirectFFTWaveformModelling/magphase_proj/merlin/tools/magphase_private/demos/data_48k/wavs_syn/hvd_593_copy_syn_lossless.mag'
    magfile_a   = '/home/s1373426/Dropbox/Education/UoE/Projects/fft_feats_DirectFFTWaveformModelling/magphase_proj/merlin/tools/magphase_private/demos/data_48k/wavs_syn/hvd_593_copy_syn_low_dim.mag'

    fft_len = 4096
    alpha = 0.77

    nx = 52 #100

    # PLOTS:====================================================================
    #lp.close('all')

    fft_len_half = fft_len/2+1
    #m_mag_ref = lu.read_binfile(magfile_ref, dim=60)
    m_mag_ref = la.db(lu.read_binfile(magfile_ref, dim=fft_len_half))
    m_mag_a   = la.db(np.exp(la.sp_mel_unwarp(lu.read_binfile(magfile_a, dim=60), fft_len_half, alpha=alpha, in_type='log')))


    if False:
        lp.plotm(m_mag_ref)
        lp.plotm(m_mag_a)
        lp.plotm(m_mag_ref - m_mag_a)
        lp.plotm((m_mag_ref - m_mag_a)[:50,:50])

        lp.plotm(np.diff(m_mag_ref, axis=0))
        lp.plotm(np.diff(m_mag_a, axis=0))


    if True:
        nx=26; lp.figure(); lp.plot(m_mag_ref[nx,:]); lp.plot(m_mag_a[nx,:]); lp.grid()


    if True:
        nx=26; lp.figure(); lp.plot(m_mag_ref[nx,:]); lp.plot(m_mag_a[nx,:]); lp.plot(m_mag_ref[nx-1,:]); lp.plot(m_mag_a[nx-1,:]); lp.grid()
        nx=37; lp.figure(); lp.plot(m_mag_ref[nx,:]); lp.plot(m_mag_a[nx,:]); lp.plot(m_mag_ref[nx-1,:]); lp.plot(m_mag_a[nx-1,:]); lp.grid()


    if True:
        nxb=150; lp.figure(); lp.plot(m_mag_ref[:,nxb]); lp.plot(m_mag_a[:,nxb]); lp.grid()



    # Prue:
    nx_lf = 2
    m_mag_ref_fix = la.log(lu.read_binfile(magfile_ref, dim=fft_len_half))
    m_mag_ref_fix[:,0] =  m_mag_ref_fix[:,nx_lf]
    m_mag_ref_fix[:,1] =  m_mag_ref_fix[:,nx_lf]

    m_mag_ref_fix_mel   = la.sp_mel_warp(m_mag_ref_fix, 60, alpha=0.77, in_type=2)
    m_mag_ref_fix_after = la.sp_mel_unwarp(m_mag_ref_fix_mel, fft_len_half, alpha=alpha, in_type='log')
    m_mag_ref_fix_after = la.db(np.exp(m_mag_ref_fix_after))


    if True:
        lp.plotm(m_mag_ref_fix)
        lp.plotm(m_mag_ref_fix_mel)
        lp.plotm(m_mag_ref_fix_after)

    if True:
        nx=26; lp.figure(); lp.plot(m_mag_ref[nx,:]); lp.plot(m_mag_ref_fix_after[nx,:]); lp.plot(m_mag_ref[nx-1,:]); lp.plot(m_mag_ref_fix_after[nx-1,:]); lp.grid()
        nx=37; lp.figure(); lp.plot(m_mag_ref[nx,:]); lp.plot(m_mag_ref_fix_after[nx,:]); lp.plot(m_mag_ref[nx-1,:]); lp.plot(m_mag_ref_fix_after[nx-1,:]); lp.grid()




    # Study about windows:
    v_win  = np.hanning(500)
    v_fft  = np.fft.fft(v_win, n=fft_len)
    v_spec = np.absolute(v_fft)
    v_ph   = np.angle(v_fft)

    v_spec_1 = (v_spec + 5.0)
    v_spec_2 = 1.0 / (v_spec + 5.0)

    v_ifft_1 = np.fft.ifft(v_spec_1).real
    v_ifft_2 = np.fft.ifft(v_spec_2).real


    if True:
        lp.figure(); lp.plot(v_win); lp.grid()
        lp.figure(); lp.plot(v_spec); lp.grid()
        lp.figure(); lp.plot(v_spec_1); lp.grid()
        lp.figure(); lp.plot(v_spec_2); lp.grid()
        lp.figure(); lp.plot(v_ph); lp.grid()
        lp.figure(); lp.plot(v_ifft_1); lp.grid()
        lp.figure(); lp.plot(v_ifft_2); lp.grid()

