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
from os import path
from scipy import interpolate


def phase_feats_mel_unwarp(m_ph_mel, alpha, ncoeffs_mag):
    ncoeffs_cmp = m_ph_mel.shape[1]
    f_intrp_ph = interpolate.interp1d(np.arange(ncoeffs_cmp), m_ph_mel, kind='nearest', fill_value='extrapolate')
    m_ph_mel = f_intrp_ph(np.arange(ncoeffs_mag))
    m_ph = la.sp_mel_unwarp(m_ph_mel, fft_len_half, alpha=alpha, in_type='log')

    return m_ph

if __name__ == '__main__':  

    # CONSTANTS:==========================================================================

    # INPUT:==============================================================================
    filename  = 'hvd_593'
    feats_dir = 'data_48k/data_predict/pf'
    wavs_syn_dir = 'data_48k/wavs_syn_phase_compress'

    fft_len = 4096
    fs = 48000
    alpha = 0.77

    n_cmp_coeffs = 10

    # Setup:========================================================================================
    lu.mkdir(wavs_syn_dir)
    fft_len_half = 1 + fft_len / 2

    # Read parameters:==============================================================================
    m_mag_mel_log = lu.read_binfile(path.join(feats_dir, filename + '.mag'),  dim=60)
    m_real_mel    = lu.read_binfile(path.join(feats_dir, filename + '.real'), dim=45)
    m_imag_mel    = lu.read_binfile(path.join(feats_dir, filename + '.imag'), dim=45)
    v_lf0         = lu.read_binfile(path.join(feats_dir, filename + '.lf0'),  dim=1)

    m_real = phase_feats_mel_unwarp(m_real_mel, alpha, 60)
    m_imag = phase_feats_mel_unwarp(m_imag_mel, alpha, 60)
    m_mag_log = la.sp_mel_unwarp(m_mag_mel_log, fft_len_half, alpha=alpha, in_type='log')

    # Phase compression:============================================================================
    n_ph_coeffs = 45
    m_real_rcep = la.rceps(m_real_mel, in_type='log', out_type='compact')[:,:n_cmp_coeffs]

    m_real_mel_cmp = la.remove_hermitian_half(np.fft.fft(m_real_rcep, n=2*(n_ph_coeffs-1)).real)
    m_real_cmp = phase_feats_mel_unwarp(m_real_mel_cmp, alpha, 60)

    m_imag_rcep = la.rceps(m_imag_mel, in_type='log', out_type='compact')[:,:n_cmp_coeffs]

    m_imag_mel_cmp = la.remove_hermitian_half(np.fft.fft(m_imag_rcep, n=2*(n_ph_coeffs-1)).real)
    m_imag_cmp = phase_feats_mel_unwarp(m_imag_mel_cmp, alpha, 60)

    if False: # PLOTS:
        nx= 250; figure('250 mel - 10'); plot(m_real_mel[nx,:]); plot(m_real_mel_cmp[nx,:]); grid()
        nx= 250; figure('250 linear - 10'); plot(m_real[nx,:]); plot(m_real_cmp[nx,:]); grid()

    if False:
        plm(m_real_mel)
        plm(m_imag_mel)

        plm(m_real)
        plm(m_imag)

        plm(m_real_rcep)

    if False:
        #nx= 100; figure(); plot(m_real_mel[nx,:]); plot(m_imag_mel[nx,:]); grid()

        nx= 250; figure('250'); plot(m_real_mel[nx,:]); plot(m_imag_mel[nx,:]); plot(0.5 * m_mag_mel_log[nx,:]); grid()
        nx= 250; figure('250'); plot(m_real[nx,:]); plot(m_imag[nx,:]); plot(0.5 * m_mag_log[nx,:]); grid()



        #nx= 100; figure(); plot(m_real[nx,:]); plot(m_imag[nx,:]); grid()



    # Synthesis:====================================================================================
    '''
    v_syn_sig = mp.synthesis_from_compressed_type1(m_mag_mel_log, m_real_mel, m_imag_mel, v_lf0, fs)
    la.write_audio_file(path.join(wavs_syn_dir, filename + '.wav'), v_syn_sig, fs)

    v_syn_sig_ph_cmp = mp.synthesis_from_compressed_type1(m_mag_mel_log, m_real_mel_cmp, m_imag_mel_cmp, v_lf0, fs)
    la.write_audio_file(path.join(wavs_syn_dir, filename + '_ph_cmp_1.wav'), v_syn_sig_ph_cmp, fs)
    '''



    print('Done!')



