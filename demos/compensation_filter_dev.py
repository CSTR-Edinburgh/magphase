#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Felipe Espic

DESCRIPTION:
This script extracts low-dimensional acoustic parameters from a wave file.
Then, it resynthesises the signal from these features.
Features:
- m_mag_mel_log: Mel-scaled Log-Mag (dim=nbins_mel,   usually 60).
- m_real_mel:    Mel-scaled real    (dim=nbins_phase, usually 45).
- m_imag_mel:    Mel-scaled imag    (dim=nbins_phase, usually 45).
- v_lf0:         Log-F0 (dim=1).

INSTRUCTIONS:
This demo should work out of the box. Just run it by typing: python <script name>
If wanted, you can modify the input options and/or perform some modification to the
extracted features before re-synthesis. See the main function below for details.
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

    # INPUT:==============================================================================
    #wav_file_orig = 'data_48k/wavs_nat/hvd_593.wav'
    #wav_file_orig = 'data_48k/wavs_nat/hvd_594.wav'
    wav_file_orig = 'data_48k/wavs_nat_no_silence/hvd_593_no_sil.wav'

    # PROCESS:============================================================================
    # Analysis orig:
    m_mag, m_real, m_imag, v_f0, fs, v_shift = mp.analysis_lossless(wav_file_orig)
    fft_len_half = m_mag.shape[1]
    fft_len = 2 * (fft_len_half - 1)

    m_mag_mel_log, m_real_mel, m_imag_mel, v_lf0_smth = mp.format_for_modelling(m_mag, m_real, m_imag, v_f0, fs)
    m_mag_smth = np.exp(la.sp_mel_unwarp(m_mag_mel_log, fft_len_half, alpha=0.77, in_type='log'))

    # Synthesis:
    v_lf0 = la.f0_to_lf0(v_f0)
    v_syn_sig = mp.synthesis_from_compressed_type1(m_mag_mel_log, m_real_mel, m_imag_mel, v_lf0, fs)

    # Analysis from synthesised:

    v_pm = la.shift_to_pm(v_shift)
    m_fft_syn, v_shift = mp.analysis_with_del_comp_from_pm(v_syn_sig, fs, v_pm)
    m_mag_syn = np.absolute(m_fft_syn)
    m_mag_syn_mel  = la.sp_mel_warp(m_mag_syn, 60, alpha=0.77, in_type=3)
    m_mag_syn_smth = la.sp_mel_unwarp(m_mag_syn_mel, fft_len_half, alpha=0.77, in_type='abs')

    # Energy per frame:
    v_ener = np.sum(la.add_hermitian_half(m_mag)**2, axis=1) / float(fft_len)

    # PLOTS:=============================================================================
    m_mag_db = la.db(m_mag)
    m_mag_syn_db = la.db(m_mag_syn)
    m_mag_syn_smth_db = la.db(m_mag_syn_smth)
    m_mag_smth_db = la.db(m_mag_smth)
    v_voi = v_f0 > 1.0

    if False:
        plm(m_mag_db)
        plm(m_mag_syn_db)
        plm(m_mag_syn_db - m_mag_db)

    if False:
        nx=38; figure(); plot(m_mag_db[nx,:]); plot(m_mag_syn_db[nx,:]); grid()
        nx=9; figure(); plot(m_mag_db[nx,:]); plot(m_mag_syn_db[nx,:]); plot(m_mag_smth_db[nx,:]); plot(m_mag_syn_smth_db[nx,:]); grid()
        nx=200; figure(); plot(m_mag_db[nx,:]); plot(m_mag_syn_db[nx,:]); plot(m_mag_smth_db[nx,:]); plot(m_mag_syn_smth_db[nx,:]); grid()


    if False:
        nx_a=197; nx_b=199; figure(); plot(m_mag_smth_db[nx_a:nx_b+1,:].T); grid()
        nx_a=197; nx_b=199; figure(); plot(m_mag_syn_smth_db[nx_a:nx_b+1,:].T); grid()

        nx_a=197; nx_b=199; figure(); plot(m_mag_db[nx_a:nx_b+1,:].T); grid()
        nx_a=197; nx_b=199; figure(); plot(m_mag_syn_db[nx_a:nx_b+1,:].T); grid()

    if False:
        nx_a=180; nx_b=227; surf(m_mag_db[nx_a:nx_b+1,:], aratio=2.0)
        nx_a=180; nx_b=227; surf(m_mag_syn_db[nx_a:nx_b+1,:], aratio=2.0)

        # Averages:
        #figure(); plot(np.mean(m_mag_db, axis=1)); plot(np.mean(m_mag_syn_db, axis=1)); grid()
        #figure(); plot(np.mean(m_mag_db, axis=0)); plot(np.mean(m_mag_syn_db, axis=0)); grid()

    if False:
        # Averages per voi/unv:

        figure('voi')
        plot(np.mean(m_mag_db[v_voi,:], axis=0))
        plot(np.mean(m_mag_syn_db[v_voi,:], axis=0))
        plot(np.mean(m_mag_syn_db[v_voi,:], axis=0) - np.mean(m_mag_db[v_voi,:], axis=0))
        grid()

    if False:
        figure('voi with smoothing versions')
        plot(np.mean(m_mag_db[v_voi,:], axis=0))
        plot(np.mean(m_mag_syn_db[v_voi,:], axis=0))
        plot(np.mean(m_mag_smth_db[v_voi,:], axis=0))
        plot(np.mean(m_mag_syn_smth_db[v_voi,:], axis=0))
        grid()

    if True:

        # Bins warping:
        alpha = 0.77
        #v_bins_warp = la.build_mel_curve(alpha, fft_len_half)
        #v_bins_warp = la.build_mel_curve(0.60, fft_len_half, amp=3.0) - 3.0
        #v_bins_warp = la.build_mel_curve(alpha, fft_len_half, amp=3.0) - 3.0
        v_bins_warp = la.build_mel_curve(alpha, fft_len_half, amp=np.pi) - np.pi

        figure('unv with smoothing versions')
        plot(np.mean(m_mag_db[~v_voi,:], axis=0))
        plot(np.mean(m_mag_syn_db[~v_voi,:], axis=0))
        plot(np.mean(m_mag_smth_db[~v_voi,:], axis=0))
        plot(np.mean(m_mag_syn_smth_db[~v_voi,:], axis=0))
        plot(np.mean(m_mag_syn_smth_db[~v_voi,:], axis=0) - np.mean(m_mag_smth_db[~v_voi,:], axis=0))
        plot(np.mean(m_mag_syn_db[~v_voi,:], axis=0) - np.mean(m_mag_db[~v_voi,:], axis=0))
        plot(-v_bins_warp)
        grid()


    if False:
        figure('unv')
        plot(np.mean(m_mag_db[~v_voi,:], axis=0))
        plot(np.mean(m_mag_syn_db[~v_voi,:], axis=0))
        plot(np.mean(np.mean(m_mag_syn_db[~v_voi,:], axis=0) - m_mag_db[~v_voi,:], axis=0))
        grid()

        figure('voi and unv diffs')
        plot(np.mean(m_mag_syn_db[v_voi,:], axis=0) - np.mean(m_mag_db[v_voi,:], axis=0))
        plot(np.mean(m_mag_syn_db[~v_voi,:], axis=0) - np.mean(m_mag_db[~v_voi,:], axis=0))
        grid()




    # SAVE WAV FILE:
    #print("Saving wav file..............................................")
    #wav_file_syn = out_dir + '/' + lu.get_filename(wav_file_orig) + '_copy_syn_low_dim_prue.wav'
    #la.write_audio_file(wav_file_syn, v_syn_sig, fs)


    print('Done!')



