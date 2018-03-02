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
    wav_file_orig = 'data_48k/wavs_nat/hvd_594.wav'

    # PROCESS:============================================================================
    # Analysis orig:
    m_mag, m_real, m_imag, v_f0, fs, v_shift = mp.analysis_lossless(wav_file_orig)
    m_mag_mel_log, m_real_mel, m_imag_mel, v_lf0_smth = mp.format_for_modelling(m_mag, m_real, m_imag, v_f0, fs)

    # Synthesis:
    v_lf0 = la.f0_to_lf0(v_f0)
    v_syn_sig = mp.synthesis_from_compressed_type1(m_mag_mel_log, m_real_mel, m_imag_mel, v_lf0, fs)

    # Analysis from synthesised:
    v_pm = la.shift_to_pm(v_shift)
    m_fft_syn, v_shift = mp.analysis_with_del_comp_from_pm(v_syn_sig, fs, v_pm)
    m_mag_syn = np.absolute(m_fft_syn)

    # PLOTS:=============================================================================

    if True:
        m_mag_db = la.db(m_mag)
        m_mag_syn_db = la.db(m_mag_syn)
        plm(m_mag_db)
        plm(m_mag_syn_db)
        plm(m_mag_db - m_mag_syn_db)

        nx=50; figure(); plot(m_mag_db[nx,:]); plot(m_mag_syn_db[nx,:]); grid()
        nx=75; figure(); plot(m_mag_db[nx,:]); plot(m_mag_syn_db[nx,:]); grid()
        nx=106; figure(); plot(m_mag_db[nx,:]); plot(m_mag_syn_db[nx,:]); grid()
        nx=92; figure(); plot(m_mag_db[nx,:]); plot(m_mag_syn_db[nx,:]); grid()

        # Averages:
        figure(); plot(np.mean(m_mag_db, axis=1)); plot(np.mean(m_mag_syn_db, axis=1)); grid()
        figure(); plot(np.mean(m_mag_db, axis=0)); plot(np.mean(m_mag_syn_db, axis=0)); grid()

        # Averages per voi/unv:
        v_voi = v_f0 > 1.0
        figure(); plot(np.mean(m_mag_db, axis=0)); plot(np.mean(m_mag_syn_db, axis=0)); grid()






    # SAVE WAV FILE:
    #print("Saving wav file..............................................")
    #wav_file_syn = out_dir + '/' + lu.get_filename(wav_file_orig) + '_copy_syn_low_dim_prue.wav'
    #la.write_audio_file(wav_file_syn, v_syn_sig, fs)


    print('Done!')



