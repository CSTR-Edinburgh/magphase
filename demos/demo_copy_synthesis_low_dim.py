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
curr_dir = os.getcwd()
sys.path.append(os.path.realpath(curr_dir + '/../src'))
import numpy as np
import libutils as lu
import libaudio as la
from libplot import lp
import magphase as mp

def analysis(wav_file, fft_len, mvf, nbins_mel=60, nbins_phase=45):
    est_file = lu.ins_pid('temp.est')
    la.reaper(wav_file, est_file)
    m_mag_mel_log, m_real_mel, m_imag_mel, v_shift, v_lf0, fs = mp.analysis_with_del_comp__ph_enc__f0_norm__from_files2(wav_file, est_file, fft_len, mvf, f0_type='lf0', mag_mel_nbins=nbins_mel, cmplx_ph_mel_nbins=nbins_phase)
    os.remove(est_file)
    return m_mag_mel_log, m_real_mel, m_imag_mel, v_lf0

def synthesis(m_mag_mel_log, m_real_mel, m_imag_mel, v_lf0, fs):
    v_syn_sig = mp.synthesis_with_del_comp_and_ph_encoding5(m_mag_mel_log, m_real_mel, m_imag_mel, v_lf0, fft_len, fs, mvf, f0_type='lf0')
    return v_syn_sig

def plots(m_mag_mel_log, m_real_mel, m_imag_mel, v_lf0):
    lp.plotm(m_mag_mel_log)
    lp.title('Log-Magnitude Spectrum')
    lp.xlabel('Time (frames)')
    lp.ylabel('Mel-scaled frequency bins')

    lp.plotm(m_real_mel)
    lp.title('"R" Feature Phase Spectrum')
    lp.xlabel('Time (frames)')
    lp.ylabel('Mel-scaled frequency bins')

    lp.plotm(m_imag_mel)
    lp.title('"I" Feature Phase Spectrum')
    lp.xlabel('Time (frames)')
    lp.ylabel('Mel-scaled frequency bins')

    lp.figure()
    lp.plot(np.exp(v_lf0)) # unlog for better visualisation
    lp.title('F0')
    lp.xlabel('Time (frames)')
    lp.ylabel('F0')
    lp.grid()
    return


if __name__ == '__main__':  
    # CONSTANTS: So far, the vocoder has been tested only with the following constants:
    fft_len = 4096
    fs      = 48000

    # INPUT:==============================================================================
    wav_file_orig = 'data/wavs_nat/hvd_593.wav' # Original natural wave file. You can choose anyone provided in the /wavs_nat directory.
    out_dir       = 'data/wavs_syn' # Where the synthesised waveform will be stored.

    b_plots       = True # True if you want to plot the extracted parameters.
    mvf           = 4500 # Maximum voiced frequency (Hz)
    nbins_mel     = 60   # Number of Mel-scaled frequency bins.
    nbins_phase   = 45   # Number of Mel-scaled frequency bins kept for phase features (real and imag). It must be <= nbins_mel

    # PROCESS:============================================================================
    lu.mkdir(out_dir)

    # ANALYSIS:
    print("Analysing.....................................................")
    m_mag_mel_log, m_real_mel, m_imag_mel, v_lf0 = analysis(wav_file_orig, fft_len, mvf, nbins_mel=nbins_mel, nbins_phase=nbins_phase)

    # MODIFICATIONS:
    # If wanted, you can do modifications to the parameters here.

    # SYNTHESIS:
    print("Synthesising.................................................")
    v_syn_sig = synthesis(m_mag_mel_log, m_real_mel, m_imag_mel, v_lf0, fs)

    # SAVE WAV FILE:
    print("Saving wav file..............................................")
    wav_file_syn = out_dir + '/' + lu.get_filename(wav_file_orig) + '_copy_syn_low_dim.wav'
    la.write_audio_file(wav_file_syn, v_syn_sig, fs)

    # PLOTS:===============================================================================
    if b_plots:
        plots(m_mag_mel_log, m_real_mel, m_imag_mel, v_lf0)
        raw_input("Press Enter to close de figs and finish...")
        lp.close('all')

    print('Done!')







