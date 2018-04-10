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

def plots(m_mag_mel_log, m_real_mel, m_imag_mel, v_lf0):
    lp.plotm(m_mag_mel_log)
    lp.title(' Mel-scaled Log-Magnitude Spectrum')
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

    # CONSTANTS:==========================================================================
    out_dir = 'data_48k/wavs_syn_mel_comp_2d_dev' # Where the synthesised waveform will be stored.
    b_new_phase_comp = True
    const_rate_ms = -1 # 5 # -1

    # INPUT:==============================================================================
    wav_file_orig = 'data_48k/wavs_nat/hvd_593.wav' # Original natural wave file. You can choose anyone provided in the /wavs_nat directory.
    nbins_phase   = 15
    b_mag_fbank_mel = False

    # PROCESS:============================================================================
    lu.mkdir(out_dir)

    if b_new_phase_comp:
        print("Analysing.....................................................")
        m_mag_mel_log, m_real_mel, m_imag_mel, v_lf0, v_shift, fs, fft_len = mp.analysis_compressed_type1_with_phase_comp(wav_file_orig,
                                                                nbins_phase=nbins_phase, const_rate_ms=const_rate_ms, b_mag_fbank_mel=b_mag_fbank_mel)


        print("Synthesising.................................................")
        v_syn_sig = mp.synthesis_from_compressed_type1_with_phase_comp(m_mag_mel_log, m_real_mel, m_imag_mel, v_lf0, fs,
                                                                        b_fbank_mel=b_mag_fbank_mel, const_rate_ms=const_rate_ms)


        print("Saving wav file..............................................")
        #wav_file_syn = out_dir + '/' + lu.get_filename(wav_file_orig) + '_copy_syn_low_dim_ph_15_mel_comp_sptk_normal.wav'
        wav_file_syn = out_dir + '/' + lu.get_filename(wav_file_orig) + '_copy_syn_low_dim_ph_15_mel_comp_fbank_prue.wav'

    else:
        print("Analysing.....................................................")
        m_mag_mel_log, m_real_mel, m_imag_mel, v_lf0, v_shift, fs, fft_len = mp.analysis_compressed_type1(wav_file_orig)

        print("Synthesising.................................................")
        v_syn_sig = mp.synthesis_from_compressed_type1(m_mag_mel_log, m_real_mel, m_imag_mel, v_lf0, fs)


        print("Saving wav file..............................................")
        wav_file_syn = out_dir + '/' + lu.get_filename(wav_file_orig) + '_copy_syn_low_dim_ph_comp_baseline_voi.wav'



    la.write_audio_file(wav_file_syn, v_syn_sig, fs)

    print('Done!')



