#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Felipe Espic

DESCRIPTION:
This script synthesises waveforms from the MagPhase parameters predicted by Merlin:
- *.mag:  Mel-scaled Log-Mag (dim=nbins_mel,   usually 60).
- *.real: Mel-scaled real    (dim=nbins_phase, usually 45).
- *.imag: Mel-scaled imag    (dim=nbins_phase, usually 45).
- *.lf0:  Log-F0 (dim=1).

NOTE: Actually, it can be used to synthesise waveforms from any MagPhase parameters (no Merlin required).

INSTRUCTIONS:
This demo should work out of the box. Just run it by typing: python <script name>
If you want to use this demo with real data to work with Merlin, just modify the directories, input files, accordingly.
See the main function below for details.
"""
import sys, os
curr_dir = os.getcwd()
sys.path.append(os.path.realpath(curr_dir + '/../../src'))
import libutils as lu
import libaudio as la
from libplot import lp
import magphase as mp

def synthesis(in_feats_dir, filename_token, out_syn_dir, nbins_mel, nbins_phase, mvf, fs, fft_len, b_postfilter):

    print('\nGenerating wavefile: ' + filename_token + '................................')

    # Reading parameter files:
    m_mag_mel_log = lu.read_binfile(in_feats_dir + '/' + filename_token + '.mag' , dim=nbins_mel)
    m_real_mel    = lu.read_binfile(in_feats_dir + '/' + filename_token + '.real', dim=nbins_phase)
    m_imag_mel    = lu.read_binfile(in_feats_dir + '/' + filename_token + '.imag', dim=nbins_phase)
    v_lf0         = lu.read_binfile(in_feats_dir + '/' + filename_token + '.lf0' , dim=1)

    if b_postfilter:
        m_mag_mel_log = mp.post_filter(m_mag_mel_log)

    # Waveform generation:
    v_syn_sig = mp.synthesis_with_del_comp_and_ph_encoding5(m_mag_mel_log, m_real_mel, m_imag_mel, v_lf0, fft_len, fs, mvf, f0_type='lf0')
    la.write_audio_file(out_syn_dir + '/' + filename_token + '.wav', v_syn_sig, fs)
    return


if __name__ == '__main__':  

    # CONSTANTS: So far, the vocoder has been tested only with the following constants:
    fft_len = 4096
    fs      = 48000

    # INPUT:==============================================================================
    files_scp     = '../data/file_id.scp'     # List of file names (tokens). Format used by Merlin.
    in_feats_dir  = '../data/params'          # Input directory that contains the predicted features.
    out_syn_dir   = '../data/wavs_syn_merlin' # Where the synthesised waveform will be stored.

    nbins_mel     = 60    # Number of Mel-scaled frequency bins.
    nbins_phase   = 45    # Number of Mel-scaled frequency bins kept for phase features (real and imag). It must be <= nbins_mel
    mvf           = 4500  # Maximum voiced frequency (Hz)
    b_parallel    = True  # If True, it synthesises using all the available cores in parallel. If False, it just uses one core (slower).
    b_postfilter  = True  # If True, the MagPhase vocoder post-filter is applied. Note: If you want to use the one included in Merlin, disable this one.

    # FILES SETUP:========================================================================
    lu.mkdir(out_syn_dir)
    l_file_tokns = lu.read_text_file2(files_scp, dtype='string', comments='#').tolist()

    # PROCESSING:=========================================================================
    if b_parallel:
        lu.run_multithreaded(synthesis, in_feats_dir, l_file_tokns, out_syn_dir, nbins_mel, nbins_phase, mvf, fs, fft_len, b_postfilter)
    else:
        for nx in xrange(len(l_file_tokns)):
            synthesis(in_feats_dir, l_file_tokns[nx], out_syn_dir, nbins_mel, nbins_phase, mvf, fs, fft_len, b_postfilter)


    print('Done!')







