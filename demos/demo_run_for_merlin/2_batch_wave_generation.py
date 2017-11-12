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
from libplot import lp
import magphase as mp

def synthesis(in_feats_dir, filename_token, out_syn_dir, nbins_mel, nbins_phase, fs, b_postfilter):
    print('\nGenerating wavefile: ' + filename_token + '................................')
    mp.synthesis_from_acoustic_modelling(in_feats_dir, filename_token, out_syn_dir, nbins_mel, nbins_phase, fs, b_postfilter)
    return

if __name__ == '__main__':  

    # CONSTANTS:
    fs = 48000

    # INPUT:==============================================================================
    files_scp     = '../data_48k/file_id.scp'     # List of file names (tokens). Format used by Merlin.
    in_feats_dir  = '../data_48k/params'          # Input directory that contains the predicted features.
    out_syn_dir   = '../data_48k/wavs_syn_merlin' # Where the synthesised waveform will be stored.

    nbins_mel     = 60    # Number of Mel-scaled frequency bins.
    nbins_phase   = 45    # Number of Mel-scaled frequency bins kept for phase features (real and imag). It must be <= nbins_mel
    b_postfilter  = True  # If True, the MagPhase vocoder post-filter is applied. Note: If you want to use the one included in Merlin, disable this one.

    b_parallel    = False  # If True, it synthesises using all the available cores in parallel. If False, it just uses one core (slower).


    # FILES SETUP:========================================================================
    lu.mkdir(out_syn_dir)
    l_file_tokns = lu.read_text_file2(files_scp, dtype='string', comments='#').tolist()

    # PROCESSING:=========================================================================
    if b_parallel:
        lu.run_multithreaded(synthesis, in_feats_dir, l_file_tokns, out_syn_dir, nbins_mel, nbins_phase, fs, b_postfilter)
    else:
        for file_tokn in l_file_tokns:
            synthesis(in_feats_dir, file_tokn, out_syn_dir, nbins_mel, nbins_phase, fs, b_postfilter)


    print('Done!')



