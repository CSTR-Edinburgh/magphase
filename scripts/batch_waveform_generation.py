#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Felipe Espic

DESCRIPTION:
This script synthesises waveforms from the MagPhase parameters predicted by Merlin:
- *.mag:  Mel-scaled Log-Mag (dim=mag_dim,   usually 60).
- *.real: Mel-scaled real    (dim=phase_dim, usually 45).
- *.imag: Mel-scaled imag    (dim=phase_dim, usually 45).
- *.lf0:  Log-F0 (dim=1).

NOTE: Actually, it can be used to synthesise waveforms from any MagPhase parameters (no Merlin required).

INSTRUCTIONS:
This demo should work out of the box. Just run it by typing: python <script name>
If you want to use this demo with real data to work with Merlin, just modify the directories, input files, accordingly.
See the main function below for details.
"""
import sys, os
curr_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.realpath(curr_dir + '/../src'))
import libutils as lu
from libplot import lp
import magphase as mp

def synthesis(in_feats_dir, filename_token, out_syn_dir, mag_dim, phase_dim, fs, pf_type):
    mp.synthesis_from_acoustic_modelling(in_feats_dir, filename_token, out_syn_dir, mag_dim, phase_dim, fs, pf_type=pf_type, b_const_rate=False)
    return

if __name__ == '__main__':  

    # CONSTANTS:
    fs = 48000

    # INPUT:==============================================================================

    files_scp     = '../demos/data_48k/file_id_predict.scp'     # List of file names (tokens). Format used by Merlin.
    in_feats_dir  = '../demos/data_48k/params_predicted'          # Input directory that contains the predicted features.
    out_syn_dir   = '../demos/data_48k/wavs_syn_from_predicted' # Where the synthesised waveform will be stored.


    mag_dim     = 60         # Number of Mel-scaled frequency bins.
    phase_dim   = 45         # Number of Mel-scaled frequency bins kept for phase features (real and imag). It must be <= mag_dim
    pf_type     = 'magphase' # "magphase": MagPhase's own postfilter (in development)
                             # "merlin":   Merlin's style postfilter.
                             # "no":       No postfilter.

    b_multiproc   = False    # If True, it synthesises using all the available cores in parallel. If False, it just uses one core (slower).


    # FILES SETUP:========================================================================
    lu.mkdir(out_syn_dir)
    l_file_tokns = lu.read_text_file2(files_scp, dtype='string', comments='#').tolist()

    # PROCESSING:=========================================================================
    if b_multiproc:
        lu.run_multithreaded(synthesis, in_feats_dir, l_file_tokns, out_syn_dir, mag_dim, phase_dim, fs, pf_type)
    else:
        for file_tokn in l_file_tokns:
            synthesis(in_feats_dir, file_tokn, out_syn_dir, mag_dim, phase_dim, fs, pf_type)


    print('Done!')



