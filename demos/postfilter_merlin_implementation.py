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



if __name__ == '__main__':  

    # CONSTANTS:=========================================================================
    fs = 48000

    # INPUT:==============================================================================
    filename     = 'hvd_593'
    feats_dir    = './data_48k/data_predict/no_pf'
    wavs_syn_dir = './data_48k/wavs_syn_pf_merlin_dev'
    pf_type      =  'merlin' #'no' #  'magphase' # 'merlin'

    # Synthesis Predicted:================================================================

    lu.mkdir(wavs_syn_dir)

    # Read parameters:
    m_mag_mel_log = lu.read_binfile(path.join(feats_dir, filename + '.mag'),  dim=60)
    m_real_mel    = lu.read_binfile(path.join(feats_dir, filename + '.real'), dim=45)
    m_imag_mel    = lu.read_binfile(path.join(feats_dir, filename + '.imag'), dim=45)
    v_lf0         = lu.read_binfile(path.join(feats_dir, filename + '.lf0'),  dim=1)

    # SYNTHESIS:
    print("Synthesising.................................................")
    mp.synthesis_from_acoustic_modelling_dev(feats_dir, filename, wavs_syn_dir, 60, 45, fs, pf_type=pf_type)


    print('Done!')



