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


def eq_ave_general(m_mag_pred_log, m_mag_pred_pf_log, filt_len=401):

    v_mag_pred_log_ave    = np.mean(m_mag_pred_log, axis=0)
    v_mag_pred_pf_log_ave = np.mean(m_mag_pred_pf_log, axis=0)
    v_mag_pred_diff_log   = v_mag_pred_log_ave - v_mag_pred_pf_log_ave

    v_mag_pred_diff_log_smth = la.smooth_by_conv(v_mag_pred_diff_log, v_win=np.hanning(filt_len))

    if False:
        figure(); plot(m_mag_pred_log_ave); plot(m_mag_pred_pf_log_ave); plot(m_mag_pred_diff_log); plot(v_mag_pred_diff_log_smth); grid()

    m_mag_pred_pf_ave_gen_log = m_mag_pred_pf_log + v_mag_pred_diff_log_smth
    m_mag_pred_pf_ave_gen_mel_log = la.sp_mel_warp(m_mag_pred_pf_ave_gen_log, 60, alpha=0.77, in_type=2)

    return m_mag_pred_pf_ave_gen_mel_log


def eq_ave_voi_unv(m_mag_pred_log, m_mag_pred_pf_log, v_voi, filt_len=401):

    m_mag_pred_eq_mel_log = np.zeros((v_voi.size, 60))

    m_mag_pred_eq_mel_log[v_voi,:] = eq_ave_general(m_mag_pred_log[v_voi,:], m_mag_pred_pf_log[v_voi,:], filt_len=filt_len)
    m_mag_pred_eq_mel_log[~v_voi,:] = eq_ave_general(m_mag_pred_log[~v_voi,:], m_mag_pred_pf_log[~v_voi,:], filt_len=filt_len)

    return m_mag_pred_eq_mel_log


def eq_per_frame(m_mag_pred_log, m_mag_pred_pf_log, filt_len=401):

    nfrms = m_mag_pred_log.shape[0]
    m_mag_pred_eq_log = np.zeros(m_mag_pred_log.shape)

    for nxf in xrange(nfrms):
        v_mag_pred_diff_log      = m_mag_pred_log[nxf,:] - m_mag_pred_pf_log[nxf,:]
        v_mag_pred_diff_log_smth = la.smooth_by_conv(v_mag_pred_diff_log, v_win=np.hanning(filt_len))
        m_mag_pred_eq_log[nxf,:] = m_mag_pred_pf_log[nxf,:] + v_mag_pred_diff_log_smth

    m_mag_pred_eq_mel_log = la.sp_mel_warp(m_mag_pred_eq_log, 60, alpha=0.77, in_type=2)

    return m_mag_pred_eq_mel_log

if __name__ == '__main__':  

    # CONSTANTS:==========================================================================
    d_filenames = {'hvd_593': (60,345)}


    # INPUT:==============================================================================

    fft_len = 4096
    filename          = 'hvd_593'
    feats_pred_dir    = 'data_48k/data_predict/no_pf'
    feats_pred_pf_dir = 'data_48k/data_predict/pf'
    wavs_syn_dir      = 'data_48k/wavs_syn_dev_pf'

    # Synthesis Predicted:================================================================
    # Get some constants:
    fft_len_half = 1 + fft_len / 2
    fs = 48000

    # Read parameters:
    m_mag_pred_mel_log = lu.read_binfile(path.join(feats_pred_dir, filename + '.mag'),  dim=60)
    m_real_pred_mel    = lu.read_binfile(path.join(feats_pred_dir, filename + '.real'), dim=45)
    m_imag_pred_mel    = lu.read_binfile(path.join(feats_pred_dir, filename + '.imag'), dim=45)
    v_lf0_pred         = lu.read_binfile(path.join(feats_pred_dir, filename + '.lf0'),  dim=1)

    m_mag_pred_pf_mel_log = lu.read_binfile(path.join(feats_pred_pf_dir, filename + '.mag'),  dim=60)

    # Unwarping:
    m_mag_pred = np.exp(la.sp_mel_unwarp(m_mag_pred_mel_log, fft_len_half, alpha=0.77, in_type='log'))
    m_mag_pred_pf = np.exp(la.sp_mel_unwarp(m_mag_pred_pf_mel_log, fft_len_half, alpha=0.77, in_type='log'))

    # Equalisation Average:================================================================
    m_mag_pred_log    = la.log(m_mag_pred)
    m_mag_pred_pf_log = la.log(m_mag_pred_pf)

    v_voi = np.exp(v_lf0_pred) > 1.0

    m_mag_pred_pf_ave_gen_mel_log = eq_ave_general(m_mag_pred_log, m_mag_pred_pf_log, filt_len=401)
    m_mag_pred_pf_ave_voi_unv_mel_log = eq_ave_voi_unv(m_mag_pred_log, m_mag_pred_pf_log, v_voi, filt_len=401)
    m_mag_pred_pf_per_frame_mel_log = eq_per_frame(m_mag_pred_log, m_mag_pred_pf_log, filt_len=801)

    # Synthesis:===========================================================================
    v_syn_sig = mp.synthesis_from_compressed_type1(m_mag_pred_mel_log, m_real_pred_mel, m_imag_pred_mel, v_lf0_pred, fs)
    la.write_audio_file(path.join(wavs_syn_dir, filename + '_pred_no_pf.wav'), v_syn_sig, fs)

    v_syn_sig = mp.synthesis_from_compressed_type1(m_mag_pred_pf_mel_log, m_real_pred_mel, m_imag_pred_mel, v_lf0_pred, fs)
    la.write_audio_file(path.join(wavs_syn_dir, filename + '_pred_pf.wav'), v_syn_sig, fs)

    v_syn_sig = mp.synthesis_from_compressed_type1(m_mag_pred_pf_ave_gen_mel_log, m_real_pred_mel, m_imag_pred_mel, v_lf0_pred, fs)
    la.write_audio_file(path.join(wavs_syn_dir, filename + '_pred_pf_eq_ave_gen.wav'), v_syn_sig, fs)

    v_syn_sig = mp.synthesis_from_compressed_type1(m_mag_pred_pf_ave_voi_unv_mel_log, m_real_pred_mel, m_imag_pred_mel, v_lf0_pred, fs)
    la.write_audio_file(path.join(wavs_syn_dir, filename + '_pred_pf_eq_voi_unv.wav'), v_syn_sig, fs)

    v_syn_sig = mp.synthesis_from_compressed_type1(m_mag_pred_pf_per_frame_mel_log, m_real_pred_mel, m_imag_pred_mel, v_lf0_pred, fs)
    la.write_audio_file(path.join(wavs_syn_dir, filename + '_pred_pf_eq_per_frame.wav'), v_syn_sig, fs)

    # Plots:===============================================================================
    m_mag_pred_db = la.db(m_mag_pred)
    m_mag_pred_pf_db = la.db(m_mag_pred_pf)



    if False:
        plm(m_mag_pred_db)
        plm(m_mag_pred_pf_db)

    if False:
        nx=100; figure(); plot(m_mag_pred_db[nx,:]); plot(m_mag_pred_pf_db[nx,:]); plot(m_mag_pred_db[nx,:] - m_mag_pred_pf_db[nx,:]); grid()
        nx=218; figure(); plot(m_mag_pred_db[nx,:]); plot(m_mag_pred_pf_db[nx,:]); plot(m_mag_pred_db[nx,:] - m_mag_pred_pf_db[nx,:]); grid()

    if False:
        figure('Unv Averages')
        plot(np.mean(m_mag_pred_db[~v_voi,:], axis=0))
        plot(np.mean(m_mag_pred_pf_db[~v_voi,:], axis=0))
        plot(np.mean(m_mag_pred_db[~v_voi,:], axis=0) - np.mean(m_mag_pred_pf_db[~v_voi,:], axis=0))
        grid()

    if False:
        figure('Voi Averages')
        plot(np.mean(m_mag_pred_db[v_voi,:], axis=0))
        plot(np.mean(m_mag_pred_pf_db[v_voi,:], axis=0))
        plot(np.mean(m_mag_pred_db[v_voi,:], axis=0) - np.mean(m_mag_pred_pf_db[v_voi,:], axis=0))
        grid()



    print('Done!')



