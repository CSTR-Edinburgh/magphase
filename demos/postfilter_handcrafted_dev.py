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

    # CONSTANTS:==========================================================================
    d_filenames = {'hvd_593': (60,345)}


    # INPUT:==============================================================================
    filename  = 'hvd_593'
    wavs_nat_dir   = 'data_48k/wavs_nat'
    feats_pred_dir = 'data_48k/data_predict/no_pf'
    wavs_syn_dir   = 'data_48k/wavs_syn_dev_pf'

    # Analysis Natural:===================================================================
    m_mag_nat, m_real_nat, m_imag_nat, v_f0_nat, fs, v_shift_nat = mp.analysis_lossless(path.join(wavs_nat_dir, filename + '.wav'))

    # Cut silence at the beginning and at the end of natural parameters:
    nx_strt = d_filenames[filename][0]
    nx_end  = d_filenames[filename][1]
    m_mag_nat   = m_mag_nat[nx_strt:nx_end,:]
    m_real_nat  = m_real_nat[nx_strt:nx_end,:]
    m_imag_nat  = m_imag_nat[nx_strt:nx_end,:]
    v_f0_nat    = v_f0_nat[nx_strt:nx_end]
    v_shift_nat = v_shift_nat[nx_strt:nx_end]

    # Get some constants:
    fft_len_half = m_mag_nat.shape[1]
    fft_len = 2 * (fft_len_half - 1)

    # Compressing raw features:
    m_mag_nat_mel_log, m_real_nat_mel, m_imag_nat_mel, v_lf0_nat_smth = mp.format_for_modelling(m_mag_nat, m_real_nat, m_imag_nat, v_f0_nat, fs)
    m_mag_nat_smth = np.exp(la.sp_mel_unwarp(m_mag_nat_mel_log, fft_len_half, alpha=0.77, in_type='log'))

    # Synthesis Predicted:================================================================

    # Read parameters:
    m_mag_pred_mel_log = lu.read_binfile(path.join(feats_pred_dir, filename + '.mag'),  dim=60)
    m_real_pred_mel    = lu.read_binfile(path.join(feats_pred_dir, filename + '.real'), dim=45)
    m_imag_pred_mel    = lu.read_binfile(path.join(feats_pred_dir, filename + '.imag'), dim=45)
    v_lf0_pred         = lu.read_binfile(path.join(feats_pred_dir, filename + '.lf0'),  dim=1)

    m_mag_pred = np.exp(la.sp_mel_unwarp(m_mag_pred_mel_log, fft_len_half, alpha=0.77, in_type='log'))

    # Plot to check silence removal:
    if m_mag_pred_mel_log.shape!=m_mag_nat_mel_log.shape:
        raise ValueError('Shapes of natural and predicted do not match.')

    if False:
        plm(m_mag_nat_mel_log)
        plm(m_mag_pred_mel_log)

    # Postfilter:
    m_mag_nat_mel_log_pf_dummy,  m_mag_nat_mel_log_norm  = mp.post_filter_dev(m_mag_nat_mel_log, fs, av_len_at_zero=None, av_len_at_nyq=None, boost_at_zero=None, boost_at_nyq=None)
    m_mag_pred_mel_log_pf_dummy, m_mag_pred_mel_log_norm = mp.post_filter_dev(m_mag_pred_mel_log, fs, av_len_at_zero=None, av_len_at_nyq=None, boost_at_zero=None, boost_at_nyq=None)
    m_mag_nat_norm = np.exp(la.sp_mel_unwarp(m_mag_nat_mel_log_norm, fft_len_half, alpha=0.77, in_type='log'))
    m_mag_pred_norm = np.exp(la.sp_mel_unwarp(m_mag_pred_mel_log_norm, fft_len_half, alpha=0.77, in_type='log'))


    # Synthesis:
    v_syn_sig = mp.synthesis_from_compressed_type1(m_mag_pred_mel_log, m_real_pred_mel, m_imag_pred_mel, v_lf0_pred, fs)
    la.write_audio_file(path.join(wavs_syn_dir, filename + '_pred.wav'), v_syn_sig, fs)
    # Analysis from Synthesised:==========================================================

    v_f0_pred = np.exp(v_lf0_pred)
    v_shift_pred = mp.f0_to_shift(v_f0_pred, fs)
    v_pm = la.shift_to_pm(v_shift_pred)
    m_fft_syn, v_shift = mp.analysis_with_del_comp_from_pm(v_syn_sig, fs, v_pm)
    m_mag_syn = np.absolute(m_fft_syn)
    m_mag_syn_mel  = la.sp_mel_warp(m_mag_syn, 60, alpha=0.77, in_type=3)
    m_mag_syn_smth = la.sp_mel_unwarp(m_mag_syn_mel, fft_len_half, alpha=0.77, in_type='abs')

    # Energy per frame (of nat):
    v_ener = np.sum(la.add_hermitian_half(m_mag_nat)**2, axis=1) / float(fft_len)

    # PLOTS:=============================================================================
    m_mag_nat_db = la.db(m_mag_nat)
    m_mag_syn_db = la.db(m_mag_syn)
    m_mag_nat_smth_db = la.db(m_mag_nat_smth)
    m_mag_syn_smth_db = la.db(m_mag_syn_smth)

    m_mag_nat_mel_db  = la.db(np.exp(m_mag_nat_mel_log))
    m_mag_pred_mel_db = la.db(np.exp(m_mag_pred_mel_log))
    m_mag_pred_mel_db_pf = la.db(np.exp(m_mag_pred_mel_log_pf))

    m_mag_nat_mel_db_norm = la.db(np.exp(m_mag_nat_mel_log_norm))
    m_mag_pred_mel_db_norm = la.db(np.exp(m_mag_pred_mel_log_norm))

    m_mag_pred_db = la.db(m_mag_pred)
    m_mag_nat_norm_db = la.db(m_mag_nat_norm)
    m_mag_pred_norm_db = la.db(m_mag_pred_norm)



    v_voi = (v_f0_pred * v_f0_nat) > 1.0
    v_unv = (v_f0_pred < 1.0) * (v_f0_nat < 1.0)

    if False: # F0, voi and unv
        figure(); plot(v_f0_nat); plot(v_f0_pred); plot(v_f0_nat * v_f0_pred); grid()
        figure(); plot(v_f0_nat); plot(v_f0_pred); plot(v_voi); plot(v_unv); grid()

    if False: # Spectrograms fft_len_half
        plm(m_mag_nat_db)
        plm(m_mag_syn_db)
        plm(m_mag_syn_db - m_mag_nat_db)
        plm(m_mag_syn_smth_db - m_mag_nat_smth_db)

        plm(m_mag_nat_smth_db)
        plm(m_mag_syn_smth_db)
        plm(np.abs(m_mag_syn_smth_db - m_mag_nat_smth_db))

    if False: # Spectrograms mel warped (features):

        plm(m_mag_nat_mel_db)
        plm(m_mag_pred_mel_db)
        plm(m_mag_nat_mel_db - m_mag_pred_mel_db)
        plm(np.abs(m_mag_nat_mel_db - m_mag_pred_mel_db))

        plm(m_mag_nat_mel_db_norm)
        plm(m_mag_pred_mel_db_norm)
        plm(np.abs(m_mag_nat_mel_db_norm - m_mag_pred_mel_db_norm))

        surf(np.abs(m_mag_nat_mel_db_norm - m_mag_pred_mel_db_norm), aratio=0.5)
        surf(m_mag_nat_mel_db_norm - m_mag_pred_mel_db_norm, aratio=0.5)

    if False: # Spectra mel warped NORM (features):
        nx=130; figure(); plot(m_mag_nat_mel_db_norm[nx,:]); plot(m_mag_pred_mel_db_norm[nx,:]); grid()

        nxb=10; figure(); plot(m_mag_nat_mel_db[:,nxb]); plot(m_mag_pred_mel_db[:,nxb]); grid()
        nxb=10; figure(); plot(m_mag_nat_mel_db_norm[:,nxb]); plot(m_mag_pred_mel_db_norm[:,nxb]); grid()

        nxb=40; figure(); plot(m_mag_nat_mel_db[:,nxb]); plot(m_mag_pred_mel_db[:,nxb]); grid()
        nxb=40; figure(); plot(m_mag_nat_mel_db_norm[:,nxb]); plot(m_mag_pred_mel_db_norm[:,nxb]); grid()


    if False: # Spectra mel warped (features):
        nx=128; figure(); plot(m_mag_nat_mel_db[nx,:]); plot(m_mag_pred_mel_db[nx,:]); grid()
        nx=162; figure(); plot(m_mag_nat_mel_db[nx,:]); plot(m_mag_pred_mel_db[nx,:]); grid()
        nx=39; figure(); plot(m_mag_nat_mel_db[nx,:]); plot(m_mag_pred_mel_db[nx,:]); grid()
        nx=141; figure(); plot(m_mag_nat_mel_db[nx,:]); plot(m_mag_pred_mel_db[nx,:]); grid()
        nx=13; figure(); plot(m_mag_nat_mel_db[nx,:]); plot(m_mag_pred_mel_db[nx,:]); grid()
        nx=250; figure(); plot(m_mag_nat_mel_db[nx,:]); plot(m_mag_pred_mel_db[nx,:]); plot(m_mag_pred_mel_db_pf[nx,:]); grid()

    if False: # High ress from low ress:
        nx=217; figure(); plot(m_mag_nat_mel_db[nx,:]); plot(m_mag_pred_mel_db[nx,:]); grid()
        nx=217; figure(); plot(m_mag_nat_db[nx,:]); plot(m_mag_pred_db[nx,:]); grid()
        nx=217; figure(); plot(m_mag_nat_smth_db[nx,:]); plot(m_mag_pred_db[nx,:]); grid()

        nx=100; figure(); plot(m_mag_nat_smth_db[nx,:]); plot(m_mag_pred_db[nx,:]); grid()
        nx=100; figure(); plot(m_mag_nat_norm_db[nx,:]); plot(m_mag_pred_norm_db[nx,:]); grid()

        #v_exp = np.log(m_mag_nat_norm_db[nx,:]) / np.log(m_mag_pred_norm_db[nx,:])

    if False:
        figure('Voi mean mel warped')
        plot(np.mean(m_mag_nat_mel_db[v_voi,:], axis=0))
        plot(np.mean(m_mag_pred_mel_db[v_voi,:], axis=0))
        grid()


    if False:
        figure('Unv mean mel warped')
        plot(np.mean(m_mag_nat_mel_db[v_unv,:], axis=0))
        plot(np.mean(m_mag_pred_mel_db[v_unv,:], axis=0))
        grid()

    #------------------------------------------------------------------------------------------------------------------------------

    if False:
        nx=128; figure(); plot(m_mag_nat_db[nx,:]); plot(m_mag_syn_db[nx,:]); plot(m_mag_nat_smth_db[nx,:]); plot(m_mag_syn_smth_db[nx,:]); grid()
        nx=75; figure(); plot(m_mag_nat_db[nx,:]); plot(m_mag_syn_db[nx,:]); plot(m_mag_nat_smth_db[nx,:]); plot(m_mag_syn_smth_db[nx,:]); grid()
        nx=100; figure(); plot(m_mag_nat_db[nx,:]); plot(m_mag_syn_db[nx,:]); plot(m_mag_nat_smth_db[nx,:]); plot(m_mag_syn_smth_db[nx,:]); grid()





    if False:
        nx_a=74; nx_b=77; figure(); plot(m_mag_nat_smth_db[nx_a:nx_b+1,:].T); grid()
        nx_a=74; nx_b=77; figure(); plot(m_mag_syn_smth_db[nx_a:nx_b+1,:].T); grid()

        nx_a=197; nx_b=199; figure(); plot(m_mag_nat_db[nx_a:nx_b+1,:].T); grid()
        nx_a=197; nx_b=199; figure(); plot(m_mag_syn_db[nx_a:nx_b+1,:].T); grid()

    if False:
        nx_a=180; nx_b=227; surf(m_mag_nat_db[nx_a:nx_b+1,:], aratio=2.0)
        nx_a=180; nx_b=227; surf(m_mag_syn_db[nx_a:nx_b+1,:], aratio=2.0)

        # Averages:
        #figure(); plot(np.mean(m_mag_nat_db, axis=1)); plot(np.mean(m_mag_syn_db, axis=1)); grid()
        #figure(); plot(np.mean(m_mag_nat_db, axis=0)); plot(np.mean(m_mag_syn_db, axis=0)); grid()

    if False:
        # Averages per voi/unv:

        figure('voi')
        plot(np.mean(m_mag_nat_db[v_voi,:], axis=0))
        plot(np.mean(m_mag_syn_db[v_voi,:], axis=0))
        plot(np.mean(m_mag_syn_db[v_voi,:], axis=0) - np.mean(m_mag_nat_db[v_voi,:], axis=0))
        grid()

    if False:
        figure('voi with smoothing versions')
        plot(np.mean(m_mag_nat_db[v_voi,:], axis=0))
        plot(np.mean(m_mag_syn_db[v_voi,:], axis=0))
        plot(np.mean(m_mag_nat_smth_db[v_voi,:], axis=0))
        plot(np.mean(m_mag_syn_smth_db[v_voi,:], axis=0))
        grid()

    if False:
        figure('unv with smoothing versions')
        plot(np.mean(m_mag_nat_db[v_unv,:], axis=0))
        plot(np.mean(m_mag_syn_db[v_unv,:], axis=0))
        plot(np.mean(m_mag_nat_smth_db[v_unv,:], axis=0))
        plot(np.mean(m_mag_syn_smth_db[v_unv,:], axis=0))
        plot(np.mean(m_mag_syn_smth_db[v_unv,:], axis=0) - np.mean(m_mag_nat_smth_db[v_unv,:], axis=0))
        plot(np.mean(m_mag_syn_db[v_unv,:], axis=0) - np.mean(m_mag_nat_db[v_unv,:], axis=0))
        grid()


    if False:
        figure('unv')
        plot(np.mean(m_mag_nat_db[v_unv,:], axis=0))
        plot(np.mean(m_mag_syn_db[v_unv,:], axis=0))
        plot(np.mean(np.mean(m_mag_syn_db[v_unv,:], axis=0) - m_mag_nat_db[v_unv,:], axis=0))
        grid()

        figure('voi and unv diffs')
        plot(np.mean(m_mag_syn_db[v_voi,:], axis=0) - np.mean(m_mag_nat_db[v_voi,:], axis=0))
        plot(np.mean(m_mag_syn_db[v_unv,:], axis=0) - np.mean(m_mag_nat_db[v_unv,:], axis=0))
        grid()

    print('Done!')



