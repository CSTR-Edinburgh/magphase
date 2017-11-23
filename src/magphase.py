# Created on Thu Jun 30 21:04:40 2016

"""
MagPhase Vocoder library.
@author: Felipe Espic
"""

#==============================================================================
# IMPORTS
#==============================================================================
import numpy as np
import libutils as lu 
import libaudio as la
import soundfile as sf
from scipy import interpolate
from scipy import signal
import os
import warnings

#==============================================================================
# BODY
#==============================================================================

def ola(m_frm, v_pm, win_func=None):

    v_pm = v_pm.astype(int)
    nfrms, frmlen = m_frm.shape
    v_sig = np.zeros(v_pm[-1] + frmlen)

    v_shift = la.pm_to_shift(v_pm)
    v_shift = np.append(v_shift, v_shift[-1]) # repeating last value
    strt  = 0
    for i in xrange(nfrms):

        if win_func is not None:
            #g = m_frm[i,:].copy()
            v_win = la.gen_centr_win(v_shift[i], v_shift[i+1], frmlen, win_func=win_func)
            m_frm[i,:] *= v_win

        # Add frames:
        v_sig[strt:(strt+frmlen)] += m_frm[i,:]
        strt += v_shift[i+1]

    # Cut beginning:
    v_sig = v_sig[(frmlen/2 - v_pm[0]):]

    return v_sig

#------------------------------------------------------------------------------

# Todo, add a input param to control the mix:
def voi_noise_window(length):
    return np.bartlett(length)**2.5 # 2.5 optimum # max: 4
    #return np.bartlett(length)**4

#==============================================================================
# If win_func == None, no window is applied (i.e., boxcar)
# win_func: None, window function, or list of window functions.
def windowing(v_sig, v_pm, win_func=np.hanning):
    n_smpls = np.size(v_sig)
    
    # Round to int:
    v_pm = lu.round_to_int(v_pm) 
    
    # Pitch Marks Extension: 
    v_pm_plus = np.hstack((0,v_pm, (n_smpls-1)))    
    n_pm      = np.size(v_pm_plus) - 2     
    v_lens    = np.zeros(n_pm, dtype=int)
    v_shift   = np.zeros(n_pm, dtype=int)
    v_rights  = np.zeros(n_pm, dtype=int)
    l_frames  = []
    
    for f in xrange(0,n_pm):
        left_lim  = v_pm_plus[f]
        pm        = v_pm_plus[f+1]
        right_lim = v_pm_plus[f+2]        
        
        # Curr raw frame:
        v_frm   = v_sig[left_lim:(right_lim+1)]  
        
        # win lengts:
        left_len  = pm - left_lim
        right_len = right_lim - pm
        
        # Apply window:
        if isinstance(win_func, list):
            v_win = la.gen_non_symmetric_win(left_len, right_len, win_func[f])  
            v_frm = v_frm * v_win            
            
        elif callable(open): # if it is a function:
            v_win = la.gen_non_symmetric_win(left_len, right_len, win_func)  
            v_frm = v_frm * v_win
            
        elif None:
            pass               
            
        # Store:
        l_frames.append(v_frm) 
        v_lens[f]   = len(v_frm)
        v_shift[f]  = left_len
        v_rights[f] = right_len
        
    return l_frames, v_lens, v_pm_plus, v_shift, v_rights

#==============================================================================
# From (after) 'analysis_with_del_comp':
# new: returns voi/unv decision.
# new: variable length FFT
def analysis_with_del_comp_from_est_file_2(v_in_sig, est_file, fs):
    # Pitch Marks:-------------------------------------------------------------
    v_pm_sec, v_voi = la.read_reaper_est_file(est_file, check_len_smpls=len(v_in_sig), fs=fs)    
    v_pm_smpls = v_pm_sec * fs
    
    # Windowing:---------------------------------------------------------------    
    l_frms, v_lens, v_pm_plus, v_shift, v_rights = windowing(v_in_sig, v_pm_smpls)  
    
    n_frms = len(l_frms)
    l_sp   = []
    l_ph   = []
    for f in xrange(n_frms):   
                
        v_frm = l_frms[f]        
        # un-delay the signal:
        v_frm = np.hstack((v_frm[v_shift[f]:], v_frm[0:v_shift[f]])) 
        
        v_fft = np.fft.fft(v_frm)
        v_sp  = np.absolute(v_fft)
        v_ph  = np.angle(v_fft)        
            
        # Remove second (hermitian) half:
        v_sp = la.remove_hermitian_half(v_sp)
        v_ph = la.remove_hermitian_half(v_ph)
        
        # Storing:
        l_sp.append(v_sp)
        l_ph.append(v_ph)
    
    return l_sp, l_ph, v_shift, v_voi

#==============================================================================
# From (after) 'analysis_with_del_comp':
# new: returns voi/unv decision.
def analysis_with_del_comp_from_est_file(v_in_sig, est_file, fs, nFFT=None, win_func=np.hanning, b_ph_unv_zero=False, nwin_per_pitch_period=0.5):

    if nFFT is None: # If fft length is not provided, some standard values are assumed.
        if fs==48000:
            nFFT=4096
        elif fs==16000:
            nFFT=2048

    # Pitch Marks:-------------------------------------------------------------
    v_pm_sec, v_voi = la.read_reaper_est_file(est_file, check_len_smpls=len(v_in_sig), fs=fs)    
    v_pm_smpls = v_pm_sec * fs

    m_sp, m_ph, v_shift, m_frms, m_fft = analysis_with_del_comp_from_pm(v_in_sig, v_pm_smpls, nFFT, win_func=win_func, nwin_per_pitch_period=nwin_per_pitch_period)

    if b_ph_unv_zero:
        m_ph = m_ph * v_voi[:,None]

    return m_sp, m_ph, v_shift, v_voi, m_frms, m_fft



#==============================================================================
# From (after) 'analysis_with_del_comp':
# new: returns voi/unv decision.
def analysis_with_del_comp_from_pm(v_in_sig, fs, v_pm_smpls, fft_len=None, win_func=np.hanning, nwin_per_pitch_period=0.5):

    # If the FFT length is not provided, some safe values are assumed.
    # You can try decreasing the fft length if wanted.

    if fft_len is None:
        fft_len = define_fft_len(fs)

    # Generate intermediate epocs:
    v_pm_smpls_defi = v_pm_smpls
    if nwin_per_pitch_period==0.5: # original design
        pass

    elif nwin_per_pitch_period>=1.0:
        n_eps_per_pitch_per = int((nwin_per_pitch_period * 2))

        v_pm_smpls_diff = np.diff(v_pm_smpls)
        v_pm_smpls_step = v_pm_smpls_diff / float(n_eps_per_pitch_per)
        m_pm_smpls_step = np.tile(v_pm_smpls_step, (n_eps_per_pitch_per, 1))
        m_pm_smpls_step = np.multiply(m_pm_smpls_step, np.arange(n_eps_per_pitch_per)[:,None])
        m_pm_smpls_step = np.add(m_pm_smpls_step, v_pm_smpls[:-1])
        v_pm_smpls_defi = m_pm_smpls_step.flatten(order='F')
    
    # Windowing:---------------------------------------------------------------    
    l_frms, v_lens, v_pm_plus, v_shift, v_rights = windowing(v_in_sig, v_pm_smpls_defi, win_func=win_func)
    
    # FFT:---------------------------------------------------------------------
    #len_max = np.max(v_lens) # max frame length in file
    #if fft_len < len_max:
    #    warnings.warn("fft_len (%d) is shorter than the maximum detected frame length (%d). " \
    #                  + "This issue is not very critical, but if it occurs often (e.g., more than 3 times per utterance), " \
    #                  + "please increase de FFT length." % (fft_len,len_max))
    
    n_frms = len(l_frms)
    m_frms = np.zeros((n_frms, fft_len))
    
    # For paper:--------------------------------
    #m_frms_orig = np.zeros((n_frms, fft_len))
    # ------------------------------------------
    warnmess  = "fft_len (%d) is shorter than the current detected frame length (%d). "
    warnmess += "This issue is not very critical, but if it occurs often "
    warnmess += "(e.g., more than 3 times per utterance), please increase de FFT length."
    for f in xrange(n_frms):

        if v_lens[f]<=fft_len:
            m_frms[f,0:v_lens[f]] = l_frms[f]
        else:
            m_frms[f,:] = l_frms[f][:fft_len]
            warnings.warn(warnmess % (fft_len, v_lens[f]))

        # un-delay the signal:
        v_curr_frm  = m_frms[f,:]     
        
        # For paper:----------------------------
        #m_frms_orig[f,:] = v_curr_frm        
        # --------------------------------------
                   
        m_frms[f,:] = np.hstack((v_curr_frm[v_shift[f]:], v_curr_frm[0:v_shift[f]]))         

                   
    m_fft = np.fft.fft(m_frms)
    m_sp  = np.absolute(m_fft) 
    m_ph  = np.angle(m_fft) 

    # Remove redundant second half:--------------------------------------------
    m_sp  = la.remove_hermitian_half(m_sp)
    m_ph  = la.remove_hermitian_half(m_ph) 
    m_fft = la.remove_hermitian_half(m_fft)
    
    return m_fft, v_shift

#==============================================================================

def analysis_with_del_comp(v_in_sig, nFFT, fs):
    # Pitch Marks:-------------------------------------------------------------
    v_pm = la.get_pitch_marks(v_in_sig, fs)
    v_pm_smpls = v_pm * fs
    
    # Windowing:---------------------------------------------------------------    
    l_frms, v_lens, v_pm_plus, v_shift, v_rights = windowing(v_in_sig, v_pm_smpls)
    
    # FFT:---------------------------------------------------------------------
    len_max = np.max(v_lens) # max frame length in file    
    if nFFT < len_max:
        raise ValueError("nFFT (%d) is shorter than the maximum frame length (%d)" % (nFFT,len_max))
    
    n_frms = len(l_frms)
    m_frms = np.zeros((n_frms, nFFT))
    
    for f in xrange(n_frms):           
        m_frms[f,0:v_lens[f]] = l_frms[f]
        # un-delay the signal:
        v_curr_frm  = m_frms[f,:]        
        m_frms[f,:] = np.hstack((v_curr_frm[v_shift[f]:], v_curr_frm[0:v_shift[f]])) 
        
    m_fft = np.fft.fft(m_frms)
    m_sp  = np.absolute(m_fft) 
    m_ph  = np.angle(m_fft) 
    
    # Remove redundant second half:--------------------------------------------
    m_sp = la.remove_hermitian_half(m_sp)
    m_ph = la.remove_hermitian_half(m_ph) 
    
    return m_sp, m_ph, v_shift

#==============================================================================
def synthesis_with_del_comp(m_sp, m_ph, v_shift, win_func=np.hanning, win_flat_to_len=0.3):
    
    # Enforce int:
    v_shift = lu.round_to_int(v_shift)
    
    # Mirorring second half of spectrum:
    m_sp = la.add_hermitian_half(m_sp)
    m_ph = la.add_hermitian_half(m_ph, data_type='phase')    
    
    # To complex:
    m_fft = m_sp * np.exp(m_ph * 1j)

    # To time domain:
    m_frms = np.fft.ifft(m_fft).real 
    
    # OLA:---------------------------------------------------------------------
    n_frms, nFFT = np.shape(m_sp)
    #v_out_sig    = np.zeros(np.sum(v_shift[:-1]) + nFFT + 1) # despues ver como cortar! (debe estar malo este largo!)
    v_out_sig    = np.zeros(la.shift_to_pm(v_shift)[-1] + nFFT)   
    # Metodo 2:----------------------------------------------------------------
    # Flip frms:      
    m_frms = np.fft.fftshift(m_frms, axes=1)
    strt   = 0
    v_win  = np.zeros(nFFT)
    mid_frm_nx = nFFT / 2
    for f in xrange(1,n_frms): 
        # wrap frame:
        v_curr_frm  = m_frms[f-1,:]  
        
        # Window Correction:
        if win_flat_to_len < 1:
            v_win[:] = 0
            v_win_shrt = la.gen_wider_window(win_func,v_shift[f-1], v_shift[f], win_flat_to_len)        
            v_win[(mid_frm_nx-v_shift[f-1]):(mid_frm_nx+v_shift[f])] = v_win_shrt            
            rms_orig   = np.sqrt(np.mean(v_curr_frm**2))
            v_curr_frm = v_curr_frm * v_win
            rms_after_win = np.sqrt(np.mean(v_curr_frm**2))
            # Energy compensation:
            if rms_after_win > 0:
                v_curr_frm = v_curr_frm * rms_orig / rms_after_win        
        
        # Add frames:
        v_out_sig[strt:(strt+nFFT)] += v_curr_frm        
        strt += v_shift[f] 
        
    # Cut remainders (TODO!!) (only beginning done!):
    v_out_sig = v_out_sig[(nFFT/2 - v_shift[0]):]     
   
    return v_out_sig

#==============================================================================

def ph_enc(m_ph):
    m_phs = np.sin(m_ph)    
    m_phc = np.cos(m_ph)
    return m_phs, m_phc  
    

# mode = 'sign': Relies on the cosine value, and uses sine's sign to disambiguate.    
#      = 'angle': Computes the angle between phs (imag) and phc (real)  
def ph_dec(m_phs, m_phc, mode='angle'):  
    
    if mode == 'sign':    
        m_bs = np.arcsin(m_phs)
        m_bc = np.arccos(m_phc)   
        m_ph = np.sign(m_bs) * np.abs(m_bc)   
        
    elif mode == 'angle':
        m_ph = np.angle(m_phc + m_phs * 1j)
        
    return m_ph 


#==============================================================================
# From 'analysis_with_del_comp_and_ph_encoding_from_files'
# f0_type: 'f0', 'lf0'
def analysis_with_del_comp__ph_enc__f0_norm__from_files(wav_file, est_file, nFFT, mvf, f0_type='f0', b_ph_unv_zero=False, win_func=np.hanning):

    m_spmgc, m_phs_mgc, m_phc_mgc, v_shift, v_voi, fs = analysis_with_del_comp_and_ph_encoding_from_files(wav_file, est_file, nFFT, mvf, b_ph_unv_zero=b_ph_unv_zero, win_func=win_func)
    
    v_f0 = shift_to_f0(v_shift, v_voi, fs, out=f0_type)
    
    return m_spmgc, m_phs_mgc, m_phc_mgc, v_shift, v_voi, v_f0, fs  

    
#==============================================================================    
def compute_lossless_feats(m_fft, v_shift, v_voi, fs):

    m_mag  = np.absolute(m_fft) 

    # Debug:
    #m_mag[:10,:10] = 0.0

    warnings.filterwarnings('ignore', 'divide\ by\ zero')
    m_real = m_fft.real / m_mag # = p_phc
    m_imag = m_fft.imag / m_mag # = p_phs
    warnings.filterwarnings('default', 'divide\ by\ zero')

    # Protection against division by zero.
    m_real[np.abs(m_real)==np.inf] = 0
    m_imag[np.abs(m_imag)==np.inf] = 0

    v_f0   = shift_to_f0(v_shift, v_voi, fs, out='f0', b_smooth=False)

    return m_mag, m_real, m_imag, v_f0


#=======================================================================================

def analysis_with_del_comp__ph_enc__f0_norm__from_files_raw(wav_file, est_file, nFFT=None, win_func=np.hanning, nwin_per_pitch_period=0.5):
    '''
    This function does not perform any Mel warping or data compression
    b_double_win: 2 windows per 2 pitch periods.
    '''
    # Read wav file:-----------------------------------------------------------
    v_in_sig, fs = sf.read(wav_file)

    # Check for 16kHz or 48kHz sample rate. TODO: extend to work with any sample rate.
    if (fs!=48000) or (fs!=16000):
        raise ValueError('MagPhase works only at 16kHz and 48kHz sample rates, for now. The wavefile\'s sample rate is %d (Hz).\nConsider resampling the data beforehand if wanted.' % (fs))

    # Analysis:----------------------------------------------------------------
    m_sp_dummy, m_ph_dummy, v_shift, v_voi, m_frms, m_fft = analysis_with_del_comp_from_est_file(v_in_sig, est_file, fs, nFFT=nFFT, win_func=win_func, nwin_per_pitch_period=nwin_per_pitch_period)

    # Get fft-params:----------------------------------------------------------
    m_mag, m_real, m_imag = get_fft_params_from_complex_data(m_fft)

    return m_mag, m_real, m_imag, v_shift, v_voi, m_frms, fs

    
#==============================================================================   
# v2: New fft feats (mag, real, imag) in Mel-frequency scale.
#     Selection of number of coeffs.
# mvf: Maximum voiced frequency for phase encoding
# After 'analysis_with_del_comp_and_ph_encoding'    
# new: returns voi/unv decision.
# This function performs Mel Warping and vector cutting (for phase)
def analysis_with_del_comp__ph_enc__f0_norm__from_files2(wav_file, est_file, mvf, nFFT=None, f0_type='f0', win_func=np.hanning, mag_mel_nbins=60, cmplx_ph_mel_nbins=45):

    m_mag, m_real, m_imag, v_shift, v_voi, m_frms, fs  = analysis_with_del_comp__ph_enc__f0_norm__from_files_raw(wav_file, est_file, nFFT=nFFT, win_func=win_func)

    # Mel warp:----------------------------------------------------------------
    if fs==48000:
        alpha = 0.77
    elif fs==16000:
        alpha = 0.58

    m_mag_mel = la.sp_mel_warp(m_mag, mag_mel_nbins, alpha=alpha, in_type=3)
    m_mag_mel_log = np.log(m_mag_mel)

    # Phase:-------------------------------------------------------------------
    m_imag_mel = la.sp_mel_warp(m_imag, mag_mel_nbins, alpha=alpha, in_type=2)
    m_real_mel = la.sp_mel_warp(m_real, mag_mel_nbins, alpha=alpha, in_type=2)

    # Cutting phase vectors:
    m_imag_mel = m_imag_mel[:,:cmplx_ph_mel_nbins]
    m_real_mel = m_real_mel[:,:cmplx_ph_mel_nbins]
    
    m_real_mel = np.clip(m_real_mel, -1, 1)
    m_imag_mel = np.clip(m_imag_mel, -1, 1)

    # F0:----------------------------------------------------------------------
    v_f0 = shift_to_f0(v_shift, v_voi, fs, out=f0_type)
    
    return m_mag_mel_log, m_real_mel, m_imag_mel, v_shift, v_f0, fs


# mvf: Maximum voiced frequency for phase encoding=============================
# After 'analysis_with_del_comp_and_ph_encoding'    
# new: returns voi/unv decision.
def analysis_with_del_comp_and_ph_encoding_from_files(wav_file, est_file, nFFT, mvf, b_ph_unv_zero=False, win_func=np.hanning):

    # Read wav file:
    v_in_sig, fs = sf.read(wav_file)

    m_sp, m_ph, v_shift, v_voi, m_frms = analysis_with_del_comp_from_est_file(v_in_sig, est_file, nFFT, fs, b_ph_unv_zero=b_ph_unv_zero, win_func=win_func)
   
    # Phase encoding:
    m_phs, m_phc = ph_enc(m_ph)
    
    # Sp to MGC:
    m_spmgc = la.sp_to_mcep(m_sp)  

    # Ph to MGC up to MVF:        
    nFFT        = 2*(np.size(m_sp,1) - 1)
    mvf_bin     = lu.round_to_int(mvf * nFFT / np.float(fs))
    nFFThalf_ph = la.next_pow_of_two(mvf_bin) + 1    

    m_phs_shrt       = m_phs[:,:mvf_bin]    
    m_phc_shrt       = m_phc[:,:mvf_bin]
    f_interps        = interpolate.interp1d(np.arange(mvf_bin), m_phs_shrt, kind='cubic')
    f_interpc        = interpolate.interp1d(np.arange(mvf_bin), m_phc_shrt, kind='cubic')
    m_phs_shrt_intrp = f_interps(np.linspace(0,mvf_bin-1,nFFThalf_ph))    
    m_phc_shrt_intrp = f_interpc(np.linspace(0,mvf_bin-1,nFFThalf_ph))    
    m_phs_mgc        = la.sp_to_mcep(m_phs_shrt_intrp, in_type=1)    
    m_phc_mgc        = la.sp_to_mcep(m_phc_shrt_intrp, in_type=1) 
    
    return m_spmgc, m_phs_mgc, m_phc_mgc, v_shift, v_voi, fs

    
# mvf: Maximum voiced frequency for phase encoding    
def analysis_with_del_comp_and_ph_encoding(v_in_sig, nFFT, fs, mvf):

    m_sp, m_ph, v_shift = analysis_with_del_comp(v_in_sig, nFFT, fs)
    
    # Phase encoding:
    m_phs, m_phc = ph_enc(m_ph)
    
    # Sp to MGC:
    m_spmgc    = la.sp_to_mcep(m_sp)     
    
    # Ph to MGC up to MVF:    
    #mvf         = 4500    
    nFFT        = 2*(np.size(m_sp,1) - 1)
    mvf_bin     = lu.round_to_int(mvf * nFFT / np.float(fs))
    nFFThalf_ph = la.next_pow_of_two(mvf_bin) + 1    

    m_phs_shrt       = m_phs[:,:mvf_bin]    
    m_phc_shrt       = m_phc[:,:mvf_bin]
    f_interps        = interpolate.interp1d(np.arange(mvf_bin), m_phs_shrt, kind='cubic')
    f_interpc        = interpolate.interp1d(np.arange(mvf_bin), m_phc_shrt, kind='cubic')
    m_phs_shrt_intrp = f_interps(np.linspace(0,mvf_bin-1,nFFThalf_ph))    
    m_phc_shrt_intrp = f_interpc(np.linspace(0,mvf_bin-1,nFFThalf_ph))    
    m_phs_mgc        = la.sp_to_mcep(m_phs_shrt_intrp, in_type=1)    
    m_phc_mgc        = la.sp_to_mcep(m_phc_shrt_intrp, in_type=1) 
    
    return m_spmgc, m_phs_mgc, m_phc_mgc, v_shift
    

#==============================================================================
# Input: f0, instead of shifts (v_shift).
# v_voi: Used to construct the ap mask:
# if v_voi[n] > 0, frame is voiced. If v_voi[n] == 0, frame is unvoiced. 
# If v_voy=='estim', the mask is estimated from phase data.
def synthesis_with_del_comp__ph_enc__from_f0(m_spmgc, m_phs, m_phc, v_f0, nFFT, fs, mvf, ph_hf_gen, v_voi='estim'):
    
    v_shift   = f0_to_shift(v_f0, fs)    
    v_syn_sig = synthesis_with_del_comp_and_ph_encoding(m_spmgc, m_phs, m_phc, v_shift, nFFT, fs, mvf, ph_hf_gen, v_voi=v_voi)
    
    # Debug:
    #v_syn_sig = synthesis_with_del_comp_and_ph_encoding_voi_unv_separated(m_spmgc, m_phs, m_phc, v_shift, v_voi, nFFT, fs, mvf, ph_hf_gen)

    return v_syn_sig
    
#==============================================================================
# v2: Improved phase generation. 
# v3: specific window handling for aperiodic spectrum in voiced segments.
# v4: Splitted window support
# v5: Works with new fft params: mag_mel_log, real_mel, and imag_mel
# If ph_hf_gen=='rand', generates random numbers for the phase above mvf
# If ph_hf_gen=='template_mask', uses a phase template to fill the gaps given by the aperiodic mask.
# If ph_hf_gen=='rand_mask' The same as above, but it uses random numbers instead of a template.
# The aperiodic mask is computed (estimated) according to the total phase energy per frame.
# v_voi: Used to construct the ap mask:
# if v_voi[n] > 0, frame is voiced. If v_voi[n] == 0, frame is unvoiced. 
# If v_voy=='estim', the mask is estimated from phase data.
# hf_slope_coeff: 1=no slope, 2=finishing with twice the energy at highest frequency.
#def synthesis_with_del_comp_and_ph_encoding5(m_mag_mel_log, m_real_mel, m_imag_mel, v_f0, nfft, fs, mvf, f0_type='lf0', hf_slope_coeff=1.0, b_use_ap_voi=True, b_voi_ap_win=True):
def synthesis_from_compressed(m_mag_mel_log, m_real_mel, m_imag_mel, v_lf0, fs, fft_len=None, hf_slope_coeff=1.0, b_voi_ap_win=True):
    
    # Constants for spectral crossfade (in Hz):
    crsf_cf, crsf_bw = define_crossfade_params(fs)
    alpha = define_alpha(fs)
    if fft_len==None:
        fft_len = define_fft_len(fs)

    fft_len_half = fft_len / 2 + 1
    v_f0 = np.exp(v_lf0)
    nfrms, ncoeffs_mag = m_mag_mel_log.shape
    ncoeffs_comp = m_real_mel.shape[1] 

    # Magnitude mel-unwarp:----------------------------------------------------
    m_mag = np.exp(la.sp_mel_unwarp(m_mag_mel_log, fft_len_half, alpha=alpha, in_type='log'))

    # Complex mel-unwarp:------------------------------------------------------
    f_intrp_real = interpolate.interp1d(np.arange(ncoeffs_comp), m_real_mel, kind='nearest', fill_value='extrapolate')
    f_intrp_imag = interpolate.interp1d(np.arange(ncoeffs_comp), m_imag_mel, kind='nearest', fill_value='extrapolate')
    
    m_real_mel = f_intrp_real(np.arange(ncoeffs_mag))
    m_imag_mel = f_intrp_imag(np.arange(ncoeffs_mag)) 
    
    m_real = la.sp_mel_unwarp(m_real_mel, fft_len_half, alpha=alpha, in_type='log')
    m_imag = la.sp_mel_unwarp(m_imag_mel, fft_len_half, alpha=alpha, in_type='log')
    
    # Noise Gen:---------------------------------------------------------------
    v_shift = f0_to_shift(v_f0, fs, unv_frm_rate_ms=5).astype(int)
    v_pm    = la.shift_to_pm(v_shift)
    
    ns_len = v_pm[-1] + (v_pm[-1] - v_pm[-2]) 
    v_ns   = np.random.uniform(-1, 1, ns_len)     
    
    # Noise Windowing:---------------------------------------------------------
    l_ns_win_funcs = [ np.hanning ] * nfrms
    vb_voi = v_f0 > 1 # case voiced  (1 is used for safety)  
    if b_voi_ap_win:        
        for i in xrange(nfrms):
            if vb_voi[i]:         
                l_ns_win_funcs[i] = voi_noise_window

    l_frm_ns, v_lens, v_pm_plus, v_shift_dummy, v_rights = windowing(v_ns, v_pm, win_func=l_ns_win_funcs)   # Checkear!! 
    
    m_frm_ns  = la.frm_list_to_matrix(l_frm_ns, v_shift, fft_len)
    m_frm_ns  = np.fft.fftshift(m_frm_ns, axes=1)    
    m_ns_cmplx = la.remove_hermitian_half(np.fft.fft(m_frm_ns))

    # AP-Mask:-----------------------------------------------------------------   
    # Norm gain:
    m_ns_mag  = np.absolute(m_ns_cmplx)
    rms_noise = np.sqrt(np.mean(m_ns_mag**2)) # checkear!!!!
    m_ap_mask = np.ones(m_ns_mag.shape)
    m_ap_mask = m_mag * m_ap_mask / rms_noise

    m_zeros = np.zeros((nfrms, fft_len_half))
    m_ap_mask[vb_voi,:] = la.spectral_crossfade(m_zeros[vb_voi,:], m_ap_mask[vb_voi,:], crsf_cf, crsf_bw, fs, freq_scale='hz')
    
    # HF - enhancement:          
    v_slope  = np.linspace(1, hf_slope_coeff, num=fft_len_half)
    m_ap_mask[~vb_voi,:] = m_ap_mask[~vb_voi,:] * v_slope 
    
    # Det-Mask:----------------------------------------------------------------    
    m_det_mask = m_mag
    m_det_mask[~vb_voi,:] = 0
    m_det_mask[vb_voi,:]  = la.spectral_crossfade(m_det_mask[vb_voi,:], m_zeros[vb_voi,:], crsf_cf, crsf_bw, fs, freq_scale='hz')
    
    # Applying masks:----------------------------------------------------------
    m_ap_cmplx  = m_ap_mask  * m_ns_cmplx
    m_det_cmplx = m_real + m_imag * 1j

    # Protection:
    m_det_cmplx_abs = np.absolute(m_det_cmplx)
    m_det_cmplx_abs[m_det_cmplx_abs==0.0] = 1.0

    m_det_cmplx = m_det_mask * m_det_cmplx / m_det_cmplx_abs

    # bin width: bw=11.71875 Hz
    # Final synth:-------------------------------------------------------------
    m_syn_cmplx = la.add_hermitian_half(m_ap_cmplx + m_det_cmplx, data_type='complex')    
    m_syn_td    = np.fft.ifft(m_syn_cmplx).real
    m_syn_td    = np.fft.fftshift(m_syn_td,  axes=1)
    v_syn_sig   = ola(m_syn_td,  v_pm, win_func=None)
       
    # HPF:---------------------------------------------------------------------     
    fc    = 60
    order = 4
    fc_norm   = fc / (fs / 2.0)
    bc, ac    = signal.ellip(order,0.5 , 80, fc_norm, btype='highpass')
    v_syn_sig = signal.lfilter(bc, ac, v_syn_sig)

    return v_syn_sig   

#==============================================================================
# v2: Improved phase generation. 
# v3: specific window handling for aperiodic spectrum in voiced segments.
# v4: Splitted window support
# If ph_hf_gen=='rand', generates random numbers for the phase above mvf
# If ph_hf_gen=='template_mask', uses a phase template to fill the gaps given by the aperiodic mask.
# If ph_hf_gen=='rand_mask' The same as above, but it uses random numbers instead of a template.
# The aperiodic mask is computed (estimated) according to the total phase energy per frame.
# v_voi: Used to construct the ap mask:
# if v_voi[n] > 0, frame is voiced. If v_voi[n] == 0, frame is unvoiced. 
# If v_voy=='estim', the mask is estimated from phase data.
def synthesis_with_del_comp_and_ph_encoding4(m_spmgc, m_phs_mgc, m_phc_mgc, v_shift, nFFT, fs, mvf, v_voi, b_medfilt=False, win_func=None):
    
    #Protection:
    v_shift = v_shift.astype(int)
    
    # Ph and MVF:
    mvf_bin     = lu.round_to_int(mvf * nFFT / np.float(fs))
    nFFThalf_ph = la.next_pow_of_two(mvf_bin) + 1      

    # MGC to Ph up to MVF:
    m_phs_shrt_intrp_syn = la.mcep_to_sp_sptk(m_phs_mgc, 2*(nFFThalf_ph-1), out_type=0)
    m_phc_shrt_intrp_syn = la.mcep_to_sp_sptk(m_phc_mgc, 2*(nFFThalf_ph-1), out_type=0)
    f_interps_syn        = interpolate.interp1d(np.arange(nFFThalf_ph), m_phs_shrt_intrp_syn, kind='cubic')
    f_interpc_syn        = interpolate.interp1d(np.arange(nFFThalf_ph), m_phc_shrt_intrp_syn, kind='cubic')
    m_phs_shrt_syn       = f_interps_syn(np.linspace(0,nFFThalf_ph-1,mvf_bin))
    m_phc_shrt_syn       = f_interpc_syn(np.linspace(0,nFFThalf_ph-1,mvf_bin))
        
    # Deterministic Phase decoding:----------------------
    nfrms    = np.size(m_phs_shrt_syn,0)
    nFFThalf = nFFT / 2 + 1
    m_phs_shrt_syn = np.clip(m_phs_shrt_syn, -1, 1)  
    m_phc_shrt_syn = np.clip(m_phc_shrt_syn, -1, 1)  
    m_ph_deter     = ph_dec(m_phs_shrt_syn, m_phc_shrt_syn, mode='angle')
    
    # Debug:
    f = interpolate.interp1d(np.arange(mvf_bin), m_ph_deter, kind='nearest', fill_value='extrapolate')
    m_ph_deter = f(np.arange(nFFThalf))

    # TD Noise Gen:---------------------------------------    
    v_pm    = la.shift_to_pm(v_shift)
    sig_len = v_pm[-1] + (v_pm[-1] - v_pm[-2]) 
    v_noise = np.random.uniform(-1, 1, sig_len)
    
    # Extract noise magnitude and phase for unvoiced segments: (TODO: make it more efficient!)-------------------------------
    win_func_unv = np.hanning    
    if win_func is la.cos_win:
        win_func_unv = la.cos_win    
        
    l_frm_noise, v_lens, v_pm_plus, v_shift_dummy, v_rights = windowing(v_noise, v_pm, win_func=win_func_unv)    
    m_frm_noise = la.frm_list_to_matrix(l_frm_noise, v_shift, nFFT)
    m_frm_noise = np.fft.fftshift(m_frm_noise, axes=1)
    
    m_noise_sp  = la.remove_hermitian_half(np.fft.fft(m_frm_noise))
    m_noise_ph  = np.angle(m_noise_sp)    
    m_noise_mag = np.absolute(m_noise_sp)
    m_noise_mag_log = np.log(m_noise_mag)
    # Noise amp-normalisation:
    rms_noise = np.sqrt(np.mean(m_noise_mag**2))
    m_noise_mag_log = m_noise_mag_log - np.log(rms_noise)  
    
    # Extract noise magnitude and phase for voiced segments: (TODO: make it more efficient!)-------------------------------------
    l_frm_voi_noise, v_lens, v_pm_plus, v_shift_dummy, v_rights = windowing(v_noise, v_pm, win_func=voi_noise_window)    
    m_frm_voi_noise = la.frm_list_to_matrix(l_frm_voi_noise, v_shift, nFFT)
    m_frm_voi_noise = np.fft.fftshift(m_frm_voi_noise, axes=1)
    m_voi_noise_sp  = la.remove_hermitian_half(np.fft.fft(m_frm_voi_noise))
    m_voi_noise_ph  = np.angle(m_voi_noise_sp)      
    m_voi_noise_mag = np.absolute(m_voi_noise_sp)
    m_voi_noise_mag_log = np.log(m_voi_noise_mag)
    # Noise amp-normalisation:
    rms_voi_noise = np.sqrt(np.mean(m_voi_noise_mag**2))
    m_voi_noise_mag_log = m_voi_noise_mag_log - np.log(rms_voi_noise)      
    
    #------------------------------------------------------------------------------------------------------------------------------
    
    # ap mask:
    v_voi_mask =  np.clip(v_voi, 0, 1)

    # target sp from mgc:
    m_sp_targ = la.mcep_to_sp_sptk(m_spmgc, nFFT)
    
    # medfilt:
    if b_medfilt:
        m_sp_targ = signal.medfilt(m_sp_targ, kernel_size=[3,1])        

    
    # Alloc:    
    m_frm_syn = np.zeros((nfrms, nFFT))
    m_mag_syn = np.zeros((nfrms, nFFThalf)) # just for debug
    m_mag     = np.zeros((nfrms, nFFThalf)) # just for debug
   
    # Spectral crossfade constants (TODO: Improve this):
    muf = 3500 # "minimum unvoiced freq."
    bw = (mvf - muf) - 20 # values found empirically. assuming mvf > 4000
    cut_off = (mvf + muf) / 2
    v_zeros = np.zeros((1,nFFThalf))         
    
    # Iterates through frames:
    for i in xrange(nfrms):        
        

        if v_voi_mask[i] == 1: # voiced case            
            # Magnitude:----------------------------------------- 
            v_mag_log = m_voi_noise_mag_log[i,:]                                 
            v_mag_log = la.spectral_crossfade(v_zeros, v_mag_log[None,:], cut_off, bw, fs, freq_scale='hz')[0]    

            # Debug:
            v_mag_log = np.squeeze(v_zeros)
    

            # Phase:--------------------------------------------                      
            v_ph = la.spectral_crossfade(m_ph_deter[None, i,:], m_voi_noise_ph[None,i,:], cut_off, bw, fs, freq_scale='hz')[0]

            # Debug:
            
            v_ph_deters, v_ph_deterc = ph_enc(m_ph_deter[i,:])
            v_voi_noise_phs, v_voi_noise_phc = ph_enc(m_voi_noise_ph[i,:])
            
            v_phsA = la.spectral_crossfade(v_ph_deters[None,:], v_voi_noise_phs[None,:], 5000, 2000, fs, freq_scale='hz')[0]
            v_phcA = la.spectral_crossfade(v_ph_deterc[None,:], v_voi_noise_phc[None,:], 5000, 2000, fs, freq_scale='hz')[0]
            
            v_ph = ph_dec(v_phsA, v_phcA)

            
        elif v_voi_mask[i] == 0: # unvoiced case
            # Magnitude:---------------------------------------
            v_mag_log = m_noise_mag_log[i,:]       
            # Debug:
            v_mag_log = np.squeeze(v_zeros)
            
            # Phase:--------------------------------------------
            v_ph = m_noise_ph[i,:]      
            
        # To complex:
        m_mag[i,:] = np.exp(v_mag_log) # just for debug
        v_mag = np.exp(v_mag_log) * m_sp_targ[i,:]
        v_sp  = v_mag * np.exp(v_ph * 1j) 
        v_sp  = la.add_hermitian_half(v_sp[None,:], data_type='complex')        
        
        
        # Save:
        #print(i)
        m_mag_syn[i,:] = v_mag # for inspection    
        m_frm_syn[i,:] = np.fft.fftshift(np.fft.ifft(v_sp).real)     
        
    v_sig_syn = la.ola(m_frm_syn, v_pm, win_func=win_func)
     
    return v_sig_syn, m_frm_syn, m_mag_syn, m_sp_targ, m_frm_noise, m_frm_voi_noise, m_mag    

    
#==============================================================================
# v3: specific window handling for aperiodic spectrum in voiced segments.
# v2: Improved phase generation. 
# If ph_hf_gen=='rand', generates random numbers for the phase above mvf
# If ph_hf_gen=='template_mask', uses a phase template to fill the gaps given by the aperiodic mask.
# If ph_hf_gen=='rand_mask' The same as above, but it uses random numbers instead of a template.
# The aperiodic mask is computed (estimated) according to the total phase energy per frame.
# v_voi: Used to construct the ap mask:
# if v_voi[n] > 0, frame is voiced. If v_voi[n] == 0, frame is unvoiced. 
# If v_voy=='estim', the mask is estimated from phase data.
def synthesis_with_del_comp_and_ph_encoding3(m_spmgc, m_phs_mgc, m_phc_mgc, v_shift, nFFT, fs, mvf, v_voi, b_medfilt=False):
    
    #Protection:
    v_shift = v_shift.astype(int)
    
    # Ph and MVF:
    mvf_bin     = lu.round_to_int(mvf * nFFT / np.float(fs))
    nFFThalf_ph = la.next_pow_of_two(mvf_bin) + 1      

    # MGC to Ph up to MVF:
    m_phs_shrt_intrp_syn = la.mcep_to_sp(m_phs_mgc, 2*(nFFThalf_ph-1), out_type=0)
    m_phc_shrt_intrp_syn = la.mcep_to_sp(m_phc_mgc, 2*(nFFThalf_ph-1), out_type=0)
    f_interps_syn        = interpolate.interp1d(np.arange(nFFThalf_ph), m_phs_shrt_intrp_syn, kind='cubic')
    f_interpc_syn        = interpolate.interp1d(np.arange(nFFThalf_ph), m_phc_shrt_intrp_syn, kind='cubic')
    m_phs_shrt_syn       = f_interps_syn(np.linspace(0,nFFThalf_ph-1,mvf_bin))
    m_phc_shrt_syn       = f_interpc_syn(np.linspace(0,nFFThalf_ph-1,mvf_bin))
        
    # Deterministic Phase decoding:----------------------
    nfrms    = np.size(m_phs_shrt_syn,0)
    nFFThalf = nFFT / 2 + 1
    m_phs_shrt_syn = np.clip(m_phs_shrt_syn, -1, 1)  
    m_phc_shrt_syn = np.clip(m_phc_shrt_syn, -1, 1)  
    m_ph_deter     = ph_dec(m_phs_shrt_syn, m_phc_shrt_syn, mode='angle') 
    m_ph_deter     = np.hstack((m_ph_deter, np.zeros((nfrms,nFFThalf-mvf_bin))))

    # TD Noise Gen:---------------------------------------    
    v_pm    = la.shift_to_pm(v_shift)
    sig_len = v_pm[-1] + (v_pm[-1] - v_pm[-2]) 
    v_noise = np.random.uniform(-1, 1, sig_len)    
    #v_noise = np.random.normal(size=sig_len)
    
    # Extract noise magnitude and phase for unvoiced segments: (TODO: make it more efficient!)-------------------------------
    l_frm_noise, v_lens, v_pm_plus, v_shift_dummy, v_rights = windowing(v_noise, v_pm, win_func=np.hanning)    
    m_frm_noise = la.frm_list_to_matrix(l_frm_noise, v_shift, nFFT)
    m_frm_noise = np.fft.fftshift(m_frm_noise, axes=1)
    
    m_noise_sp  = la.remove_hermitian_half(np.fft.fft(m_frm_noise))
    m_noise_ph  = np.angle(m_noise_sp)    
    m_noise_mag = np.absolute(m_noise_sp)
    m_noise_mag_log = np.log(m_noise_mag)
    # Noise amp-normalisation:
    rms_noise = np.sqrt(np.mean(m_noise_mag**2))
    m_noise_mag_log = m_noise_mag_log - np.log(rms_noise)  
    
    # Extract noise magnitude and phase for voiced segments: (TODO: make it more efficient!)-------------------------------------
    l_frm_voi_noise, v_lens, v_pm_plus, v_shift_dummy, v_rights = windowing(v_noise, v_pm, win_func=voi_noise_window)    
    m_frm_voi_noise = la.frm_list_to_matrix(l_frm_voi_noise, v_shift, nFFT)
    m_frm_voi_noise = np.fft.fftshift(m_frm_voi_noise, axes=1)
    m_voi_noise_sp  = la.remove_hermitian_half(np.fft.fft(m_frm_voi_noise))
    m_voi_noise_ph  = np.angle(m_voi_noise_sp)      
    m_voi_noise_mag = np.absolute(m_voi_noise_sp)
    m_voi_noise_mag_log = np.log(m_voi_noise_mag)
    # Noise amp-normalisation:
    rms_voi_noise = np.sqrt(np.mean(m_voi_noise_mag**2))
    m_voi_noise_mag_log = m_voi_noise_mag_log - np.log(rms_voi_noise)      
    
    #------------------------------------------------------------------------------------------------------------------------------
    
    # ap mask:
    v_voi_mask =  np.clip(v_voi, 0, 1)

    # target sp from mgc:
    m_sp_targ = la.mcep_to_sp(m_spmgc, nFFT)
    
    # medfilt:
    if b_medfilt:
        m_sp_targ = signal.medfilt(m_sp_targ, kernel_size=[3,1])        

    
    # Alloc:    
    m_frm_syn = np.zeros((nfrms, nFFT))
    m_mag_syn = np.zeros((nfrms, nFFThalf)) # just for debug
    m_mag     = np.zeros((nfrms, nFFThalf)) # just for debug
   
    # Spectral crossfade constants (TODO: Improve this):
    muf = 3500 # "minimum unvoiced freq."
    bw = (mvf - muf) - 20 # values found empirically. assuming mvf > 4000
    cut_off = (mvf + muf) / 2
    v_zeros = np.zeros((1,nFFThalf))         
    
    # Iterates through frames:
    for i in xrange(nfrms):        
        

        if v_voi_mask[i] == 1: # voiced case            
            # Magnitude: 
            v_mag_log = m_voi_noise_mag_log[i,:]                                 
            v_mag_log = la.spectral_crossfade(v_zeros, v_mag_log[None,:], cut_off, bw, fs, freq_scale='hz')[0]        

            # Phase:   
            #v_ph = la.spectral_crossfade(m_ph_deter[None, i,:], m_noise_ph[None,i,:], cut_off, bw, fs, freq_scale='hz')[0]        
            v_ph = la.spectral_crossfade(m_ph_deter[None, i,:], m_voi_noise_ph[None,i,:], cut_off, bw, fs, freq_scale='hz')[0]
            
        elif v_voi_mask[i] == 0: # unvoiced case
            # Magnitude:
            v_mag_log = m_noise_mag_log[i,:]       

            # Phase:
            v_ph = m_noise_ph[i,:]      
            
        # To complex:
        m_mag[i,:] = np.exp(v_mag_log) # just for debug
        v_mag = np.exp(v_mag_log) * m_sp_targ[i,:]
        v_sp  = v_mag * np.exp(v_ph * 1j) 
        v_sp  = la.add_hermitian_half(v_sp[None,:], data_type='complex')        
        
        # Save:
        #print(i)
        m_mag_syn[i,:] = v_mag # for inspection    
        m_frm_syn[i,:] = np.fft.fftshift(np.fft.ifft(v_sp).real)     
        
    v_sig_syn = la.ola(m_frm_syn, v_pm)
     
    return v_sig_syn, m_frm_syn, m_mag_syn, m_sp_targ, m_frm_noise, m_frm_voi_noise, m_mag
 
#==============================================================================
#def synthesis_wit_del_comp_from_raw_params(m_mag, m_real, m_imag, v_f0, fs):
def synthesis_from_lossless(m_mag, m_real, m_imag, v_f0, fs):

    m_ph_cmpx = m_real + m_imag * 1j
    m_fft     = m_mag * m_ph_cmpx / np.absolute(m_ph_cmpx)
    m_fft     = la.add_hermitian_half(m_fft, data_type='complex')
    m_frm     = np.fft.ifft(m_fft).real
    m_frm     = np.fft.fftshift(m_frm,  axes=1)
    v_shift   = f0_to_shift(v_f0, fs, unv_frm_rate_ms=5)
    v_pm      = la.shift_to_pm(v_shift)

    v_syn_sig = ola(m_frm,v_pm)

    return v_syn_sig

'''
#==============================================================================
# v2: Improved phase generation. 
# If ph_hf_gen=='rand', generates random numbers for the phase above mvf
# If ph_hf_gen=='template_mask', uses a phase template to fill the gaps given by the aperiodic mask.
# If ph_hf_gen=='rand_mask' The same as above, but it uses random numbers instead of a template.
# The aperiodic mask is computed (estimated) according to the total phase energy per frame.
# v_voi: Used to construct the ap mask:
# if v_voi[n] > 0, frame is voiced. If v_voi[n] == 0, frame is unvoiced. 
# If v_voy=='estim', the mask is estimated from phase data.
def synthesis_with_del_comp_and_ph_encoding2(m_spmgc, m_phs_mgc, m_phc_mgc, v_shift, nFFT, fs, mvf, v_voi, win_func=np.hanning):
    
    # Ph and MVF:
    mvf_bin     = lu.round_to_int(mvf * nFFT / np.float(fs))
    nFFThalf_ph = la.next_pow_of_two(mvf_bin) + 1      

    # MGC to Ph up to MVF:
    m_phs_shrt_intrp_syn = la.mcep_to_sp(m_phs_mgc, 2*(nFFThalf_ph-1), out_type=0)
    m_phc_shrt_intrp_syn = la.mcep_to_sp(m_phc_mgc, 2*(nFFThalf_ph-1), out_type=0)
    f_interps_syn        = interpolate.interp1d(np.arange(nFFThalf_ph), m_phs_shrt_intrp_syn, kind='cubic')
    f_interpc_syn        = interpolate.interp1d(np.arange(nFFThalf_ph), m_phc_shrt_intrp_syn, kind='cubic')
    m_phs_shrt_syn       = f_interps_syn(np.linspace(0,nFFThalf_ph-1,mvf_bin))
    m_phc_shrt_syn       = f_interpc_syn(np.linspace(0,nFFThalf_ph-1,mvf_bin))
        
    # Deterministic Phase decoding:----------------------
    nfrms    = np.size(m_phs_shrt_syn,0)
    nFFThalf = nFFT / 2 + 1
    m_phs_shrt_syn = np.clip(m_phs_shrt_syn, -1, 1)  
    m_phc_shrt_syn = np.clip(m_phc_shrt_syn, -1, 1)  
    m_ph_deter     = ph_dec(m_phs_shrt_syn, m_phc_shrt_syn, mode='angle') 
    m_ph_deter     = np.hstack((m_ph_deter, np.zeros((nfrms,nFFThalf-mvf_bin))))
    
    # Estimating aperiodicity mask:-----------------------
    #m_ph_ap_mask = get_ap_mask_from_uv_decision(v_voi, nFFT, fs, mvf)

    # TD Noise Gen:---------------------------------------    
    v_pm    = la.shift_to_pm(v_shift)
    sig_len = v_pm[-1] + (v_pm[-1] - v_pm[-2]) 
    v_noise = np.random.uniform(-1, 1, sig_len)    
    #v_noise = np.random.normal(size=sig_len)
    
    # Extract noise magnitude and phase:    
    l_frm_noise, v_lens, v_pm_plus, v_shift_dummy, v_rights = windowing(v_noise, v_pm, win_func=win_func)    
    m_frm_noise = la.frm_list_to_matrix(l_frm_noise, v_shift, nFFT)
    m_frm_noise = np.fft.fftshift(m_frm_noise, axes=1)
    m_noise_sp  = la.remove_hermitian_half(np.fft.fft(m_frm_noise))
    m_noise_ph  = np.angle(m_noise_sp)
    
    m_noise_mag     = np.absolute(m_noise_sp)
    m_noise_mag_log = np.log(m_noise_mag)

    
    # ap mask:
    v_voi_mask =  np.clip(v_voi, 0, 1)

    # target sp from mgc:
    m_sp_targ = la.mcep_to_sp(m_spmgc, nFFT)
    
    # Debug:
    #v_voi_mask[:] = 0
    # m_noise_ph = gen_rand_phase_by_template('../database/ph_template_1.npy',nfrms, nFFThalf)

    # Minimum phase filter for ap signal:
    #m_sp_targ = np.tile(m_sp_targ[30,:], (nfrms,1))
    m_sp_comp_mph = la.sp_to_min_phase(m_sp_targ, in_type='sp')
    m_sp_ph_mph   = np.angle(m_sp_comp_mph)
    m_noise_ph    = m_noise_ph + m_sp_ph_mph
    
    
    # Alloc:    
    m_frm_syn = np.zeros((nfrms, nFFT))
    m_mag_syn = np.zeros((nfrms, nFFThalf)) # just for debug
    
    # Noise amp-normalisation:
    rms_noise = np.sqrt(np.mean(m_noise_mag**2))
    m_noise_mag_log = m_noise_mag_log - np.log(rms_noise)  
   
    # Spectral crossfade constants (TODO: Improve this):
    muf = 3500 # "minimum unvoiced freq."
    bw = (mvf - muf) - 20 # values found empirically. assuming mvf > 4000
    cut_off = (mvf + muf) / 2
    v_zeros = np.zeros((1,nFFThalf))         
    
    # Iterates through frames:
    for i in xrange(nfrms):        
        v_mag_log = m_noise_mag_log[i,:]

        if v_voi_mask[i] == 1: # voiced case            
            # Magnitude:                                    
            #v_mag_log[:mvf_bin] = 0
            v_mag_log = la.spectral_crossfade(v_zeros, v_mag_log[None,:], cut_off, bw, fs, freq_scale='hz')[0]        

            # Phase:
            #v_ph = np.hstack((m_ph_deter[i,:], m_noise_ph[i,mvf_bin:]))       
            v_ph = la.spectral_crossfade(m_ph_deter[None, i,:], m_noise_ph[None,i,:], cut_off, bw, fs, freq_scale='hz')[0]        
            
        elif v_voi_mask[i] == 0: # unvoiced case
            # Phase:
            v_ph = m_noise_ph[i,:]
      
            
        # To complex:          
        v_mag = np.exp(v_mag_log) * m_sp_targ[i,:]
        #Debug:
        #v_mag = np.exp(v_mag_log) 
        #v_mag = m_sp_targ[114,:]

        v_sp  = v_mag * np.exp(v_ph * 1j) 
        v_sp  = la.add_hermitian_half(v_sp[None,:], data_type='complex')        
        
        # Save:
        print(i)
        m_mag_syn[i,:] = v_mag # for inspection    
        m_frm_syn[i,:] = np.fft.fftshift(np.fft.ifft(v_sp).real)       
        

    v_sig_syn = la.ola(m_frm_syn, v_pm)
    # la.write_audio_file('hola.wav', v_sig, fs)
    
    return v_sig_syn, m_frm_syn, m_mag_syn, m_sp_targ, m_frm_noise
'''
#==============================================================================
    
    
#==============================================================================
# If ph_hf_gen=='rand', generates random numbers for the phase above mvf
# If ph_hf_gen=='template_mask', uses a phase template to fill the gaps given by the aperiodic mask.
# If ph_hf_gen=='rand_mask' The same as above, but it uses random numbers instead of a template.
# The aperiodic mask is computed (estimated) according to the total phase energy per frame.
# v_voi: Used to construct the ap mask:
# if v_voi[n] > 0, frame is voiced. If v_voi[n] == 0, frame is unvoiced. 
# If v_voy=='estim', the mask is estimated from phase data.
def synthesis_with_del_comp_and_ph_encoding(m_spmgc, m_phs_mgc, m_phc_mgc, v_shift, nFFT, fs, mvf, ph_hf_gen="rand", v_voi='estim', win_func=np.hanning, win_flat_to_len=0.3):
    
    # MGC to SP:
    m_sp_syn = la.mcep_to_sp(m_spmgc, nFFT)
    
    # Ph and MVF:
    mvf_bin     = lu.round_to_int(mvf * nFFT / np.float(fs))
    nFFThalf_ph = la.next_pow_of_two(mvf_bin) + 1      
    
    # MGC to Ph up to MVF:
    m_phs_shrt_intrp_syn = la.mcep_to_sp(m_phs_mgc, 2*(nFFThalf_ph-1), out_type=0)
    m_phc_shrt_intrp_syn = la.mcep_to_sp(m_phc_mgc, 2*(nFFThalf_ph-1), out_type=0)
    f_interps_syn        = interpolate.interp1d(np.arange(nFFThalf_ph), m_phs_shrt_intrp_syn, kind='cubic')
    f_interpc_syn        = interpolate.interp1d(np.arange(nFFThalf_ph), m_phc_shrt_intrp_syn, kind='cubic')
    m_phs_shrt_syn       = f_interps_syn(np.linspace(0,nFFThalf_ph-1,mvf_bin))
    m_phc_shrt_syn       = f_interpc_syn(np.linspace(0,nFFThalf_ph-1,mvf_bin))
    
    # Generate phase up to Nyquist:
    nfrms    = np.size(m_phs_shrt_syn,0)
    nFFThalf = nFFT / 2 + 1
    m_phs_shrt_syn = np.clip(m_phs_shrt_syn, -1, 1)  
    m_phc_shrt_syn = np.clip(m_phc_shrt_syn, -1, 1)   
    
    if ph_hf_gen is 'rand':   
        m_phs_syn  = np.hstack((m_phs_shrt_syn, np.random.uniform(-1, 1, size=(nfrms,nFFThalf-mvf_bin))))
        m_phc_syn  = np.hstack((m_phc_shrt_syn, np.random.uniform(-1, 1, size=(nfrms,nFFThalf-mvf_bin))))
        
        # Phase decoding:
        m_ph_syn = ph_dec(m_phs_syn, m_phc_syn) 

    elif ph_hf_gen is 'template_mask' or 'rand_mask':
        
        # Deterministic Phase decoding:----------------------
        m_ph_deter     = ph_dec(m_phs_shrt_syn, m_phc_shrt_syn, mode='angle') 
        m_ph_deter     = np.hstack((m_ph_deter, np.zeros((nfrms,nFFThalf-mvf_bin))))
        
        # Estimating aperiodicity mask:-----------------------
        if v_voi is 'estim':
            m_ph_ap_mask = estim_ap_mask_from_ph_data(m_phs_shrt_syn, nFFT, fs, mvf)
            
        elif type(v_voi) is np.ndarray: 
            
            # Debug:
            #v_voi[:] = 0
            
            m_ph_ap_mask = get_ap_mask_from_uv_decision(v_voi, nFFT, fs, mvf)
        
        # Gen aperiodic phase:--------------------------------
        if ph_hf_gen is 'template_mask':   
            m_ap_ph = gen_rand_phase_by_template('../database/ph_template_1.npy',nfrms, nFFThalf)
    
        elif ph_hf_gen is 'rand_mask':       
            m_ap_ph = np.random.uniform(-np.pi, np.pi, size=(nfrms,nFFThalf))

            
        # Mix:
        m_ph_syn = m_ap_ph * m_ph_ap_mask + m_ph_deter * (1 - m_ph_ap_mask)       

    # Final Synthesis:
    v_syn_sig = synthesis_with_del_comp(m_sp_syn, m_ph_syn, v_shift, win_func=win_func, win_flat_to_len=win_flat_to_len)       
    
    # Debug:
    #v_syn_sig = synthesis_with_del_comp_2(m_sp_syn, m_ph_syn, m_ph_ap_mask, v_shift) 
    
    return v_syn_sig

#==============================================================================

def get_ap_mask_from_uv_decision(v_voi, nFFT, fs, mvf, fade_len=40):

    # Body:-------------------------------------    
    v_ph_ap_mask =  1 - np.clip(v_voi, 0, 1)
    mvf_bin      = lu.round_to_int(mvf * nFFT / np.float(fs)) 
    m_ph_ap_mask = np.tile(v_ph_ap_mask[:,None],[1,mvf_bin])
    
    # Smoothing of the mask arounf mvf:    
    v_ramp = np.linspace(1,0,fade_len)
    m_ph_ap_mask = 1 - m_ph_ap_mask
    m_ph_ap_mask[:,-fade_len:] = m_ph_ap_mask[:,-fade_len:] * v_ramp
    m_ph_ap_mask = 1 - m_ph_ap_mask
    
    nfrms    = len(v_voi)
    nFFThalf = nFFT / 2 + 1    
    m_ph_ap_mask = np.hstack((m_ph_ap_mask, np.ones((nfrms,nFFThalf-mvf_bin))))
    
    return m_ph_ap_mask


#==============================================================================
def estim_ap_mask_from_ph_data(m_mask_ref, nFFT, fs, mvf):        
    # Constants:    
    clip_range = [-28.1 , -10.3]
    fade_len   = 40
    
    # Body:-------------------------------------------------
    v_mask_ref = la.db(np.sqrt(np.mean(m_mask_ref**2,1)))
    
    v_ph_ap_mask = -np.clip(v_mask_ref, clip_range[0], clip_range[1])        
    v_ph_ap_mask = (v_ph_ap_mask + clip_range[1]) / float(clip_range[1] - clip_range[0])  
    
    # Phase mask in 3D:
    
    mvf_bin = lu.round_to_int(mvf * nFFT / np.float(fs))     
    m_ph_ap_mask = np.tile(v_ph_ap_mask[:,None],[1,mvf_bin])
    
    # Smoothing of the mask arounf mvf:
    
    v_ramp = np.linspace(1,0,fade_len)
    m_ph_ap_mask = 1 - m_ph_ap_mask
    m_ph_ap_mask[:,-fade_len:] = m_ph_ap_mask[:,-fade_len:] * v_ramp
    m_ph_ap_mask = 1 - m_ph_ap_mask
    
    nFFThalf = nFFT / 2 + 1
    nfrms = np.size(m_mask_ref,0)
    m_ph_ap_mask = np.hstack((m_ph_ap_mask, np.ones((nfrms,nFFThalf-mvf_bin))))

    return m_ph_ap_mask


#==============================================================================
# Transform data from picth sync to constant rate in provided in ms.
def to_constant_rate(m_data, targ_shift_ms, v_shift, fs, interp_kind='linear'):
    
    v_in_cntr_nxs    = np.cumsum(v_shift)
    in_est_sig_len   = v_in_cntr_nxs[-1] + v_shift[-1] # Instead of using sig_len, it could be estimated like this    
    targ_shift_smpls = targ_shift_ms / 1000.0 * fs   
    v_targ_cntr_nxs  = np.arange(targ_shift_smpls, in_est_sig_len, targ_shift_smpls) # checkear que el codigo DNN indexe los frames asi tamnbien!
    v_targ_cntr_nxs  = v_targ_cntr_nxs.astype(int)   
        
    # Interpolation:     
    f_interp = interpolate.interp1d(v_in_cntr_nxs, m_data, axis=0, fill_value='extrapolate', kind=interp_kind)            
    m_data   = f_interp(v_targ_cntr_nxs)  
    
    return m_data
   
#==============================================================================
# v2: allows fine frame state position (adds relative position within the state as decimal number).
# shift file in samples
def frame_to_state_mapping2(shift_file, state_lab_file, fs, states_per_phone=5, b_refine=True):
    #Read files:
    v_shift = lu.read_binfile(shift_file, dim=1)
    v_pm = la.shift_to_pm(v_shift)
    m_state_times = np.loadtxt(state_lab_file, usecols=(0,1))    
    
    # to miliseconds:
    v_pm_ms = 1000 * v_pm / fs
    m_state_times_ms = m_state_times / 10000.0    
    
    # Compare:
    nfrms = len(v_pm_ms)
    v_st = np.zeros(nfrms) - 1 # init
    for f in xrange(nfrms):
        vb_greater = (v_pm_ms[f] >= m_state_times_ms[:,0])  # * (v_pm_ms[f] <  m_state_times_ms[:,1])
        state_nx   = np.where(vb_greater)[0][-1]
        v_st[f]    = np.remainder(state_nx, states_per_phone)

        # Refining:
        if b_refine:
            state_len_ms = m_state_times_ms[state_nx,1] - m_state_times_ms[state_nx,0]
            fine_pos = ( v_pm_ms[f] - m_state_times_ms[state_nx,0] ) / state_len_ms
            v_st[f] += fine_pos 
            
    # Protection against wrong ended label files:
    np.clip(v_st, 0, states_per_phone, out=v_st)      
            
    return v_st
    
#==============================================================================

def frame_to_state_mapping(shift_file, lab_file, fs, states_per_phone=5):
    #Read files:
    v_shift = lu.read_binfile(shift_file, dim=1)
    v_pm = la.shift_to_pm(v_shift)
    m_state_times = np.loadtxt(lab_file, usecols=(0,1))    
    
    # to miliseconds:
    v_pm_ms = 1000 * v_pm / fs
    m_state_times_ms = m_state_times / 10000.0    
    
    # Compare:
    nfrms = len(v_pm_ms)
    v_st = np.zeros(nfrms) - 1 # init
    for f in xrange(nfrms):
        vb_greater = (v_pm_ms[f] >= m_state_times_ms[:,0])  # * (v_pm_ms[f] <  m_state_times_ms[:,1])
        state_nx   = np.where(vb_greater)[0][-1]
        v_st[f]    = np.remainder(state_nx, states_per_phone)
    return v_st
    
#==============================================================================
def get_n_frms_per_unit(v_shifts, in_lab_state_al_file, fs, unit_type='phone', n_sts_x_ph=5):
    raise ValueError('Deprecated. Use "get_num_of_frms_per_phon_unit", instead')
    return


#==============================================================================
# in_lab_aligned_file: in HTS format
# n_lines_x_unit: e.g., number of states per phoneme. (each state in one line)
# TODO: Change name of variables. e.g, states -> lines
# v_shift in samples.
# nfrms_tolerance: Maximum number of frames of difference between shifts and lab file allowed (Some times, the end of lab files is not acurately defined).
def get_num_of_frms_per_state(v_shift, lab_state_align_file, fs, b_prevent_zeros=False, n_states_x_phone=5, nfrms_tolerance=6):

    # Read lab file:
    m_labs_state = np.loadtxt(lab_state_align_file, usecols=(0,1))
    m_labs_state_ms = m_labs_state / 10000.0

    # Epoch Indexes:
    v_ep_nxs = np.cumsum(v_shift)
    v_ep_nxs_ms = v_ep_nxs * 1000.0 / fs

    # Get number of frames per state:
    n_states        = np.size(m_labs_state_ms,axis=0)
    v_nfrms_x_state = np.zeros(n_states)

    for st in xrange(n_states):
        vb_to_right = (m_labs_state_ms[st,0] <= v_ep_nxs_ms)
        vb_to_left  = (v_ep_nxs_ms < m_labs_state_ms[st,1])
        vb_inter    = vb_to_right * vb_to_left
        v_nfrms_x_state[st] = sum(vb_inter)

    # Correct if there is only one frame of difference:
    nfrms_diff = np.size(v_shift) - np.sum(v_nfrms_x_state)
    if (nfrms_diff > 0) and (nfrms_diff <= nfrms_tolerance):
        v_nfrms_x_state[-1] += nfrms_diff

    # Checking number of frames:
    if np.sum(v_nfrms_x_state) != np.size(v_shift):
        raise ValueError('Total number of frames is different to the number of frames of the shifts.')
    m_nfrms_x_ph  = np.reshape(v_nfrms_x_state, (n_states/n_states_x_phone, n_states_x_phone))
    v_nfrms_x_ph  = np.sum(m_nfrms_x_ph, axis=1)

    # Checking that the number of frames per phoneme should be greater than 0:
    if any(v_nfrms_x_ph == 0.0):
        raise ValueError('There is some phoneme(s) that do(es) not contain any frame.')

    # Preventing zeros:
    if b_prevent_zeros:
        v_nfrms_x_state[v_nfrms_x_state==0] = 1

    return v_nfrms_x_state
    
#==============================================================================
# in_lab_aligned_file: in HTS format
# n_lines_x_unit: e.g., number of states per phoneme. (each state in one line)
# TODO: Change name of variables. e.g, states -> lines
# v_shift in samples.
# nfrms_tolerance: Maximum number of frames of difference between shifts and lab file allowed (Some times, the end of lab files is not acurately defined).
def get_num_of_frms_per_phon_unit(v_shift, in_lab_aligned_file, fs, n_lines_x_unit=5, nfrms_tolerance=1):   

    # Read lab file:
    m_labs_state = np.loadtxt(in_lab_aligned_file, usecols=(0,1))
    m_labs_state_ms = m_labs_state / 10000.0
    
    # Epoch Indexes:
    v_ep_nxs = np.cumsum(v_shift)
    v_ep_nxs_ms = v_ep_nxs * 1000.0 / fs
    
    # Get number of frames per state:
    n_states        = np.size(m_labs_state_ms,axis=0)
    v_nfrms_x_state = np.zeros(n_states)    
    
    for st in xrange(n_states):
        vb_to_right = (m_labs_state_ms[st,0] <= v_ep_nxs_ms)
        vb_to_left  = (v_ep_nxs_ms < m_labs_state_ms[st,1])        
        vb_inter    = vb_to_right * vb_to_left        
        v_nfrms_x_state[st] = sum(vb_inter) 
        
    # Correct if there is only one frame of difference:  
    nfrms_diff = np.size(v_shift) - np.sum(v_nfrms_x_state)     
    if (nfrms_diff > 0) and (nfrms_diff <= nfrms_tolerance):
        v_nfrms_x_state[-1] += nfrms_diff 
        
    # Checking number of frames:  
    if np.sum(v_nfrms_x_state) != np.size(v_shift):
        raise ValueError('Total number of frames is different to the number of frames of the shifts.')

    m_nfrms_x_ph  = np.reshape(v_nfrms_x_state, (n_states/n_lines_x_unit, n_lines_x_unit) )
    v_nfrms_x_ph  = np.sum(m_nfrms_x_ph, axis=1)
            
    # Checking that the number of frames per phoneme should be greater than 0:
    if any(v_nfrms_x_ph == 0.0):
        raise ValueError('There is some phoneme(s) that do(es) not contain any frame.') 
        
    return v_nfrms_x_ph
    
#==============================================================================
# out: 'f0' or 'lf0'
def shift_to_f0(v_shift, v_voi, fs, out='f0', b_smooth=True):
    v_f0 = v_voi * fs / v_shift.astype('float64')

    if b_smooth:
        v_f0 = v_voi * signal.medfilt(v_f0)

    if out == 'lf0':
        v_f0 = la.f0_to_lf0(v_f0)
     
    return v_f0
    
#==============================================================================
def f0_to_shift(v_f0_in, fs, unv_frm_rate_ms=5):
    v_f0 = v_f0_in.copy()
    v_f0[v_f0 == 0] = 1000.0 / unv_frm_rate_ms    
    v_shift = fs / v_f0
    
    return v_shift     
    
#============================================================================== 
def interp_from_variable_to_const_frm_rate(m_data, v_pm_smpls, const_rate_ms, fs, interp_type='linear') :
    
    dur_total_smpls  = v_pm_smpls[-1]
    const_rate_smpls = fs * const_rate_ms / 1000
    #cons_frm_rate_frm_len = 2 * frm_rate_smpls # This assummed according to the Merlin code. E.g., frame_number = int((end_time - start_time)/50000)
    v_c_rate_centrs_smpls = np.arange(const_rate_smpls, dur_total_smpls, const_rate_smpls) 
  
    # Interpolation m_spmgc:         
    f_interp = interpolate.interp1d(v_pm_smpls, m_data, axis=0, kind=interp_type)  
    m_data_const_rate = f_interp(v_c_rate_centrs_smpls) 

    return m_data_const_rate

#==============================================================================
def interp_from_const_to_variable_rate(m_data, v_frm_locs_smpls, frm_rate_ms, fs, interp_type='linear'):

    n_c_rate_frms  = np.size(m_data,0)                
    frm_rate_smpls = fs * frm_rate_ms / 1000
    
    v_c_rate_centrs_smpls = frm_rate_smpls * np.arange(1,n_c_rate_frms+1)

    f_interp     = interpolate.interp1d(v_c_rate_centrs_smpls, m_data, axis=0, kind=interp_type)            
    m_data_intrp = f_interp(v_frm_locs_smpls) 
    
    return m_data_intrp 

#==============================================================================
# NOTE: "v_frm_locs_smpls" are the locations of the target frames (centres) in the constant rate data to sample from.
# This function should be used along with the function "interp_from_const_to_variable_rate"
def get_shifts_and_frm_locs_from_const_shifts(v_shift_c_rate, frm_rate_ms, fs, interp_type='linear'):
        
    # Interpolation in reverse:
    n_c_rate_frms      = np.size(v_shift_c_rate,0)                
    frm_rate_smpls     = fs * frm_rate_ms / 1000
    
    v_c_rate_centrs_smpls = frm_rate_smpls * np.arange(1,n_c_rate_frms+1)
    f_interp   = interpolate.interp1d(v_c_rate_centrs_smpls, v_shift_c_rate, axis=0, kind=interp_type)            
    v_shift_vr = np.zeros(n_c_rate_frms * 2) # * 2 just in case, Improve these!
    v_frm_locs_smpls = np.zeros(n_c_rate_frms * 2)
    curr_pos_smpl = v_c_rate_centrs_smpls[-1]
    for i_vr in xrange(len(v_shift_vr)-1,0, -1):        
        #print(i_vr)
        v_frm_locs_smpls[i_vr] = curr_pos_smpl            
        try:                
            v_shift_vr[i_vr] = f_interp(curr_pos_smpl)
        except ValueError:
            v_frm_locs_smpls = v_frm_locs_smpls[i_vr+1:]                
            v_shift_vr  = v_shift_vr[i_vr+1:]
            break
        
        curr_pos_smpl = curr_pos_smpl - v_shift_vr[i_vr]
        
    return v_shift_vr, v_frm_locs_smpls


def post_filter_backup_old(m_mag_mel_log):

    ncoeffs = m_mag_mel_log.shape[1]
    if ncoeffs!=60:
        warnings.warn('The postfilter has been only tested with 60 dimensional mag data. If you use another dimension, the result may be suboptimal.')

    # TODO: Define and test the av_len* with dimensions other than 60:
    av_len_strt = lu.round_to_int(11.0 * ncoeffs / 60.0)
    av_len_end  = lu.round_to_int(3.0  * ncoeffs / 60.0)

    # Body:
    nfrms, nbins_mel = m_mag_mel_log.shape
    v_ave       = np.zeros(nbins_mel)
    v_nx        = np.arange(np.floor(av_len_strt/2), nbins_mel - np.floor(av_len_end/2)).astype(int)
    v_lens      = np.linspace(av_len_strt, av_len_end, v_nx.size)
    v_lens      = (2*np.ceil(v_lens/2) - 1).astype(int)

    m_mag_mel_log_enh = np.zeros(m_mag_mel_log.shape)
    for nxf in xrange(nfrms):

        v_mag_mel_log = m_mag_mel_log[nxf,:]

        # Average:
        for nxb in v_nx:
            halflen    = np.floor(v_lens[nxb-v_nx[0]]/2).astype(int)
            v_ave[nxb] = np.mean(v_mag_mel_log[(nxb-halflen):(nxb+halflen+1)])

        # Fixing boundaries:
        v_ave[:v_nx[0]]  = v_ave[v_nx[0]]
        v_ave[v_nx[-1]:] = v_ave[v_nx[-1]]

        # Substracting average:
        v_mag_mel_log_norm = v_mag_mel_log - v_ave

        # Enhance:==========================================================================
        v_tilt_fact = np.linspace(2,6,nbins_mel)
        v_mag_mel_log_enh = (v_mag_mel_log_norm * v_tilt_fact) + v_ave
        v_mag_mel_log_enh[0]  = v_mag_mel_log[0]
        v_mag_mel_log_enh[-1] = v_mag_mel_log[-1]

        # Saving:
        m_mag_mel_log_enh[nxf,:] = v_mag_mel_log_enh

    return m_mag_mel_log_enh


def post_filter(m_mag_mel_log, fs, av_len_at_zero=None, av_len_at_nyq=None, boost_at_zero=None, boost_at_nyq=None):

    nfrms, nbins_mel = m_mag_mel_log.shape
    if nbins_mel!=60:
        warnings.warn('Post-filter: It has been only tested with 60 dimensional mag data. If you use another dimension, the result may be suboptimal.')

    # Defaults in case options are not provided by the user:
    if fs==48000:
        if av_len_at_zero is None:
            av_len_at_zero = lu.round_to_int(11.0 * (nbins_mel / 60.0))

        if av_len_at_nyq is None:
            av_len_at_nyq = lu.round_to_int(3.0  * (nbins_mel / 60.0))

        if boost_at_zero is None:
            boost_at_zero = 1.8 # 2.0

        if boost_at_nyq is None:
            boost_at_nyq = 2.0 # 6.0

    elif fs==16000:
        if any(option is None for option in [av_len_at_zero, av_len_at_nyq, boost_at_zero, boost_at_nyq]):
            warnings.warn('Post-filter: The default parameters for 16kHz sample rate have not being tunned.')

        if av_len_at_zero is None:
            av_len_at_zero = lu.round_to_int(9.0 * (nbins_mel / 60.0))

        if av_len_at_nyq is None:
            av_len_at_nyq = lu.round_to_int(12.0  * (nbins_mel / 60.0))

        if boost_at_zero is None:
            boost_at_zero = 2.0 # 2.0

        if boost_at_nyq is None:
            boost_at_nyq = 1.6 # 1.6

    else: # No default values for other sample rates yet.
        if any(option is None for option in [av_len_at_zero, av_len_at_nyq, boost_at_zero, boost_at_nyq]):
            raise ValueError('Post-filter: It has only been tested with 16kHz and 48kHz sample rates.' + \
                '\nProvide your own values for the options: av_len_at_zero, av_len_at_nyq, boost_at_zero,' + \
                '\nboost_at_nyq if you use another sample rate')

    # Body:
    v_ave  = np.zeros(nbins_mel)
    v_nx   = np.arange(np.floor(av_len_at_zero/2), nbins_mel - np.floor(av_len_at_nyq/2)).astype(int)
    v_lens = np.linspace(av_len_at_zero, av_len_at_nyq, v_nx.size)
    v_lens = (2*np.ceil(v_lens/2) - 1).astype(int)

    m_mag_mel_log_enh = np.zeros(m_mag_mel_log.shape)
    for nxf in xrange(nfrms):

        v_mag_mel_log = m_mag_mel_log[nxf,:]

        # Average:
        for nxb in v_nx:
            halflen    = np.floor(v_lens[nxb-v_nx[0]]/2).astype(int)
            v_ave[nxb] = np.mean(v_mag_mel_log[(nxb-halflen):(nxb+halflen+1)])

        # Fixing boundaries:
        v_ave[:v_nx[0]]  = v_ave[v_nx[0]]
        v_ave[v_nx[-1]:] = v_ave[v_nx[-1]]

        # Substracting average:
        v_mag_mel_log_norm = v_mag_mel_log - v_ave

        # Debug:
        if False:
            from libplot import lp; lp.figure(); lp.plot(v_mag_mel_log); lp.plot(v_ave); lp.plot(v_mag_mel_log_norm); lp.grid(); lp.show()

        # Enhance:
        v_tilt_fact = np.linspace(boost_at_zero, boost_at_nyq, nbins_mel)
        v_mag_mel_log_enh = (v_mag_mel_log_norm * v_tilt_fact) + v_ave
        v_mag_mel_log_enh[0]  = v_mag_mel_log[0]
        v_mag_mel_log_enh[-1] = v_mag_mel_log[-1]

        # Saving:
        m_mag_mel_log_enh[nxf,:] = v_mag_mel_log_enh

    return m_mag_mel_log_enh


def format_for_modelling(m_mag, m_real, m_imag, v_f0, fs, nbins_mel=60, nbins_phase=45):

    # alpha:
    alpha = define_alpha(fs)

    # f0 to Smoothed lf0:
    v_voi = (v_f0>0).astype('float')
    v_f0_smth  = v_voi * signal.medfilt(v_f0)
    v_lf0_smth = la.f0_to_lf0(v_f0_smth)

    # Mag to Log-Mag-Mel (compression):
    m_mag_mel = la.sp_mel_warp(m_mag, nbins_mel, alpha=alpha, in_type=3)
    m_mag_mel_log = np.log(m_mag_mel)

    # Phase feats to Mel-phase (compression):
    m_imag_mel = la.sp_mel_warp(m_imag, nbins_mel, alpha=alpha, in_type=2)
    m_real_mel = la.sp_mel_warp(m_real, nbins_mel, alpha=alpha, in_type=2)

    # Cutting phase vectors:
    m_real_mel = m_real_mel[:,:nbins_phase]
    m_imag_mel = m_imag_mel[:,:nbins_phase]

    # Removing phase in unvoiced frames ("random" values):
    m_real_mel = m_real_mel * v_voi[:,None]
    m_imag_mel = m_imag_mel * v_voi[:,None]

    # Clipping between -1 and 1:
    m_real_mel = np.clip(m_real_mel, -1, 1)
    m_imag_mel = np.clip(m_imag_mel, -1, 1)

    return m_mag_mel_log, m_real_mel, m_imag_mel, v_lf0_smth


def write_featfile(m_data, out_dir, filename):

    filepath = os.path.join(out_dir, filename)
    lu.write_binfile(m_data, filepath)
    return

def analysis_lossless(wav_file, fft_len=None, out_dir=None):

    # Read file:
    v_sig, fs = sf.read(wav_file)

    # Epoch detection:
    est_file = lu.ins_pid('temp.est')
    la.reaper(wav_file, est_file)
    v_pm_sec, v_voi = la.read_reaper_est_file(est_file, check_len_smpls=len(v_sig), fs=fs)
    os.remove(est_file)
    v_pm_smpls = v_pm_sec * fs

    # Spectral analysis:
    m_fft, v_shift = analysis_with_del_comp_from_pm(v_sig, fs, v_pm_smpls, fft_len=fft_len)

    # Getting high-ress magphase feats:
    m_mag, m_real, m_imag, v_f0 = compute_lossless_feats(m_fft, v_shift, v_voi, fs)

    # If output directory provided, features are written to disk:
    if type(out_dir) is str:
        file_id = os.path.basename(wav_file).split(".")[0]
        write_featfile(m_mag  , out_dir, file_id + '.mag')
        write_featfile(m_real , out_dir, file_id + '.real')
        write_featfile(m_imag , out_dir, file_id + '.imag')
        write_featfile(v_f0   , out_dir, file_id + '.f0')
        write_featfile(v_shift, out_dir, file_id + '.shift')
        return

    return m_mag, m_real, m_imag, v_f0, fs, v_shift

def analysis_compressed(wav_file, fft_len=None, out_dir=None, nbins_mel=60, nbins_phase=45):

    # Analysis:
    m_mag, m_real, m_imag, v_f0, fs, v_shift = analysis_lossless(wav_file, fft_len=fft_len)

    # Formatting for Acoustic Modelling:
    m_mag_mel_log, m_real_mel, m_imag_mel, v_lf0_smth = format_for_modelling(m_mag, m_real, m_imag, v_f0, fs, nbins_mel=nbins_mel, nbins_phase=nbins_phase)
    fft_len = 2*(np.size(m_mag,1) - 1)

    # Save features:
    if type(out_dir) is str:
        file_id = os.path.basename(wav_file).split(".")[0]
        write_featfile(m_mag_mel_log, out_dir, file_id + '.mag')
        write_featfile(m_real_mel   , out_dir, file_id + '.real')
        write_featfile(m_imag_mel   , out_dir, file_id + '.imag')
        write_featfile(v_lf0_smth   , out_dir, file_id + '.lf0')
        write_featfile(v_shift      , out_dir, file_id + '.shift')
        return

    return m_mag_mel_log, m_real_mel, m_imag_mel, v_lf0_smth, v_shift, fs, fft_len


def synthesis_from_acoustic_modelling(in_feats_dir, filename_token, out_syn_dir, nbins_mel, nbins_phase, fs, fft_len=None, b_postfilter=True):

    # Reading parameter files:
    m_mag_mel_log = lu.read_binfile(in_feats_dir + '/' + filename_token + '.mag' , dim=nbins_mel)
    m_real_mel    = lu.read_binfile(in_feats_dir + '/' + filename_token + '.real', dim=nbins_phase)
    m_imag_mel    = lu.read_binfile(in_feats_dir + '/' + filename_token + '.imag', dim=nbins_phase)
    v_lf0         = lu.read_binfile(in_feats_dir + '/' + filename_token + '.lf0' , dim=1)

    if b_postfilter:
        m_mag_mel_log = post_filter(m_mag_mel_log, fs)

    # Waveform generation:
    v_syn_sig = synthesis_from_compressed(m_mag_mel_log, m_real_mel, m_imag_mel, v_lf0, fs, fft_len=fft_len)
    la.write_audio_file(out_syn_dir + '/' + filename_token + '.wav', v_syn_sig, fs)
    return

def define_alpha(fs):
    if fs==16000:
        alpha = 0.58
    elif fs==22050:
        alpha = 0.65
    elif fs==44100:
        alpha = 0.76
    elif fs==48000:
        alpha = 0.77
    else:
        raise ValueError("Sample rate %d not supported yet." % (fs))
    return alpha

def define_fft_len(fs):
    if (fs==22050) or (fs==16000):
        fft_len = 2048
    elif (fs==8000):
        fft_len = 1024
    else: # TODO: Add warning??
        fft_len = 4096
    return fft_len

def define_crossfade_params(fs):
    crsf_bw = 2000
    if fs==48000:
        crsf_cf = 5000
    elif fs==16000:
        crsf_cf = 3000
    elif fs==44100:
        crsf_cf = 4500 # TODO: test and tune this constant (for now, roughly approx.)
        warnings.warn('Constant crsf_cf not tested nor tunned to synthesise at fs=%d Hz.' % fs)
    elif fs==22050:
        crsf_cf = 3500 # TODO: test and tune this constant (for now, roughly approx.)
        warnings.warn('Constant crsf_cf not tested nor tunned to synthesise at fs=%d Hz.' % fs)
    else:
        crsf_cf = 3500
        warnings.warn('Constant crsf_cf not tested nor tunned to synthesise at fs=%d Hz.' % fs)

    return crsf_cf, crsf_bw