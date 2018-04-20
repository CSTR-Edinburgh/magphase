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
from subprocess import call

#==============================================================================
# BODY
#==============================================================================

def raised_hanning(length, att=1.0):
    '''
    att: Attenuation [0,1]
    '''
    v_win = att * (np.hanning(length))
    v_win = (1 - att) + v_win
    return v_win


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

            if False:
                from libplot import lp
                lp.figure(); lp.plot(v_win); lp.grid()

        # Add frames:
        v_sig[strt:(strt+frmlen)] += m_frm[i,:]
        strt += v_shift[i+1]

    # Cut ending and beginning:
    v_sig = v_sig[(frmlen/2 - v_pm[0]):]
    v_sig = v_sig[:(v_pm[-1] + v_shift[-1] + 1)]

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
def analysis_with_del_comp_from_pm_type2(v_in_sig, fs, v_pm_smpls, v_voi, fft_len=None,
                                         win_func=np.hanning, nwin_per_pitch_period=0.5):

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
    v_gain = np.zeros(n_frms)
    fft_len_half = fft_len / 2 + 1
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

        # Gain:
        if v_voi[f]==1: # voiced case:
            v_gain[f] = np.max(np.abs(m_frms[f,:fft_len_half]))
        else: # unv case:
            #v_gain[f] = np.sqrt(np.mean(l_frms[f]**2)) # TODO: Improve (remove dc)
            v_gain[f] = np.std(l_frms[f])

    m_fft = np.fft.fft(m_frms)
    m_sp  = np.absolute(m_fft)
    m_ph  = np.angle(m_fft)

    # Remove redundant second half:--------------------------------------------
    m_sp  = la.remove_hermitian_half(m_sp)
    m_ph  = la.remove_hermitian_half(m_ph)
    m_fft = la.remove_hermitian_half(m_fft)

    return m_fft, v_shift, v_gain


#==============================================================================
# From (after) 'analysis_with_del_comp':
# new: returns voi/unv decision.
def analysis_with_del_comp_from_pm(v_in_sig, fs, v_pm_smpls, fft_len=None,
                                    win_func=np.hanning, nwin_per_pitch_period=0.5):

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
    m_real[np.abs(m_real)==np.inf] = 0.0
    m_imag[np.abs(m_imag)==np.inf] = 0.0
    m_real[np.isnan(m_real)] = 0.0
    m_imag[np.isnan(m_imag)] = 0.0

    v_f0 = shift_to_f0(v_shift, v_voi, fs, out='f0', b_smooth=False)

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
def synthesis_from_compressed_type1_old_with_griffin_lim(m_mag_mel_log, m_real_mel, m_imag_mel, v_lf0, fs, fft_len=None,
                                        hf_slope_coeff=1.0, b_voi_ap_win=True, b_fbank_mel=False, const_rate_ms=-1.0,
                                                per_phase_type='magphase', griff_lim_type=None, griff_lim_init='magphase'):

    '''
    b_fbank_mel: If True, Mel compression done by the filter bank approach. Otherwise, it uses sptk mcep related funcs.
    per_phase_type: 'magphase', 'min_phase', or 'linear'
    griff_lim_type: None, 'whole' , 'det', 'whole' (None=Griffin-Lim disabled)
    griff_lim_init: 'magphase', 'linear', 'min_phase', 'random'
    '''
    



    # Constants for spectral crossfade (in Hz):
    crsf_cf, crsf_bw = define_crossfade_params(fs)
    alpha = define_alpha(fs)
    if fft_len==None:
        fft_len = define_fft_len(fs)

    fft_len_half = fft_len / 2 + 1
    v_f0 = np.exp(v_lf0)
    nfrms, ncoeffs_mag = m_mag_mel_log.shape


    # Magnitude mel-unwarp:----------------------------------------------------
    if b_fbank_mel:
        m_mag = np.exp(la.sp_mel_unwarp_fbank(m_mag_mel_log, fft_len_half, alpha=alpha))
    else:
        m_mag = np.exp(la.sp_mel_unwarp(m_mag_mel_log, fft_len_half, alpha=alpha, in_type='log'))

    # Complex mel-unwarp:------------------------------------------------------
    ncoeffs_comp = m_real_mel.shape[1]
    f_intrp_real = interpolate.interp1d(np.arange(ncoeffs_comp), m_real_mel, kind='nearest', fill_value='extrapolate')
    f_intrp_imag = interpolate.interp1d(np.arange(ncoeffs_comp), m_imag_mel, kind='nearest', fill_value='extrapolate')
    
    m_real_mel = f_intrp_real(np.arange(ncoeffs_mag))
    m_imag_mel = f_intrp_imag(np.arange(ncoeffs_mag)) 
    
    m_real = la.sp_mel_unwarp(m_real_mel, fft_len_half, alpha=alpha, in_type='log')
    m_imag = la.sp_mel_unwarp(m_imag_mel, fft_len_half, alpha=alpha, in_type='log')


    # Constant to variable frame rate:-----------------------------------------
    v_shift = f0_to_shift(v_f0, fs)
    if const_rate_ms>0.0:
        interp_type = 'linear' #'quadratic' , 'cubic'
        v_shift, v_frm_locs_smpls = get_shifts_and_frm_locs_from_const_shifts(v_shift, const_rate_ms, fs, interp_type=interp_type)
        m_mag  = interp_from_const_to_variable_rate(m_mag,    v_frm_locs_smpls, const_rate_ms, fs, interp_type=interp_type)
        m_real = interp_from_const_to_variable_rate(m_real,   v_frm_locs_smpls, const_rate_ms, fs, interp_type=interp_type)
        m_imag = interp_from_const_to_variable_rate(m_imag,   v_frm_locs_smpls, const_rate_ms, fs, interp_type=interp_type)
        v_voi  = interp_from_const_to_variable_rate(v_f0>0.0, v_frm_locs_smpls, const_rate_ms, fs, interp_type=interp_type) > 0.5
        v_f0   = shift_to_f0(v_shift, v_voi, fs, out='f0', b_smooth=False)
        nfrms  = v_shift.size
    
    # Noise Gen:---------------------------------------------------------------
    v_shift = v_shift.astype(int)
    v_pm    = la.shift_to_pm(v_shift)
    
    ns_len = v_pm[-1] + (v_pm[-1] - v_pm[-2]) 
    v_ns   = np.random.uniform(-1, 1, ns_len)     

    # Noise Windowing:---------------------------------------------------------
    l_ns_win_funcs = [ np.hanning ] * nfrms
    v_voi = v_f0 > 1 # case voiced  (1 is used for safety)
    if b_voi_ap_win:        
        for i in xrange(nfrms):
            if v_voi[i]:
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
    m_ap_mask[v_voi,:] = la.spectral_crossfade(m_zeros[v_voi,:], m_ap_mask[v_voi,:], crsf_cf, crsf_bw, fs, freq_scale='hz')
    
    # HF - enhancement:          
    v_slope  = np.linspace(1, hf_slope_coeff, num=fft_len_half)
    m_ap_mask[~v_voi,:] = m_ap_mask[~v_voi,:] * v_slope

    m_ap_cmplx_spec = m_ap_mask  * m_ns_cmplx
    m_ap_cmplx_spec[m_ap_mask==0.0] = 0 + 0j # protection

    # Det-Mask:----------------------------------------------------------------
    m_det_mask = m_mag.copy()
    m_det_mask[~v_voi,:] = 0
    m_det_mask[v_voi,:]  = la.spectral_crossfade(m_det_mask[v_voi,:], m_zeros[v_voi,:], crsf_cf, crsf_bw, fs, freq_scale='hz')
    
    # Deterministic Part:----------------------------------------------------------
    if per_phase_type=='magphase':
        m_det_cmplx_ph  = m_real + m_imag * 1j

        # Protection:
        m_det_cmplx_ph_mag = np.absolute(m_det_cmplx_ph)
        m_det_cmplx_ph_mag[m_det_cmplx_ph_mag==0.0] = 1.0
        m_det_cmplx_ph = m_det_cmplx_ph / m_det_cmplx_ph_mag

        m_det_cmplx_spec = m_det_mask * m_det_cmplx_ph

    if per_phase_type=='linear':
        m_det_cmplx_spec = m_det_mask

    elif per_phase_type=='min_phase':
        m_det_cmplx_spec  = la.build_min_phase_from_mag_spec(m_mag)

    # Protection:
    m_det_cmplx_spec[m_det_mask==0.0] = 0 + 0j

    # Griffin-Lim (only deterministic part):-----------------------------------
    if griff_lim_type=='det':

        # Un-delay:
        m_det_cmplx_spec = la.add_hermitian_half(m_det_cmplx_spec, data_type='complex')
        m_frms_gl = np.fft.ifft(m_det_cmplx_spec).real
        m_frms_gl = np.fft.fftshift(m_frms_gl, axes=1)
        m_det_cmplx_spec = la.remove_hermitian_half(np.fft.fft(m_frms_gl))
        m_det_cmplx_spec[m_det_mask==0.0] = 0 + 0j

        # GLA:
        m_phase_gl_init = np.angle(m_det_cmplx_spec)
        m_mag_gl        = np.absolute(m_det_cmplx_spec)
        v_syn_sig, m_phase_gl = griffin_lim(m_mag_gl, v_shift, phase_init=m_phase_gl_init, niters=10)
        m_det_cmplx_spec = m_mag_gl * np.exp(m_phase_gl * 1j)

        # Re-delay:
        m_det_cmplx_spec = la.add_hermitian_half(m_det_cmplx_spec, data_type='complex')
        m_frms_gl = np.fft.ifft(m_det_cmplx_spec).real
        m_frms_gl = np.fft.fftshift(m_frms_gl, axes=1)
        m_det_cmplx_spec = la.remove_hermitian_half(np.fft.fft(m_frms_gl))
        m_det_cmplx_spec[m_det_mask==0.0] = 0 + 0j

    # Final synth:---------------------------------------------------------------
    # bin width: bw=11.71875 Hz
    m_syn_cmplx = la.add_hermitian_half(m_det_cmplx_spec + m_ap_cmplx_spec, data_type='complex')
    m_syn_td    = np.fft.ifft(m_syn_cmplx).real
    m_syn_td    = np.fft.fftshift(m_syn_td, axes=1)
    v_syn_sig   = ola(m_syn_td, v_pm, win_func=None)

    # Griffin-Lim (whole):------------------------------------------------------------
    #if griff_lim_type is not None:
    if griff_lim_type=='whole':
        m_fft_gl   = la.remove_hermitian_half(np.fft.fft(m_syn_td))
        m_phase_gl_init = np.angle(m_fft_gl)
        m_mag_gl   = np.absolute(m_fft_gl)
        v_syn_sig, m_phase_gl = griffin_lim(m_mag_gl, v_shift, phase_init='min', niters=50)
        #v_syn_sig_gl, m_phase_gl = griffin_lim(m_mag_gl, v_shift, phase_init='min', niters=50)
        #v_syn_sig  = griffin_lim(m_mag_gl, v_shift, phase_init=m_phase_gl_init, niters=50)
        '''
        if griff_lim_type=='whole':
            v_syn_sig = v_syn_sig_gl

        elif griff_lim_type=='det':
        '''



    # griff_lim_type: None, 'whole' , 'det'(None=Griffin-Lim disabled)
    # griff_lim_init: 'magphase', 'linear', 'min_phase', 'random'


    # HPF:---------------------------------------------------------------------     
    fc    = 60
    order = 4
    fc_norm   = fc / (fs / 2.0)
    bc, ac    = signal.ellip(order,0.5 , 80, fc_norm, btype='highpass')
    v_syn_sig = signal.lfilter(bc, ac, v_syn_sig)

    return v_syn_sig   


#=============================================================================
def phase_uncompress_fbank(m_real_mel, m_imag_mel, crsf_cf, crsf_bw, alpha, fft_len, fs):

    bin_cf = lu.round_to_int(la.hz_to_bin(crsf_cf, fft_len, fs))

    max_bin_ph = bin_cf # bin_l # bin_cf # bin_r # bin_l

    fft_len_half = 1 + fft_len/2
    v_bins_mel   = la.build_mel_curve(alpha, fft_len_half)[:max_bin_ph]

    m_real_shrt = la.unwarp_from_fbank(m_real_mel, v_bins_mel, interp_kind='quadratic')
    m_imag_shrt = la.unwarp_from_fbank(m_imag_mel, v_bins_mel, interp_kind='quadratic')

    #m_real_shrt = la.unwarp_from_fbank(m_real_mel, v_bins_mel, interp_kind='cubic')
    #m_imag_shrt = la.unwarp_from_fbank(m_imag_mel, v_bins_mel, interp_kind='cubic')

    #m_real_shrt = la.unwarp_from_fbank(m_real_mel, v_bins_mel, interp_kind='slinear')
    #m_imag_shrt = la.unwarp_from_fbank(m_imag_mel, v_bins_mel, interp_kind='slinear')

    nfrms  = m_real_mel.shape[0]
    m_real = np.hstack((m_real_shrt, m_real_shrt[:,-1][:,None] + np.zeros((nfrms, fft_len_half-max_bin_ph))))
    m_imag = np.hstack((m_imag_shrt, m_imag_shrt[:,-1][:,None] + np.zeros((nfrms, fft_len_half-max_bin_ph))))

    return m_real, m_imag



#==============================================================================
def synthesis_from_compressed(m_mag_mel_log, m_real_mel, m_imag_mel, v_lf0, fs, fft_len=None, b_voi_ap_win=True,
                                    b_fbank_mel=False, b_const_rate=False, per_phase_type='magphase', alpha_phase=None, b_out_hpf=True):

    '''
    synthesis_from_compressed_type1 with phase compression based on filter bank. It didn't work very well according to experiments.

    b_fbank_mel: If True, Mel compression done by the filter bank approach. Otherwise, it uses sptk mcep related funcs.
    per_phase_type: 'magphase', 'min_phase', or 'linear'
    '''

    # Setting up constants:====================================================
    crsf_cf, crsf_bw = define_crossfade_params(fs)
    alpha = define_alpha(fs)
    if fft_len==None:
        fft_len = define_fft_len(fs)

    fft_len_half = fft_len / 2 + 1
    nfrms, ncoeffs_mag = m_mag_mel_log.shape

    # Unwarp and unlog features:===============================================
    # F0:
    v_f0    = np.exp(v_lf0)
    v_voi   = v_f0 > 1.0 # case voiced  (1.0 is used for safety)
    v_shift = f0_to_shift(v_f0, fs)

    # Magnitude mel-unwarp:
    if b_fbank_mel:
        m_mag = np.exp(la.sp_mel_unwarp_fbank(m_mag_mel_log, fft_len_half, alpha=alpha))
    else:
        m_mag = np.exp(la.sp_mel_unwarp(m_mag_mel_log, fft_len_half, alpha=alpha, in_type='log'))

    if alpha_phase is None:
        alpha_phase = alpha
    m_real, m_imag = phase_uncompress_type1_mcep(m_real_mel, m_imag_mel, alpha_phase, fft_len, fs)

    # Constant to variable frame rate:============================================
    if b_const_rate:
        const_rate_ms = 5.0
        interp_type = 'linear' #'quadratic' , 'cubic'
        v_shift, v_frm_locs_smpls = get_shifts_and_frm_locs_from_const_shifts(v_shift, const_rate_ms, fs, interp_type=interp_type)
        m_mag  = interp_from_const_to_variable_rate(m_mag,    v_frm_locs_smpls, const_rate_ms, fs, interp_type=interp_type)
        m_real = interp_from_const_to_variable_rate(m_real,   v_frm_locs_smpls, const_rate_ms, fs, interp_type=interp_type)
        m_imag = interp_from_const_to_variable_rate(m_imag,   v_frm_locs_smpls, const_rate_ms, fs, interp_type=interp_type)
        v_voi  = interp_from_const_to_variable_rate(v_voi, v_frm_locs_smpls, const_rate_ms, fs, interp_type=interp_type) > 0.5
        v_f0   = shift_to_f0(v_shift, v_voi, fs, out='f0', b_smooth=False)
        nfrms  = v_shift.size

    # Mask Generation:============================================================
    m_mask_per = np.zeros(m_mag.shape)
    m_ones     = np.ones((np.sum(v_voi.astype(int)), fft_len_half))
    m_mask_per[v_voi,:] = la.spectral_crossfade(m_ones, m_mask_per[v_voi,:], crsf_cf, crsf_bw, fs, freq_scale='hz', win_func=np.hanning)

    # Aperiodic Spectrum Generation:==============================================
    # Noise Gen:
    v_shift = v_shift.astype(int)
    v_pm    = la.shift_to_pm(v_shift)

    ns_len = v_pm[-1] + (v_pm[-1] - v_pm[-2])
    v_ns   = np.random.uniform(-1, 1, ns_len)

    # Noise Windowing:
    l_ns_win_funcs = [ np.hanning ] * nfrms
    if b_voi_ap_win:
        for i in xrange(nfrms):
            if v_voi[i]:
                l_ns_win_funcs[i] = voi_noise_window

    l_frm_ns, v_lens, v_pm_plus, v_shift_dummy, v_rights = windowing(v_ns, v_pm, win_func=l_ns_win_funcs)

    # Noise complex spectrum:
    m_frm_ns = la.frm_list_to_matrix(l_frm_ns, v_shift, fft_len)
    m_frm_ns = np.fft.fftshift(m_frm_ns, axes=1)
    m_ns_cmplx_spec = la.remove_hermitian_half(np.fft.fft(m_frm_ns))

    # Noise gain normalisation:
    m_ns_mag  = np.absolute(m_ns_cmplx_spec)

    noise_gain_voi = np.sqrt(np.exp(np.mean(la.log(m_ns_mag[v_voi,1:-1])**2)))
    noise_gain_unv = np.sqrt(np.exp(np.mean(la.log(m_ns_mag[~v_voi,1:-1])**2)))

    m_ns_cmplx_spec[v_voi,:]  = m_ns_cmplx_spec[v_voi,:] /  noise_gain_voi
    m_ns_cmplx_spec[~v_voi,:] = m_ns_cmplx_spec[~v_voi,:] / noise_gain_unv

    # Spectral Stamping of magnitude to noise spectrum:
    b_ap_min_phase_mag = False
    if b_ap_min_phase_mag:
        m_mag_min_phase_cmplx = la.build_min_phase_from_mag_spec(m_mag)
    else:
        m_mag_min_phase_cmplx = m_mag

    m_ap_cmplx_spec = m_ns_cmplx_spec * m_mag_min_phase_cmplx

    v_line = la.db(la.build_mel_curve(alpha, fft_len_half, amp=3.5) - 3.5, b_inv=True)
    m_ap_cmplx_spec[~v_voi,:] *= v_line


    # Periodic Spectrum Generation:============================================
    if per_phase_type=='magphase':
        m_per_cmplx_ph = m_real + m_imag * 1j

        # Normalisation and protection:
        m_per_cmplx_ph_mag = np.absolute(m_per_cmplx_ph)
        m_per_cmplx_ph_mag[m_per_cmplx_ph_mag==0.0] = 1.0
        m_per_cmplx_ph = m_per_cmplx_ph / m_per_cmplx_ph_mag

        m_per_cmplx_spec = m_mag * m_per_cmplx_ph

    if per_phase_type=='linear':
        m_per_cmplx_spec = m_mag

    elif per_phase_type=='min_phase':
        m_per_cmplx_spec  = la.build_min_phase_from_mag_spec(m_mag)

    # Debug. Voi segments - compensation filter: # Not really noticeable.
    # (NOTE: This only has been tested with fs=48kHz and alpha=0.77)
    v_line = la.db(la.build_mel_curve(0.6, fft_len_half, amp=2.0), b_inv=True)
    m_per_cmplx_spec[v_voi,:] *= v_line


    # Waveform Generation:=====================================================
    # Applying mask:
    crsf_curve_fact = 0.5 # Spectral crossfade courve factor
    m_per_cmplx_spec *= (m_mask_per**crsf_curve_fact)
    m_ap_cmplx_spec  *= ((1 - m_mask_per)**crsf_curve_fact)

    # Protection:
    m_per_cmplx_spec[m_mask_per==0.0] = 0 + 0j
    m_ap_cmplx_spec[m_mask_per==1.0]  = 0 + 0j

    # Synthesis:
    m_syn_cmplx = m_per_cmplx_spec + m_ap_cmplx_spec

    #Protection:
    m_syn_cmplx[:,0].real  = np.absolute(m_syn_cmplx[:,0])
    m_syn_cmplx[:,-1].real = np.absolute(m_syn_cmplx[:,-1])
    m_syn_cmplx[:,0].imag  = 0.0
    m_syn_cmplx[:,-1].imag = 0.0

    m_syn_cmplx = la.add_hermitian_half(m_syn_cmplx, data_type='complex')
    m_syn_frms  = np.fft.ifft(m_syn_cmplx).real
    m_syn_frms  = np.fft.fftshift(m_syn_frms, axes=1)


    # Window anti-ringing:
    frmlen = m_syn_frms.shape[1]
    v_shift_ext = np.r_[v_shift[0], v_shift, v_shift[-1], v_shift[-1]] # recover first shift (estimate)
    for nxf in xrange(nfrms):
        v_win = la.gen_centr_win(v_shift_ext[nxf]+v_shift_ext[nxf+1], v_shift_ext[nxf+2]+v_shift_ext[nxf+3], frmlen, win_func=raised_hanning, b_fill_w_bound_val=True)
        m_syn_frms[nxf,:] *= v_win


    v_syn_sig = ola(m_syn_frms, v_pm, win_func=None)

    # HPF - Output:============================================================
    # NOTE: The HPF unbalance the polarity of the signal, because it removed DC!

    if b_out_hpf:
        '''
        fc    = 60
        order = 4
        fc_norm   = fc / (fs / 2.0)
        bc, ac    = signal.ellip(order,0.5 , 80, fc_norm, btype='highpass')
        v_syn_sig = signal.lfilter(bc, ac, v_syn_sig)
        #'''

        # Butterworth:
        order = 4
        fc = 40 # in Hz
        fc_norm = fc /(fs/2.0)
        v_b, v_a = signal.butter(order, fc_norm, btype='highpass')
        v_syn_sig = signal.lfilter(v_b, v_a, v_syn_sig)

    return v_syn_sig

#==============================================================================
def synthesis_from_compressed_type1_with_phase_comp(m_mag_mel_log, m_real_mel, m_imag_mel, v_lf0, fs, fft_len=None,
                                    b_voi_ap_win=True, b_fbank_mel=False, const_rate_ms=-1.0, per_phase_type='magphase'):

    '''
    synthesis_from_compressed_type1 with phase compression based on filter bank. It didn't work very well according to experiments.

    b_fbank_mel: If True, Mel compression done by the filter bank approach. Otherwise, it uses sptk mcep related funcs.
    per_phase_type: 'magphase', 'min_phase', or 'linear'
    '''

    # Setting up constants:====================================================
    crsf_cf, crsf_bw = define_crossfade_params(fs)
    alpha = define_alpha(fs)
    if fft_len==None:
        fft_len = define_fft_len(fs)

    fft_len_half = fft_len / 2 + 1
    nfrms, ncoeffs_mag = m_mag_mel_log.shape

    # Debug - compensation filter:
    #m_mag_mel_log = post_filter(m_mag_mel_log, fs, av_len_at_zero=3, av_len_at_nyq=3, boost_at_zero=1.0, boost_at_nyq=1.4)

    # Unwarp and unlog features:===============================================
    # F0:
    v_f0    = np.exp(v_lf0)
    v_voi   = v_f0 > 1.0 # case voiced  (1.0 is used for safety)
    v_shift = f0_to_shift(v_f0, fs)

    # Magnitude mel-unwarp:
    if b_fbank_mel:
        m_mag = np.exp(la.sp_mel_unwarp_fbank(m_mag_mel_log, fft_len_half, alpha=alpha))
    else:
        m_mag = np.exp(la.sp_mel_unwarp(m_mag_mel_log, fft_len_half, alpha=alpha, in_type='log'))


    # Debug: ------------------------------------------------------------------------
    # Phase feats mel-unwarp:
    # Just only of one of these is used:
    # m_real_mel, m_imag_mel, crsf_cf, crsf_bw, alpha, fft_len, fs

    m_real, m_imag = phase_uncompress_fbank(m_real_mel, m_imag_mel, crsf_cf, crsf_bw, alpha, fft_len, fs)

    # Constant to variable frame rate:============================================
    if const_rate_ms>0.0:
        interp_type = 'linear' #'quadratic' , 'cubic'
        v_shift, v_frm_locs_smpls = get_shifts_and_frm_locs_from_const_shifts(v_shift, const_rate_ms, fs, interp_type=interp_type)
        m_mag  = interp_from_const_to_variable_rate(m_mag,    v_frm_locs_smpls, const_rate_ms, fs, interp_type=interp_type)
        m_real = interp_from_const_to_variable_rate(m_real,   v_frm_locs_smpls, const_rate_ms, fs, interp_type=interp_type)
        m_imag = interp_from_const_to_variable_rate(m_imag,   v_frm_locs_smpls, const_rate_ms, fs, interp_type=interp_type)
        v_voi  = interp_from_const_to_variable_rate(v_voi, v_frm_locs_smpls, const_rate_ms, fs, interp_type=interp_type) > 0.5
        v_f0   = shift_to_f0(v_shift, v_voi, fs, out='f0', b_smooth=False)
        nfrms  = v_shift.size

    # Mask Generation:============================================================
    m_mask_per = np.zeros(m_mag.shape)
    m_ones     = np.ones((np.sum(v_voi.astype(int)), fft_len_half))
    m_mask_per[v_voi,:] = la.spectral_crossfade(m_ones, m_mask_per[v_voi,:], crsf_cf, crsf_bw, fs, freq_scale='hz', win_func=np.hanning)

    # Aperiodic Spectrum Generation:==============================================
    # Noise Gen:
    v_shift = v_shift.astype(int)
    v_pm    = la.shift_to_pm(v_shift)

    ns_len = v_pm[-1] + (v_pm[-1] - v_pm[-2])
    v_ns   = np.random.uniform(-1, 1, ns_len)

    # Noise Windowing:
    l_ns_win_funcs = [ np.hanning ] * nfrms
    if b_voi_ap_win:
        for i in xrange(nfrms):
            if v_voi[i]:
                l_ns_win_funcs[i] = voi_noise_window

    l_frm_ns, v_lens, v_pm_plus, v_shift_dummy, v_rights = windowing(v_ns, v_pm, win_func=l_ns_win_funcs)

    # Noise complex spectrum:
    m_frm_ns = la.frm_list_to_matrix(l_frm_ns, v_shift, fft_len)
    m_frm_ns = np.fft.fftshift(m_frm_ns, axes=1)
    m_ns_cmplx_spec = la.remove_hermitian_half(np.fft.fft(m_frm_ns))

    # Noise gain normalisation:
    m_ns_mag  = np.absolute(m_ns_cmplx_spec)

    # Debug:
    #noise_gain_voi = np.sqrt(np.mean(m_ns_mag[v_voi,1:-1]**2)) / 1.5
    #noise_gain_unv = np.sqrt(np.mean(m_ns_mag[~v_voi,1:-1]**2))

    noise_gain_voi = np.sqrt(np.exp(np.mean(la.log(m_ns_mag[v_voi,1:-1])**2)))
    noise_gain_unv = np.sqrt(np.exp(np.mean(la.log(m_ns_mag[~v_voi,1:-1])**2)))

    m_ns_cmplx_spec[v_voi,:]  = m_ns_cmplx_spec[v_voi,:] /  noise_gain_voi
    m_ns_cmplx_spec[~v_voi,:] = m_ns_cmplx_spec[~v_voi,:] / noise_gain_unv

    # Spectral Stamping of magnitude to noise spectrum:
    b_ap_min_phase_mag = False
    if b_ap_min_phase_mag:
        m_mag_min_phase_cmplx = la.build_min_phase_from_mag_spec(m_mag)
    else:
        m_mag_min_phase_cmplx = m_mag

    m_ap_cmplx_spec = m_ns_cmplx_spec * m_mag_min_phase_cmplx

    # Debug. Unv segments - compensation filter:
    # (NOTE: This only has been tested with fs=48kHz and alpha=0.77)
    #v_line = la.db(la.build_mel_curve(0.60, fft_len_half, amp=3.0), b_inv=True)
    #v_line = la.db(la.build_mel_curve(alpha, fft_len_half, amp=np.pi) - np.pi, b_inv=True)
    v_line = la.db(la.build_mel_curve(alpha, fft_len_half, amp=3.5) - 3.5, b_inv=True)
    #v_line = la.db(la.build_mel_curve(0.66, fft_len_half, amp=np.pi) - np.pi, b_inv=True)
    m_ap_cmplx_spec[~v_voi,:] *= v_line


    # Periodic Spectrum Generation:============================================
    if per_phase_type=='magphase':
        m_per_cmplx_ph = m_real + m_imag * 1j

        # Normalisation and protection:
        m_per_cmplx_ph_mag = np.absolute(m_per_cmplx_ph)
        m_per_cmplx_ph_mag[m_per_cmplx_ph_mag==0.0] = 1.0
        m_per_cmplx_ph = m_per_cmplx_ph / m_per_cmplx_ph_mag

        m_per_cmplx_spec = m_mag * m_per_cmplx_ph

        if False:
            nx=73; figure(); plot(m_real[nx,:]); plot(m_imag[nx,:]); grid()
            nx=146; figure(); plot(np.angle(m_per_cmplx_ph[nx,:])); grid()
            nx=146; figure(); plot(np.angle(m_per_cmplx_ph[nx,:])); grid()

    if per_phase_type=='linear':
        m_per_cmplx_spec = m_mag

    elif per_phase_type=='min_phase':
        m_per_cmplx_spec  = la.build_min_phase_from_mag_spec(m_mag)

    # Debug. Voi segments - compensation filter: # Not really noticeable.
    # (NOTE: This only has been tested with fs=48kHz and alpha=0.77)
    v_line = la.db(la.build_mel_curve(0.6, fft_len_half, amp=2.0), b_inv=True)
    m_per_cmplx_spec[v_voi,:] *= v_line


    # Waveform Generation:=====================================================
    # Applying mask:
    crsf_curve_fact = 0.5 # Spectral crossfade courve factor
    m_per_cmplx_spec *= (m_mask_per**crsf_curve_fact)
    m_ap_cmplx_spec  *= ((1 - m_mask_per)**crsf_curve_fact)

    # Protection:
    m_per_cmplx_spec[m_mask_per==0.0] = 0 + 0j
    m_ap_cmplx_spec[m_mask_per==1.0]  = 0 + 0j

    # Synthesis:
    m_syn_cmplx = m_per_cmplx_spec + m_ap_cmplx_spec

    #Protection:
    #m_syn_cmplx[:,0].real  = np.absolute(m_syn_cmplx[:,0])
    #m_syn_cmplx[:,-1].real = np.absolute(m_syn_cmplx[:,-1])
    m_syn_cmplx[:,0].imag  = 0.0
    m_syn_cmplx[:,-1].imag = 0.0

    m_syn_cmplx = la.add_hermitian_half(m_syn_cmplx, data_type='complex')
    m_syn_frms  = np.fft.ifft(m_syn_cmplx).real
    m_syn_frms  = np.fft.fftshift(m_syn_frms, axes=1)

    # Debug:
    #m_syn_frms[~v_voi,:] = 0.0
    #m_syn_frms[71,:] = 0.0
    #m_syn_frms[72,:] = 0.0
    #m_syn_frms[73,:] = 0.0

    # Window anti-ringing:
    frmlen = m_syn_frms.shape[1]
    v_shift_ext = np.r_[v_shift[0], v_shift, v_shift[-1], v_shift[-1]] # recover first shift (estimate)
    for nxf in xrange(nfrms):
        v_win = la.gen_centr_win(v_shift_ext[nxf]+v_shift_ext[nxf+1], v_shift_ext[nxf+2]+v_shift_ext[nxf+3], frmlen, win_func=raised_hanning, b_fill_w_bound_val=True)
        m_syn_frms[nxf,:] *= v_win


    if False:
        # En hvd_595 problema de spike around frame 290
        # Frame 73 in hvd_595 producing pre ringing in bins 290 and 338.
        m_syn_mag_db = la.db(np.absolute(m_syn_cmplx))
        plm(m_syn_mag_db)
        nx=73; figure(); plot(np.angle(m_syn_cmplx[nx,:]), '.-'); grid()
        nx=73; figure(); plot(m_syn_mag_db[nx,:], '.-'); grid()

        plm(m_syn_frms)
        pl(m_syn_frms[292:294+1,:].T)
        pl(m_syn_frms[292:297+1,:].T)

        pl(m_syn_frms[292,:])

        pl(m_syn_frms[252:258+1,:].T)


    v_syn_sig = ola(m_syn_frms, v_pm, win_func=None)

    # HPF - Output:============================================================
    # NOTE: The HPF unbalance the polarity of the signal, because it removed DC!
    '''
    fc    = 60
    order = 4
    fc_norm   = fc / (fs / 2.0)
    bc, ac    = signal.ellip(order,0.5 , 80, fc_norm, btype='highpass')
    v_syn_sig = signal.lfilter(bc, ac, v_syn_sig)
    #'''

    # Butterworth:
    order = 4
    fc = 40 # in Hz
    fc_norm = fc /(fs/2.0)
    v_b, v_a = signal.butter(order, fc_norm, btype='highpass')
    v_syn_sig = signal.lfilter(v_b, v_a, v_syn_sig)

    if False:
        fvtool(v_b, v_a, fs=48000)

    return v_syn_sig


#==============================================================================
def phase_uncompress_type1_mcep(m_real_mel, m_imag_mel, alpha, fft_len, fs):

    ncoeffs_comp = m_real_mel.shape[1]
    crsf_cf = define_crossfade_params(fs)[0]
    nbins_mel_for_phase_comp = get_num_full_mel_coeffs_from_num_phase_coeffs(crsf_cf, ncoeffs_comp, alpha, fs)

    f_intrp_real = interpolate.interp1d(np.arange(ncoeffs_comp), m_real_mel, kind='nearest', fill_value='extrapolate')
    f_intrp_imag = interpolate.interp1d(np.arange(ncoeffs_comp), m_imag_mel, kind='nearest', fill_value='extrapolate')

    m_real_mel = f_intrp_real(np.arange(nbins_mel_for_phase_comp))
    m_imag_mel = f_intrp_imag(np.arange(nbins_mel_for_phase_comp))

    fft_len_half = 1 + fft_len/2
    m_real = la.sp_mel_unwarp(m_real_mel, fft_len_half, alpha=alpha, in_type='log')
    m_imag = la.sp_mel_unwarp(m_imag_mel, fft_len_half, alpha=alpha, in_type='log')

    return m_real, m_imag

#==============================================================================
def phase_uncompress_type1(m_real_mel, m_imag_mel, alpha, fft_len, ncoeffs_mag):
    ncoeffs_comp = m_real_mel.shape[1]
    f_intrp_real = interpolate.interp1d(np.arange(ncoeffs_comp), m_real_mel, kind='nearest', fill_value='extrapolate')
    f_intrp_imag = interpolate.interp1d(np.arange(ncoeffs_comp), m_imag_mel, kind='nearest', fill_value='extrapolate')

    m_real_mel = f_intrp_real(np.arange(ncoeffs_mag))
    m_imag_mel = f_intrp_imag(np.arange(ncoeffs_mag))

    fft_len_half = 1 + fft_len/2
    m_real = la.sp_mel_unwarp(m_real_mel, fft_len_half, alpha=alpha, in_type='log')
    m_imag = la.sp_mel_unwarp(m_imag_mel, fft_len_half, alpha=alpha, in_type='log')

    return m_real, m_imag

#==============================================================================
def synthesis_from_compressed_type1(m_mag_mel_log, m_real_mel, m_imag_mel, v_lf0, fs, fft_len=None,
                                    b_voi_ap_win=True, b_fbank_mel=False, b_const_rate=False, per_phase_type='magphase'):

    '''
    b_fbank_mel: If True, Mel compression done by the filter bank approach. Otherwise, it uses sptk mcep related funcs.
    per_phase_type: 'magphase', 'min_phase', or 'linear'
    '''

    # Setting up constants:====================================================
    crsf_cf, crsf_bw = define_crossfade_params(fs)
    alpha = define_alpha(fs)
    if fft_len==None:
        fft_len = define_fft_len(fs)

    fft_len_half = fft_len / 2 + 1
    nfrms, ncoeffs_mag = m_mag_mel_log.shape

    # Debug - compensation filter:
    #m_mag_mel_log = post_filter(m_mag_mel_log, fs, av_len_at_zero=3, av_len_at_nyq=3, boost_at_zero=1.0, boost_at_nyq=1.4)

    # Unwarp and unlog features:===============================================
    # F0:
    v_f0    = np.exp(v_lf0)
    v_voi   = v_f0 > 1.0 # case voiced  (1.0 is used for safety)
    v_shift = f0_to_shift(v_f0, fs)

    # Magnitude mel-unwarp:
    if b_fbank_mel:
        m_mag = np.exp(la.sp_mel_unwarp_fbank(m_mag_mel_log, fft_len_half, alpha=alpha))
    else:
        m_mag = np.exp(la.sp_mel_unwarp(m_mag_mel_log, fft_len_half, alpha=alpha, in_type='log'))

    # Phase feats mel-unwarp:
    # m_real_mel, m_imag_mel, alpha, fft_len, ncoeffs_mag,

    m_real, m_imag = phase_uncompress_type1(m_real_mel, m_imag_mel, alpha, fft_len, ncoeffs_mag)
    # ncoeffs_comp = m_real_mel.shape[1]
    # f_intrp_real = interpolate.interp1d(np.arange(ncoeffs_comp), m_real_mel, kind='nearest', fill_value='extrapolate')
    # f_intrp_imag = interpolate.interp1d(np.arange(ncoeffs_comp), m_imag_mel, kind='nearest', fill_value='extrapolate')

    # m_real_mel = f_intrp_real(np.arange(ncoeffs_mag))
    # m_imag_mel = f_intrp_imag(np.arange(ncoeffs_mag))

    # m_real = la.sp_mel_unwarp(m_real_mel, fft_len_half, alpha=alpha, in_type='log')
    # m_imag = la.sp_mel_unwarp(m_imag_mel, fft_len_half, alpha=alpha, in_type='log')

    # Constant to variable frame rate:============================================
    if b_const_rate:
        const_rate_ms = 5.0
        interp_type = 'linear' #'quadratic' , 'cubic'
        v_shift, v_frm_locs_smpls = get_shifts_and_frm_locs_from_const_shifts(v_shift, const_rate_ms, fs, interp_type=interp_type)
        m_mag  = interp_from_const_to_variable_rate(m_mag,  v_frm_locs_smpls, const_rate_ms, fs, interp_type=interp_type)
        m_real = interp_from_const_to_variable_rate(m_real, v_frm_locs_smpls, const_rate_ms, fs, interp_type=interp_type)
        m_imag = interp_from_const_to_variable_rate(m_imag, v_frm_locs_smpls, const_rate_ms, fs, interp_type=interp_type)
        v_voi  = interp_from_const_to_variable_rate(v_voi,  v_frm_locs_smpls, const_rate_ms, fs, interp_type=interp_type) > 0.5
        v_f0   = shift_to_f0(v_shift, v_voi, fs, out='f0', b_smooth=False)
        nfrms  = v_shift.size

    # Mask Generation:============================================================
    m_mask_per = np.zeros(m_mag.shape)
    m_ones     = np.ones((np.sum(v_voi.astype(int)), fft_len_half))
    m_mask_per[v_voi,:] = la.spectral_crossfade(m_ones, m_mask_per[v_voi,:], crsf_cf, crsf_bw, fs, freq_scale='hz', win_func=np.hanning)

    # Aperiodic Spectrum Generation:==============================================
    # Noise Gen:
    v_shift = v_shift.astype(int)
    v_pm    = la.shift_to_pm(v_shift)

    ns_len = v_pm[-1] + (v_pm[-1] - v_pm[-2])
    v_ns   = np.random.uniform(-1, 1, ns_len)

    # Noise Windowing:
    l_ns_win_funcs = [ np.hanning ] * nfrms
    if b_voi_ap_win:
        for i in xrange(nfrms):
            if v_voi[i]:
                l_ns_win_funcs[i] = voi_noise_window

    l_frm_ns, v_lens, v_pm_plus, v_shift_dummy, v_rights = windowing(v_ns, v_pm, win_func=l_ns_win_funcs)

    # Noise complex spectrum:
    m_frm_ns = la.frm_list_to_matrix(l_frm_ns, v_shift, fft_len)
    m_frm_ns = np.fft.fftshift(m_frm_ns, axes=1)
    m_ns_cmplx_spec = la.remove_hermitian_half(np.fft.fft(m_frm_ns))

    # Noise gain normalisation:
    m_ns_mag  = np.absolute(m_ns_cmplx_spec)

    # Debug:
    #noise_gain_voi = np.sqrt(np.mean(m_ns_mag[v_voi,1:-1]**2)) / 1.5
    #noise_gain_unv = np.sqrt(np.mean(m_ns_mag[~v_voi,1:-1]**2))

    noise_gain_voi = np.sqrt(np.exp(np.mean(la.log(m_ns_mag[v_voi,1:-1])**2)))
    noise_gain_unv = np.sqrt(np.exp(np.mean(la.log(m_ns_mag[~v_voi,1:-1])**2)))

    m_ns_cmplx_spec[v_voi,:]  = m_ns_cmplx_spec[v_voi,:] /  noise_gain_voi
    m_ns_cmplx_spec[~v_voi,:] = m_ns_cmplx_spec[~v_voi,:] / noise_gain_unv

    # Spectral Stamping of magnitude to noise spectrum:
    b_ap_min_phase_mag = False
    if b_ap_min_phase_mag:
        m_mag_min_phase_cmplx = la.build_min_phase_from_mag_spec(m_mag)
    else:
        m_mag_min_phase_cmplx = m_mag

    m_ap_cmplx_spec = m_ns_cmplx_spec * m_mag_min_phase_cmplx

    # Debug. Unv segments - compensation filter:
    # (NOTE: This only has been tested with fs=48kHz and alpha=0.77)
    #v_line = la.db(la.build_mel_curve(0.60, fft_len_half, amp=3.0), b_inv=True)
    #v_line = la.db(la.build_mel_curve(alpha, fft_len_half, amp=np.pi) - np.pi, b_inv=True)
    v_line = la.db(la.build_mel_curve(alpha, fft_len_half, amp=3.5) - 3.5, b_inv=True)
    #v_line = la.db(la.build_mel_curve(0.66, fft_len_half, amp=np.pi) - np.pi, b_inv=True)
    m_ap_cmplx_spec[~v_voi,:] *= v_line



    # Periodic Spectrum Generation:============================================
    if per_phase_type=='magphase':
        m_per_cmplx_ph = m_real + m_imag * 1j

        # Normalisation and protection:
        m_per_cmplx_ph_mag = np.absolute(m_per_cmplx_ph)
        m_per_cmplx_ph_mag[m_per_cmplx_ph_mag==0.0] = 1.0
        m_per_cmplx_ph = m_per_cmplx_ph / m_per_cmplx_ph_mag

        m_per_cmplx_spec = m_mag * m_per_cmplx_ph

    if per_phase_type=='linear':
        m_per_cmplx_spec = m_mag

    elif per_phase_type=='min_phase':
        m_per_cmplx_spec  = la.build_min_phase_from_mag_spec(m_mag)

    # Debug. Voi segments - compensation filter: # Not really noticeable.
    # (NOTE: This only has been tested with fs=48kHz and alpha=0.77)
    v_line = la.db(la.build_mel_curve(0.6, fft_len_half, amp=2.0), b_inv=True)
    m_per_cmplx_spec[v_voi,:] *= v_line


    # Waveform Generation:=====================================================
    # Applying mask:
    crsf_curve_fact = 0.5 # Spectral crossfade courve factor
    m_per_cmplx_spec *= (m_mask_per**crsf_curve_fact)
    m_ap_cmplx_spec  *= ((1 - m_mask_per)**crsf_curve_fact)

    # Protection:
    m_per_cmplx_spec[m_mask_per==0.0] = 0 + 0j
    m_ap_cmplx_spec[m_mask_per==1.0]  = 0 + 0j

    # Synthesis:
    m_syn_cmplx = m_per_cmplx_spec + m_ap_cmplx_spec
    m_syn_cmplx = la.add_hermitian_half(m_syn_cmplx, data_type='complex')
    m_syn_frms  = np.fft.ifft(m_syn_cmplx).real
    m_syn_frms  = np.fft.fftshift(m_syn_frms, axes=1)

    # Debug:
    #m_syn_frms[~v_voi,:] = 0.0

    v_syn_sig   = ola(m_syn_frms, v_pm, win_func=None)

    # HPF - Output:============================================================
    fc    = 60
    order = 4
    fc_norm   = fc / (fs / 2.0)
    bc, ac    = signal.ellip(order,0.5 , 80, fc_norm, btype='highpass')
    v_syn_sig = signal.lfilter(bc, ac, v_syn_sig)

    return v_syn_sig

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


def synthesis_from_compressed_type2(m_mag_mel_log, m_real_mel, m_imag_mel, v_lf0, fs, fft_len=None, hf_slope_coeff=1.0,
                                                    b_voi_ap_win=True, b_norm_mag=False, v_lgain=None, const_rate_ms=-1.0):

    # Constants for spectral crossfade (in Hz):
    crsf_cf, crsf_bw = define_crossfade_params(fs)
    alpha = define_alpha(fs)
    if fft_len==None:
        fft_len = define_fft_len(fs)

    fft_len_half = fft_len / 2 + 1
    v_f0 = np.exp(v_lf0)
    nfrms, ncoeffs_mag = m_mag_mel_log.shape
    ncoeffs_comp = m_real_mel.shape[1]

    # Debug:
    b_norm_mag = False # CHECK THIS. ONLY FOR DEBUG!!

    if False:
        from libplot import lp
        nx=172; lp.figure(); lp.plot(m_nat[nx,:]); lp.plot(m_mag_mel_log[nx,:]); lp.grid()

    # Extract gain:
    if b_norm_mag:
        v_lgain = m_mag_mel_log[:,0].copy()
        v_mean  = np.mean(m_mag_mel_log[:,1:], axis=1)
        m_mag_mel_log = (m_mag_mel_log.T - v_mean).T
        m_mag_mel_log[:,0] = m_mag_mel_log[:,1]
        #m_mag_mel_log[:,0] = -8.0
        #m_mag_mel_log = m_mag_mel_log + v_lgain[:,None]
        m_mag_mel_log = m_mag_mel_log - v_lgain[:,None]

        '''
        cf_freq = 30.0 # Hz
        cf_bin  = lu.round_to_int(cf_freq * fft_len / float(fs))
        m_mag[:,:(cf_bin+1)] = 0.0
        '''

    # Magnitude mel-unwarp:----------------------------------------------------
    m_mag = np.exp(la.sp_mel_unwarp(m_mag_mel_log, fft_len_half, alpha=alpha, in_type='log'))


    # Complex mel-unwarp:------------------------------------------------------
    f_intrp_real = interpolate.interp1d(np.arange(ncoeffs_comp), m_real_mel, kind='nearest', fill_value='extrapolate')
    f_intrp_imag = interpolate.interp1d(np.arange(ncoeffs_comp), m_imag_mel, kind='nearest', fill_value='extrapolate')

    m_real_mel = f_intrp_real(np.arange(ncoeffs_mag))
    m_imag_mel = f_intrp_imag(np.arange(ncoeffs_mag))

    m_real = la.sp_mel_unwarp(m_real_mel, fft_len_half, alpha=alpha, in_type='log')
    m_imag = la.sp_mel_unwarp(m_imag_mel, fft_len_half, alpha=alpha, in_type='log')

    # If data is constant rate:------------------------------------------------
    v_shift = f0_to_shift(v_f0, fs)
    if const_rate_ms>0.0:
        #v_shift = f0_to_shift(v_f0, fs)
        v_shift, v_frm_locs_smpls = get_shifts_and_frm_locs_from_const_shifts(v_shift, const_rate_ms, fs, interp_type='linear')
        m_mag  = interp_from_const_to_variable_rate(m_mag, v_frm_locs_smpls, const_rate_ms, fs, interp_type='linear')
        m_real = interp_from_const_to_variable_rate(m_real, v_frm_locs_smpls, const_rate_ms, fs, interp_type='linear')
        m_imag = interp_from_const_to_variable_rate(m_imag, v_frm_locs_smpls, const_rate_ms, fs, interp_type='linear')
        v_voi  = interp_from_const_to_variable_rate(v_f0>0.0, v_frm_locs_smpls, const_rate_ms, fs, interp_type='linear') > 0.5
        v_f0   = shift_to_f0(v_shift, v_voi, fs, out='f0', b_smooth=False)
        nfrms  = v_shift.size

    # Noise Gen:---------------------------------------------------------------
    v_shift = v_shift.astype(int)
    v_pm   = la.shift_to_pm(v_shift)

    ns_len = v_pm[-1] + (v_pm[-1] - v_pm[-2])
    v_ns   = np.random.uniform(-1, 1, ns_len)

    # Noise Windowing:---------------------------------------------------------
    l_ns_win_funcs = [ np.hanning ] * nfrms
    v_voi = v_f0 > 1 # case voiced  (1 is used for safety)
    if b_voi_ap_win:
        for i in xrange(nfrms):
            if v_voi[i]:
                l_ns_win_funcs[i] = voi_noise_window

    l_frm_ns, v_lens, v_pm_plus, v_shift_dummy, v_rights = windowing(v_ns, v_pm, win_func=l_ns_win_funcs)   # Checkear!!

    m_frm_ns  = la.frm_list_to_matrix(l_frm_ns, v_shift, fft_len)
    m_frm_ns  = np.fft.fftshift(m_frm_ns, axes=1)
    m_ns_cmplx = la.remove_hermitian_half(np.fft.fft(m_frm_ns))

    # AP-Mask:-----------------------------------------------------------------
    # Norm gain:
    m_ns_mag  = np.absolute(m_ns_cmplx)
    rms_noise = np.sqrt(np.mean(m_ns_mag**2)) # checkear!!!!
    m_ap_mag_smth = np.ones(m_ns_mag.shape)
    m_ap_mag_smth = m_mag * m_ap_mag_smth / rms_noise

    m_zeros = np.zeros((nfrms, fft_len_half))
    m_ap_mag_smth[v_voi,:] = la.spectral_crossfade(m_zeros[v_voi,:], m_ap_mag_smth[v_voi,:], crsf_cf, crsf_bw, fs, freq_scale='hz')

    # HF - enhancement:
    v_slope  = np.linspace(1, hf_slope_coeff, num=fft_len_half)
    m_ap_mag_smth[~v_voi,:] = m_ap_mag_smth[~v_voi,:] * v_slope

    # Det-Mask:----------------------------------------------------------------
    m_det_mask = m_mag
    m_det_mask[~v_voi,:] = 0
    m_det_mask[v_voi,:]  = la.spectral_crossfade(m_det_mask[v_voi,:], m_zeros[v_voi,:], crsf_cf, crsf_bw, fs, freq_scale='hz')

    # Applying masks:----------------------------------------------------------
    m_ap_cmplx  = m_ap_mag_smth  * m_ns_cmplx
    m_det_cmplx = m_real + m_imag * 1j

    # Protection:
    m_det_cmplx_abs = np.absolute(m_det_cmplx)
    m_det_cmplx_abs[m_det_cmplx_abs==0.0] = 1.0

    m_det_cmplx = m_det_mask * m_det_cmplx / m_det_cmplx_abs

    # bin width: bw=11.71875 Hz
    # To Time domain:-------------------------------------------------------------
    m_syn_cmplx = la.add_hermitian_half(m_ap_cmplx + m_det_cmplx, data_type='complex')
    m_syn_td    = np.fft.ifft(m_syn_cmplx).real
    m_syn_td    = np.fft.fftshift(m_syn_td, axes=1)

    # Apply window anti-ringing:----------------------------------------------------
    frmlen = m_syn_td.shape[1]
    v_shift_ext = np.r_[v_shift[0], v_shift, v_shift[-1], v_shift[-1]] # recover first shift (estimate)
    for nxf in xrange(nfrms):
        v_win = la.gen_centr_win(v_shift_ext[nxf]+v_shift_ext[nxf+1], v_shift_ext[nxf+2]+v_shift_ext[nxf+3], frmlen, win_func=raised_hanning, b_fill_w_bound_val=True)
        m_syn_td[nxf,:] *= v_win

    # Apply gain:
    '''
    if b_norm_mag:
        v_gain = np.exp(v_lgain)
        v_shift_ext2 = np.r_[v_shift, v_shift[-1]]
        nx_cntr  = np.floor(frmlen / 2.0).astype(int)
        for nxf in xrange(nfrms):
            curr_frm = m_syn_td[nxf,:]

            if v_voi[nxf]==1: #Voiced case
                curr_gain = np.max(np.abs(curr_frm))
                m_syn_td[nxf,:] = curr_frm * (v_gain[nxf]/curr_gain)
            else: #unvoiced case
                curr_gain = np.std(curr_frm[(nx_cntr-v_shift_ext2[nxf]):(nx_cntr+v_shift_ext2[nxf+1])])
                #m_syn_td[nxf,:] = curr_frm * v_gain[nxf]
                #m_syn_td[nxf,:] = curr_frm * (v_gain[nxf]/curr_gain)
    '''

    # PSOLA:--------------------------------------------------------------------------
    v_syn_sig = ola(m_syn_td, v_pm, win_func=None)

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
def interp_from_variable_to_const_frm_rate(m_data, v_pm_smpls, const_rate_ms, fs, interp_type='linear'):
    
    dp = lu.DimProtect(m_data)

    dur_total_smpls  = v_pm_smpls[-1]
    const_rate_smpls = fs * const_rate_ms / 1000
    #cons_frm_rate_frm_len = 2 * frm_rate_smpls # This assummed according to the Merlin code. E.g., frame_number = int((end_time - start_time)/50000)
    v_c_rate_centrs_smpls = np.arange(const_rate_smpls, dur_total_smpls, const_rate_smpls)

    # Interpolation:
    if v_pm_smpls[0]>0: # Protection
        f_interp = interpolate.interp1d(np.r_[0, v_pm_smpls], np.vstack((m_data[0,:], m_data)), axis=0, kind=interp_type)
    else:
        f_interp = interpolate.interp1d(v_pm_smpls, m_data, axis=0, kind=interp_type)

    m_data_const_rate = f_interp(v_c_rate_centrs_smpls) 

    dp.end(m_data_const_rate)


    return m_data_const_rate

#==============================================================================
def interp_from_const_to_variable_rate(m_data, v_frm_locs_smpls, frm_rate_ms, fs, interp_type='linear'):

    n_c_rate_frms  = np.size(m_data,0)                
    frm_rate_smpls = fs * frm_rate_ms / 1000
    
    v_c_rate_centrs_smpls = frm_rate_smpls * np.arange(1,n_c_rate_frms+1)

    f_interp     = interpolate.interp1d(v_c_rate_centrs_smpls, m_data, axis=0, kind=interp_type)            
    m_data_intrp = f_interp(v_frm_locs_smpls) 
    
    return m_data_intrp 

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



def post_filter_dev(m_mag_mel_log, fs, av_len_at_zero=None, av_len_at_nyq=None, boost_at_zero=None, boost_at_nyq=None):

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

    # Debug:
    m_mag_mel_log_norm = np.zeros(m_mag_mel_log.shape)

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
        m_mag_mel_log_norm[nxf,:] = v_mag_mel_log_norm

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

    return m_mag_mel_log_enh, m_mag_mel_log_norm
    #return m_mag_mel_log_enh



def win_squared(L):
    v_win = np.zeros(L)
    quarter = np.floor(L / 4.0).astype(int)
    half = np.floor(L / 2.0).astype(int)
    v_win[quarter:quarter+half] = 1.0
    return v_win

def get_num_full_mel_coeffs_from_num_phase_coeffs(freq_hz, nbins_phase, alpha, fs):

    crsf_cw = 2 * np.pi * freq_hz / float(fs)
    crsf_cf_mel = np.arctan(  (1-alpha**2) * np.sin(crsf_cw) / ((1+alpha**2)*np.cos(crsf_cw) - 2*alpha) )
    if crsf_cf_mel<0:
        crsf_cf_mel += np.pi

    nmelcoeffs = lu.round_to_int(1 + (np.pi * (nbins_phase - 1) / float(crsf_cf_mel)))
    return nmelcoeffs


def format_for_modelling(m_mag, m_real, m_imag, v_f0, fs, nbins_mel=60, nbins_phase=45, b_mag_fbank_mel=False, alpha_phase=None):
    '''
    b_fbank_mel: If True, Mel compression done by the filter bank approach. Otherwise, it uses sptk mcep related funcs.
    '''

    # alpha:
    alpha = define_alpha(fs)

    # f0 to Smoothed lf0:
    v_voi = (v_f0>0).astype('float')
    v_f0_smth  = v_voi * signal.medfilt(v_f0)
    v_lf0_smth = la.f0_to_lf0(v_f0_smth)

    # Mag to Log-Mag-Mel (compression):
    if b_mag_fbank_mel:
        m_mag_mel = la.sp_mel_warp_fbank(m_mag, nbins_mel, alpha=alpha)
        #m_mag_mel = la.sp_mel_warp_fbank_2d(m_mag, nbins_mel, alpha=alpha) # Don't delete
    else:
        m_mag_mel = la.sp_mel_warp(m_mag, nbins_mel, alpha=alpha, in_type=3)

    m_mag_mel_log =  la.log(m_mag_mel)

    # Phase feats to Mel-phase (compression):----------------------------------------------------------------
    # Note (Don't delete): For crsf_cf=5kHz, fft_len=4096, and fs=48kHz, bin_cf = 426
    crsf_cf, crsf_bw = define_crossfade_params(fs)
    if alpha_phase is None:
        alpha_phase = alpha

    nbins_mel_for_phase_comp = get_num_full_mel_coeffs_from_num_phase_coeffs(crsf_cf, nbins_phase, alpha_phase, fs)

    m_real_mel = la.sp_mel_warp(m_real, nbins_mel_for_phase_comp, alpha=alpha_phase, in_type=2)
    m_imag_mel = la.sp_mel_warp(m_imag, nbins_mel_for_phase_comp, alpha=alpha_phase, in_type=2)

    # Cutting phase vectors:
    m_real_mel = m_real_mel[:,:nbins_phase]
    m_imag_mel = m_imag_mel[:,:nbins_phase]

    # Removing phase in unvoiced frames ("random" values):
    m_real_mel = m_real_mel * v_voi[:,None]
    m_imag_mel = m_imag_mel * v_voi[:,None]

    # Clipping between -1 and 1:
    m_real_mel = np.clip(m_real_mel, -1, 1)
    m_imag_mel = np.clip(m_imag_mel, -1, 1)

    # -----------------------------------------------
    # Removing phase in unvoiced frames ("random" values):
    m_real_mel = m_real_mel * v_voi[:,None]
    m_imag_mel = m_imag_mel * v_voi[:,None]

    # Clipping between -1 and 1:
    m_real_mel = np.clip(m_real_mel, -1, 1)
    m_imag_mel = np.clip(m_imag_mel, -1, 1)

    return m_mag_mel_log, m_real_mel, m_imag_mel, v_lf0_smth


def format_for_modelling_phase_comp(m_mag, m_real, m_imag, v_f0, fs, nbins_mel=60, nbins_phase=10, b_mag_fbank_mel=False):
    '''
    format_for_modelling with phase compression based on filter bank. It didn't work very well according to experiments.

    b_fbank_mel: If True, Mel compression done by the filter bank approach. Otherwise, it uses sptk mcep related funcs.
    '''

    # alpha:
    alpha = define_alpha(fs)

    # f0 to Smoothed lf0:
    v_voi = (v_f0>0).astype('float')
    v_f0_smth  = v_voi * signal.medfilt(v_f0)
    v_lf0_smth = la.f0_to_lf0(v_f0_smth)

    # Mag to Log-Mag-Mel (compression):
    if b_mag_fbank_mel:

        # Debug:
        m_mag_mel = la.sp_mel_warp_fbank(m_mag, nbins_mel, alpha=alpha)
        #m_mag_mel = la.sp_mel_warp_fbank_2d(m_mag, nbins_mel, alpha=alpha)

    else:
        m_mag_mel = la.sp_mel_warp(m_mag, nbins_mel, alpha=alpha, in_type=3)

    m_mag_mel_log =  la.log(m_mag_mel)

    # Phase feats to Mel-phase (compression):
    crsf_cf, crsf_bw = define_crossfade_params(fs)
    fft_len_half = m_mag.shape[1]
    fft_len = 2 * (fft_len_half - 1)

    # Debug:-------------------------------------------------------------------------
    # Just only of one of these is used:
    # Note (Don't delete): For crsf_cf=5kHz, fft_len=4096, and fs=48kHz, bin_cf = 426
    bin_cf = lu.round_to_int(la.hz_to_bin(crsf_cf, fft_len, fs))
    #bin_l  = lu.round_to_int(la.hz_to_bin(crsf_cf - crsf_bw/2.0, fft_len, fs)) # Don't delete
    #bin_r  = lu.round_to_int(la.hz_to_bin(crsf_cf + crsf_bw/2.0, fft_len, fs)) # Don't delete

    max_bin_ph = bin_cf # bin_l # bin_cf # bin_r # bin_l

    v_bins_mel = la.build_mel_curve(alpha, fft_len_half)[:max_bin_ph]
    m_real_shrt = m_real[:,:max_bin_ph]
    m_imag_shrt = m_imag[:,:max_bin_ph]
    #--------------------------------------------------------------------------------
    m_real_mel = la.apply_fbank(m_real_shrt, v_bins_mel, nbins_phase)[0]
    m_imag_mel = la.apply_fbank(m_imag_shrt, v_bins_mel, nbins_phase)[0]


    # Debug (phase ratio):
    if False:
        nfrms = m_mag.shape[0]
        #m_ratio_shrt = np.arctan(np.abs(m_imag_shrt / m_real_shrt))
        m_ratio_shrt = np.arctan((m_imag_shrt / m_real_shrt))
        m_ratio_mel  = la.apply_fbank(m_ratio_shrt, v_bins_mel, nbins_phase)[0]
        m_ratio_shrt_rec = la.unwarp_from_fbank(m_ratio_mel, v_bins_mel, interp_kind='slinear')

        m_ratio_rec  = np.hstack((m_ratio_shrt_rec, m_ratio_shrt_rec[:,-1][:,None] + np.zeros((nfrms, fft_len_half-max_bin_ph))))

        m_ratio_mel_from_params = np.arctan(np.abs(m_imag_mel / m_real_mel))
        #m_ratio_from_params_shrt_rec = la.unwarp_from_fbank(m_real_mel, v_bins_mel, interp_kind='quadratic')

        m_real_shrt_rec = la.unwarp_from_fbank(m_real_mel, v_bins_mel, interp_kind='quadratic')
        m_imag_shrt_rec = la.unwarp_from_fbank(m_imag_mel, v_bins_mel, interp_kind='quadratic')


        m_real_rec  = np.hstack((m_real_shrt_rec, m_real_shrt_rec[:,-1][:,None] + np.zeros((nfrms, fft_len_half-max_bin_ph))))
        m_imag_rec  = np.hstack((m_imag_shrt_rec, m_imag_shrt_rec[:,-1][:,None] + np.zeros((nfrms, fft_len_half-max_bin_ph))))

        m_cmplx_ph_shrt =  np.angle(m_real_shrt[nx,:] + m_imag_shrt[nx,:] * 1j)

        # Plots:
        nx=111; figure(); plot(m_cmplx_ph_shrt) ; plot(np.abs(m_cmplx_ph_shrt)); grid()


        #-----------------------------------------------
        plm(m_ratio_shrt)
        nx=111; figure(); plot(m_real_shrt[nx,:],'.-'); plot(m_ratio_shrt[nx,:],'.-'); grid()

        nx=111; figure(); plot(m_real_mel[nx,:],'.-'); plot(m_ratio_mel[nx,:],'.-'); plot(m_ratio_mel_from_params[nx,:],'.-'); grid()







    # Debug (reconstruction):
    # Phase feats mel-unwarp:
    if False:
        nfrms = m_mag.shape[0]

        #bin_r   = lu.round_to_int(la.hz_to_bin(crsf_cf + crsf_bw/2.0, fft_len, fs))
        v_bins_mel  = la.build_mel_curve(alpha, fft_len_half)[:max_bin_ph]

        #m_real_shrt = la.unwarp_from_fbank(m_real_mel, v_bins_mel, interp_kind='slinear')
        #m_imag_shrt = la.unwarp_from_fbank(m_imag_mel, v_bins_mel, interp_kind='slinear')
        m_real_shrt_rec = la.unwarp_from_fbank(m_real_mel, v_bins_mel, interp_kind='quadratic')
        m_imag_shrt_rec = la.unwarp_from_fbank(m_imag_mel, v_bins_mel, interp_kind='quadratic')

        m_real_rec  = np.hstack((m_real_shrt_rec, m_real_shrt_rec[:,-1][:,None] + np.zeros((nfrms, fft_len_half-max_bin_ph))))
        m_imag_rec  = np.hstack((m_imag_shrt_rec, m_imag_shrt_rec[:,-1][:,None] + np.zeros((nfrms, fft_len_half-max_bin_ph))))



        nx=111;
        v_ratio =  m_imag[nx,:] / m_real[nx,:]
        v_fact  =  m_imag[nx,:] * m_real[nx,:]

        #v_ratio_2 =  (m_imag[nx,:] + 2.0) / (m_real[nx,:] + 2.0)
        m_ratio =  (m_imag) / (m_real)
        m_ratio_2 =  (m_imag + 2.0) / (m_real + 2.0)

        m_real_voi = m_real
        m_real_voi[~(v_voi).astype(bool),:] = 0.0
        m_real_td = np.fft.ifft(m_real_voi[:,:max_bin_ph]).real


        # Plots:
        nx=111; figure(); plot(m_real[nx,:]); plot(m_real_rec[nx,:]); grid()
        nx=111; figure(); plot(m_real[nx,:]); plot(m_imag[nx,:]); grid()
        #nx=111; figure(); plot(m_real[nx,:]); plot(m_imag[nx,:]); plot(v_ratio); plot(np.arctan(v_ratio)); grid()
        nx=111; figure(); plot(m_real[nx,:]); plot(m_imag[nx,:]); plot(v_ratio); plot(np.arctan(v_ratio)); grid()
        nx=111; figure(); plot(m_real[nx,:]); plot(m_imag[nx,:]); plot(v_ratio); plot(np.arctan(v_ratio)); plot(np.angle(m_real[nx,:] + m_imag[nx,:] * 1j))  ; grid()
        #nx=111; figure(); plot(m_real[nx,:]); plot(m_imag[nx,:]); plot(np.arctan(v_ratio)); plot(np.angle(m_real[nx,:] + m_imag[nx,:] * 1j))  ; grid()

        #nx=111; figure(); plot(m_real[nx,:], '.-'); plot(m_imag[nx,:], '.-'); plot(np.arctan(v_ratio), '.-'); plot(np.arctan(v_ratio_2), '.-'); grid()
        nx=111; figure(); plot(m_real[nx,:], '.-'); plot(m_imag[nx,:], '.-'); plot(np.arctan(m_ratio[nx,:]), '.-'); grid()

        nx=111; figure(); plot(m_real[nx,:]); plot(m_imag[nx,:]); plot(v_fact); grid()



        nx=73; figure(); plot(m_real[nx,:]); plot(m_real_rec[nx,:]); plot(m_real_rec_max[nx,:]); grid()
        nx=73; figure(); plot(m_imag[nx,:]); plot(m_imag_rec[nx,:]); plot(m_imag_rec_max[nx,:]); grid()
        nx=73; figure(); plot(m_ph[nx,:], '.-'); plot(np.angle(m_real_rec[nx,:] + m_imag_rec[nx,:] * 1j), '.-'); plot(np.angle(m_real_rec_max[nx,:] + m_imag_rec_max[nx,:] * 1j), '.-'); grid()

        plm(m_real)

    # Debug (reconstruction):
    # magnitude:
    if False:
        #nfrms = m_mag.shape[0]
        m_mag_log     = la.log(m_mag)
        m_mag_log_rec = la.sp_mel_unwarp(m_mag_mel_log, fft_len_half, alpha=alpha, in_type='log')

        plm(m_mag_log_rec)


        pl(m_mag_log[252:255,:].T)
        pl(m_mag_log_rec[252:255,:].T)

        figure(); plot(m_mag_log[252:255,:].T); plot(m_mag_log_rec[252:255,:].T); grid()

    # -----------------------------------------------
    #m_imag_mel = la.sp_mel_warp(m_imag, nbins_mel, alpha=alpha, in_type=2)
    #m_real_mel = la.sp_mel_warp(m_real, nbins_mel, alpha=alpha, in_type=2)

    # Cutting phase vectors:
    #m_real_mel = m_real_mel[:,:nbins_phase]
    #m_imag_mel = m_imag_mel[:,:nbins_phase]

    # Removing phase in unvoiced frames ("random" values):
    m_real_mel = m_real_mel * v_voi[:,None]
    m_imag_mel = m_imag_mel * v_voi[:,None]

    # Clipping between -1 and 1:
    m_real_mel = np.clip(m_real_mel, -1, 1)
    m_imag_mel = np.clip(m_imag_mel, -1, 1)

    return m_mag_mel_log, m_real_mel, m_imag_mel, v_lf0_smth


def format_for_modelling_old(m_mag, m_real, m_imag, v_f0, fs, nbins_mel=60, nbins_phase=45, b_fbank_mel=False):
    '''
    b_fbank_mel: If True, Mel compression done by the filter bank approach. Otherwise, it uses sptk mcep related funcs.
    '''

    # alpha:
    alpha = define_alpha(fs)

    # f0 to Smoothed lf0:
    v_voi = (v_f0>0).astype('float')
    v_f0_smth  = v_voi * signal.medfilt(v_f0)
    v_lf0_smth = la.f0_to_lf0(v_f0_smth)

    # Mag to Log-Mag-Mel (compression):
    if b_fbank_mel:
        m_mag_mel = la.sp_mel_warp_fbank(m_mag, nbins_mel, alpha=alpha)
    else:
        m_mag_mel = la.sp_mel_warp(m_mag, nbins_mel, alpha=alpha, in_type=3)

    m_mag_mel_log =  la.log(m_mag_mel)

    # Debug:-----------------
    #'''
    if False: # Debug
        m_mag_mel_debug, v_cntrs = sp_mel_warp(m_mag, nbins_mel, alpha=alpha)
        m_mag_mel_log_debug = np.log(m_mag_mel_debug)
        m_mag_log_rec_debug = sp_mel_unwarp(m_mag_mel_log_debug, v_cntrs, m_mag.shape[1])
        m_mag_log_rec = la.sp_mel_unwarp(m_mag_mel_log, m_mag.shape[1])

        err_norm = np.sqrt(np.sum((np.log(m_mag[:,:200]) - m_mag_log_rec[:,:200])**2))
        err_debug = np.sqrt(np.sum((np.log(m_mag[:,:200]) - m_mag_log_rec_debug[:,:200])**2))
    #'''

    if False:
        from libplot import lp
        plm(m_mag_log_rec)
        plm(m_mag_log_rec_debug)
        nx=171; lp.figure(); lp.plot(np.log(m_mag[nx,:])); lp.plot(m_mag_log_rec[nx,:]); lp.plot(m_mag_log_rec_debug[nx,:]); lp.grid()
        nx=20; lp.figure(); lp.plot(np.log(m_mag[nx,:])); lp.plot(m_mag_log_rec[nx,:]); lp.plot(m_mag_log_rec_debug[nx,:]); lp.grid()

    if False:
        from libplot import lp
        lp.plotm(m_mag_mel_log)
        lp.plotm(m_mag_mel_log_debug)
        lp.plotm(m_mag_mel_log - m_mag_mel_log_debug)
        nx=171; lp.figure(); lp.plot(m_mag[nx,:]); lp.plot(m_mag_mel_log[nx,:]); lp.plot(m_mag_mel_log_debug[nx,:]); lp.grid()

    # Phase feats to Mel-phase (compression):
    m_imag_mel = la.sp_mel_warp(m_imag, nbins_mel, alpha=alpha, in_type=2)
    m_real_mel = la.sp_mel_warp(m_real, nbins_mel, alpha=alpha, in_type=2)

    # Cutting phase vectors:
    # NOTE (Don't delete!): If nbins_phase=45, the approx mvb is 397 for fft_len=4096
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

def analysis_lossless_type2(wav_file, fft_len=None, out_dir=None):

    # Read file:
    v_sig, fs = sf.read(wav_file)

    # Epoch detection:
    est_file = lu.ins_pid('temp.est')
    la.reaper(wav_file, est_file)
    v_pm_sec, v_voi = la.read_reaper_est_file(est_file, check_len_smpls=len(v_sig), fs=fs)
    os.remove(est_file)
    v_pm_smpls = v_pm_sec * fs

    # Magnitude analysis:----------------------------------------------------------------
    v_nx_even = np.arange(0, v_pm_smpls.size, 2)
    v_nx_odd  = np.arange(1, v_pm_smpls.size, 2)
    m_fft_even, v_shift_even = analysis_with_del_comp_from_pm(v_sig, fs, v_pm_smpls[v_nx_even], fft_len=fft_len)
    m_fft_odd , v_shift_odd  = analysis_with_del_comp_from_pm(v_sig, fs, v_pm_smpls[v_nx_odd] , fft_len=fft_len)

    nfrms     = m_fft_even.shape[0] + m_fft_odd.shape[0]
    nfft_half = m_fft_even.shape[1]
    m_fft     = np.zeros((nfrms, nfft_half), dtype=np.complex)
    m_fft[v_nx_even,:] = m_fft_even
    m_fft[v_nx_odd,:]  = m_fft_odd
    m_fft = m_fft[1:,:]

    v_shift = la.pm_to_shift(v_pm_smpls[1:])
    #v_shift = v_shift[1:]

    # Debug:
    if False:
        from libplot import lp
        lp.plotm(la.db(np.absolute(m_fft)))

    # Getting high-ress magphase feats:
    m_mag_long, m_real_long, m_imag_long, v_f0_long = compute_lossless_feats(m_fft, v_shift, v_voi[1:], fs)
    #m_mag_long = np.absolute(m_fft)

    # True envelope:
    m_mag_env = la.true_envelope(m_mag_long, in_type='abs', ncoeffs=600, thres_db=0.1)

    # Phase analysis:------------------------------------------------------------------------
    m_fft_phase, v_shift_phase, v_gain = analysis_with_del_comp_from_pm_type2(v_sig, fs, v_pm_smpls, v_voi, fft_len=fft_len)
    m_mag, m_real, m_imag, v_f0 = compute_lossless_feats(m_fft_phase, v_shift_phase, v_voi, fs)
    m_real = m_real[1:]
    m_imag = m_imag[1:]
    v_f0   = v_f0[1:]
    v_gain = v_gain[1:]

    if False: # phase
        from libplot import lp
        nx=202; lp.figure(); lp.plot(m_real[nx,:]); lp.plot(m_real_long[nx,:]); lp.grid()

    if False:
        from libplot import lp
        m_mag_log = np.log(m_mag_long)
        m_mag_env_log = np.log(m_mag_env)
        lp.plotm(m_mag_log)
        lp.plotm(m_mag_env_log)
        lp.plotm(np.log(m_mag))
        lp.plotm(m_real)
        nx=201; lp.figure(); lp.plot(m_mag_log[nx,:]); lp.plot(m_mag_env_log[nx,:]); lp.plot(0.0 + np.log(m_mag[nx,:])); lp.grid()
        nx=101; lp.figure(); lp.plot(m_real[nx,:]); lp.grid()

    # If output directory provided, features are written to disk:
    if type(out_dir) is str:
        file_id = os.path.basename(wav_file).split(".")[0]
        write_featfile(m_mag_env, out_dir, file_id + '.mag')
        write_featfile(m_real , out_dir, file_id + '.real')
        write_featfile(m_imag , out_dir, file_id + '.imag')
        write_featfile(v_f0   , out_dir, file_id + '.f0')
        write_featfile(v_shift, out_dir, file_id + '.shift')
        return

    return m_mag_env, m_real, m_imag, v_f0, fs, v_shift, v_gain


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

    # Debug:
    if False:
        from libplot import lp
        lp.plotm(np.absolute(m_fft))

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

def analysis_compressed_type1(wav_file, fft_len=None, out_dir=None, nbins_mel=60, nbins_phase=45, const_rate_ms=-1.0):

    # Analysis:
    m_mag, m_real, m_imag, v_f0, fs, v_shift = analysis_lossless(wav_file, fft_len=fft_len)

    # To constant rate:
    if const_rate_ms>0.0:
        interp_type = 'linear' #  'quadratic' # 'linear'
        v_pm_smpls = la.shift_to_pm(v_shift)
        m_mag  = interp_from_variable_to_const_frm_rate(m_mag,  v_pm_smpls, const_rate_ms, fs, interp_type=interp_type)
        m_real = interp_from_variable_to_const_frm_rate(m_real, v_pm_smpls, const_rate_ms, fs, interp_type=interp_type)
        m_imag = interp_from_variable_to_const_frm_rate(m_imag, v_pm_smpls, const_rate_ms, fs, interp_type=interp_type)

        # f0:
        v_voi = v_f0>1.0
        v_f0  = interp_from_variable_to_const_frm_rate(np.r_[ v_f0[v_voi][0],v_f0[v_voi], v_f0[v_voi][-1] ], np.r_[ 0, v_pm_smpls[v_voi], v_pm_smpls[-1] ], const_rate_ms, fs, interp_type=interp_type).squeeze()
        v_voi = interp_from_variable_to_const_frm_rate(v_voi, v_pm_smpls, const_rate_ms, fs, interp_type=interp_type)>0.5
        v_f0  *= v_voi # Double check this. At the beginning of voiced segments.

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
        if const_rate_ms<=0.0: # If variable rate, save shift files.
            write_featfile(v_shift  , out_dir, file_id + '.shift')
        return

    return m_mag_mel_log, m_real_mel, m_imag_mel, v_lf0_smth, v_shift, fs, fft_len




def analysis_compressed(wav_file, fft_len=None, nbins_mel=60, nbins_phase=10,
                                            b_const_rate=False, b_mag_fbank_mel=False, alpha_phase=None):
    '''
    Analyses a wavefile and extract compressed features for acoustic modelling.

    Params:
    wav_file:     Waveform to be analysed.
    fft_len:      FFT length. If None, its value is set according to the sample rate.
    nbins_mel:    Number of coefficents (bins) for the Log Magnitude feature (mag).
    nbins_phase:  Number of coefficents (bins) for the phase features (real and imag).
    b_const_rate: If False, output given in variable-frame rate fashion (pitch synchronous) [Default]
                  If True, output given in 5ms constant frame rate shift.

    b_mag_fbank_mel, alpha_phase: Experimental.
    '''

    # Analysis lossless:
    m_mag, m_real, m_imag, v_f0, fs, v_shift = analysis_lossless(wav_file, fft_len=fft_len)

    # To constant rate:
    if b_const_rate:
        const_rate_ms = 5.0
        interp_type = 'linear' #  'quadratic' # 'linear'
        v_pm_smpls = la.shift_to_pm(v_shift)
        m_mag  = interp_from_variable_to_const_frm_rate(m_mag,  v_pm_smpls, const_rate_ms, fs, interp_type=interp_type)
        m_real = interp_from_variable_to_const_frm_rate(m_real, v_pm_smpls, const_rate_ms, fs, interp_type=interp_type)
        m_imag = interp_from_variable_to_const_frm_rate(m_imag, v_pm_smpls, const_rate_ms, fs, interp_type=interp_type)

        # f0:
        v_voi = v_f0>1.0
        v_f0  = interp_from_variable_to_const_frm_rate(np.r_[ v_f0[v_voi][0],v_f0[v_voi], v_f0[v_voi][-1] ],
                           np.r_[ 0, v_pm_smpls[v_voi], v_pm_smpls[-1] ], const_rate_ms, fs, interp_type=interp_type).squeeze()
        v_voi = interp_from_variable_to_const_frm_rate(v_voi, v_pm_smpls, const_rate_ms, fs, interp_type=interp_type)>0.5
        v_f0  *= v_voi # Double check this. At the beginning of voiced segments.

    # Formatting for Acoustic Modelling:
    m_mag_mel_log, m_real_mel, m_imag_mel, v_lf0_smth = format_for_modelling(m_mag, m_real, m_imag, v_f0, fs, nbins_mel=nbins_mel,
                                                                                    nbins_phase=nbins_phase, alpha_phase=alpha_phase)

    fft_len = 2*(np.size(m_mag,1) - 1)

    return m_mag_mel_log, m_real_mel, m_imag_mel, v_lf0_smth, v_shift, fs, fft_len



def analysis_for_acoustic_modelling(wav_file, out_dir, fft_len=None, nbins_mel=60, nbins_phase=10,
                                            b_const_rate=False, b_mag_fbank_mel=False, alpha_phase=None):
    '''
    Analyses a wavefile and extract compressed features for acoustic modelling.

    Params:
    wav_file:     Waveform to be analysed.
    fft_len:      FFT length. If None, its value is set according to the sample rate.
    out_dir:      Directory where MagPhase features will be stored.
    nbins_mel:    Number of coefficents (bins) for the Log Magnitude feature (mag).
    nbins_phase:  Number of coefficents (bins) for the phase features (real and imag).
    b_const_rate: If False, output given in variable-frame rate fashion (pitch synchronous) [Default]
                  If True, output given in 5ms constant frame rate shift.

    b_mag_fbank_mel, alpha_phase: Experimental.
    '''

    m_mag_mel_log, m_real_mel, m_imag_mel, v_lf0_smth, v_shift, fs, fft_len = analysis_compressed(wav_file, fft_len=fft_len, nbins_mel=nbins_mel, nbins_phase=nbins_phase,
                                                                                        b_const_rate=b_const_rate, b_mag_fbank_mel=b_mag_fbank_mel, alpha_phase=b_mag_fbank_mel)


    # Save features:
    file_id = os.path.basename(wav_file).split(".")[0]
    write_featfile(m_mag_mel_log, out_dir, file_id + '.mag')
    write_featfile(m_real_mel   , out_dir, file_id + '.real')
    write_featfile(m_imag_mel   , out_dir, file_id + '.imag')
    write_featfile(v_lf0_smth   , out_dir, file_id + '.lf0')
    if not b_const_rate: # If variable rate, save shift files.
        write_featfile(v_shift , out_dir, file_id + '.shift')

    return

def analysis_compressed_type1_with_phase_comp(wav_file, fft_len=None, out_dir=None,
                                                    nbins_mel=60, nbins_phase=10, b_const_rate=False, b_mag_fbank_mel=False):

    '''
    analysis_compressed_type1 with phase compression based on filter bank. It didn't work very well according to experiments.

    '''

    # Analysis:
    m_mag, m_real, m_imag, v_f0, fs, v_shift = analysis_lossless(wav_file, fft_len=fft_len)

    # Debug: smoothing. interesante resultado. NO BORRAR!!
    #m_mag = la.smooth_by_conv(m_mag, v_win=np.r_[0.3, 0.4, 0.3])
    #m_mag = np.exp(la.smooth_by_conv(la.log(m_mag), v_win=np.ones(3)))

    # To constant rate:
    if b_const_rate:
        const_rate_ms = 5.0
        interp_type = 'linear' #  'quadratic' # 'linear'
        v_pm_smpls = la.shift_to_pm(v_shift)
        m_mag  = interp_from_variable_to_const_frm_rate(m_mag,  v_pm_smpls, const_rate_ms, fs, interp_type=interp_type)
        m_real = interp_from_variable_to_const_frm_rate(m_real, v_pm_smpls, const_rate_ms, fs, interp_type=interp_type)
        m_imag = interp_from_variable_to_const_frm_rate(m_imag, v_pm_smpls, const_rate_ms, fs, interp_type=interp_type)

        # f0:
        v_voi = v_f0>1.0
        v_f0  = interp_from_variable_to_const_frm_rate(np.r_[ v_f0[v_voi][0],v_f0[v_voi], v_f0[v_voi][-1] ], np.r_[ 0, v_pm_smpls[v_voi], v_pm_smpls[-1] ], const_rate_ms, fs, interp_type=interp_type).squeeze()
        v_voi = interp_from_variable_to_const_frm_rate(v_voi, v_pm_smpls, const_rate_ms, fs, interp_type=interp_type)>0.5
        v_f0  *= v_voi # Double check this. At the beginning of voiced segments.

    # Debug:-----------------
    #m_real = np.zeros(m_real.shape)
    #m_real[320, 395] = 1.0

    # Formatting for Acoustic Modelling:
    m_mag_mel_log, m_real_mel, m_imag_mel, v_lf0_smth = format_for_modelling_phase_comp(m_mag, m_real, m_imag, v_f0, fs, nbins_mel=nbins_mel, nbins_phase=nbins_phase)


    # Debug: Reconstruction
    # m_mag_mel_log_orig, m_real_mel_orig, m_imag_mel_orig, v_lf0_smth_orig = format_for_modelling(m_mag, m_real, m_imag, v_f0, fs, nbins_mel=nbins_mel, nbins_phase=nbins_phase)
    # crsf_cf, crsf_bw = define_crossfade_params(fs)
    # alpha = define_alpha(fs)
    # fft_len = define_fft_len(fs)
    # m_real_rec, m_imag_rec = phase_uncompress_fbank(m_real_mel, m_imag_mel, crsf_cf, crsf_bw, alpha, fft_len, fs)

    # m_real_rec_orig, m_imag_rec_orig = phase_uncompress_type1(m_real_mel_orig, m_imag_mel_orig, alpha, fft_len, nbins_mel)

    if False:
        plm(m_real_mel_orig)
        plm(m_real_mel)
        nx=320; figure(); plot(m_real_mel_orig[nx,:]); plot(m_real_mel[nx,:]); grid()
        nx=320; figure(); plot(m_real_rec_orig[nx,:]); plot(m_real_rec[nx,:]); grid()

        nx=320; figure(); plot(m_real[nx,:]); plot(m_real_rec_orig[nx,:]); plot(m_real_rec[nx,:]); grid()

    fft_len = 2*(np.size(m_mag,1) - 1)

    # Save features:
    if type(out_dir) is str:
        file_id = os.path.basename(wav_file).split(".")[0]
        write_featfile(m_mag_mel_log, out_dir, file_id + '.mag')
        write_featfile(m_real_mel   , out_dir, file_id + '.real')
        write_featfile(m_imag_mel   , out_dir, file_id + '.imag')
        write_featfile(v_lf0_smth   , out_dir, file_id + '.lf0')
        if not b_const_rate: # If variable rate, save shift files.
            write_featfile(v_shift , out_dir, file_id + '.shift')
        return

    return m_mag_mel_log, m_real_mel, m_imag_mel, v_lf0_smth, v_shift, fs, fft_len

def compute_imag_from_real(start_sign, v_real):
    '''
    NOT FINISHED (ongoing work!!)
    '''

    nbins = v_real.size
    v_imag = np.zeros(nbins) # v_imag[0] = 0 always
    v_imag[1] = start_sign * np.sqrt(1.0 - (v_real[1]**2))

    #curr_sign = start_sign
    for nxb in xrange(2,nbins):
        prev_diff = v_imag[nxb-1] - v_imag[nxb-2]

        curr_val_pos =  np.sqrt(1.0 - (v_real[nxb]**2))
        curr_val_neg = -curr_val_pos

        curr_diff_pos = curr_val_pos - v_imag[nxb-1]
        curr_diff_neg = curr_val_neg - v_imag[nxb-1]

        if np.abs(curr_diff_pos-prev_diff)<=np.abs(curr_diff_neg-prev_diff):
            v_imag[nxb] = curr_val_pos
        else:
            v_imag[nxb] = curr_val_neg

        #v_imag[nxb] = curr_sign * curr_val

    return v_imag


def analysis_compressed_type2(wav_file, fft_len=None, out_dir=None, nbins_mel=60, nbins_phase=45, b_norm_mag=False, const_rate_ms=-1.0):

    # Analysis:
    m_mag, m_real, m_imag, v_f0, fs, v_shift, v_gain = analysis_lossless_type2(wav_file, fft_len=fft_len)

    # To constant rate:
    if const_rate_ms>0.0:
        v_pm_smpls = la.shift_to_pm(v_shift)
        m_mag  = interp_from_variable_to_const_frm_rate(m_mag,  v_pm_smpls, const_rate_ms, fs, interp_type='linear')
        m_real = interp_from_variable_to_const_frm_rate(m_real, v_pm_smpls, const_rate_ms, fs, interp_type='linear')
        m_imag = interp_from_variable_to_const_frm_rate(m_imag, v_pm_smpls, const_rate_ms, fs, interp_type='linear')
        #g = interp_from_variable_to_const_frm_rate(v_gain, v_pm_smpls, const_rate_ms, fs, interp_type='linear')
        v_gain = interp_from_variable_to_const_frm_rate(v_gain, v_pm_smpls, const_rate_ms, fs, interp_type='linear')

        # f0:
        v_voi = v_f0>1.0
        v_f0  = interp_from_variable_to_const_frm_rate(np.r_[ v_f0[v_voi][0],v_f0[v_voi], v_f0[v_voi][-1] ], np.r_[ 0, v_pm_smpls[v_voi], v_pm_smpls[-1] ], const_rate_ms, fs, interp_type='linear').squeeze()
        v_voi = interp_from_variable_to_const_frm_rate(v_voi, v_pm_smpls, const_rate_ms, fs, interp_type='linear')>0.5
        v_f0  *= v_voi # Double check this. At the beginning of voiced segments.



    # Debug:
    # ONGOING WORK (NOT FINISHED)
    #nx=0
    #start_sign = -1.0
    #v_imag_ext = compute_imag_from_real(start_sign, m_real[nx,:])
    #m_imag_ext = start_sign * np.sqrt(1.0 - (m_real**2))

    if False:

        from libplot import lp
        nx=0; lp.figure(); lp.plot(m_real[nx,:], '.-'); lp.plot(m_imag[nx,:], '.-'); lp.plot(v_imag_ext, '.-'); lp.grid()
        nx=0; lp.figure(); lp.plot(m_real[nx,:], '.-'); lp.plot(m_imag[nx,:], '.-'); lp.plot(m_imag_ext[nx,:], '.-'); lp.grid()
        nx=0; lp.figure(); lp.plot(np.diff(m_imag[nx,:]), '.-'); lp.plot(np.diff(m_imag_ext[nx,:]), '.-'); lp.grid()
        nx=0; lp.figure(); lp.plot(np.diff(np.diff(m_imag[nx,:])), '.-'); lp.plot(np.diff(np.diff(m_imag_ext[nx,:])), '.-'); lp.grid()
        nx=0; lp.figure(); lp.plot(m_real[nx,:]**2+m_imag[nx,:]**2, '.-'); lp.grid()

    # Norm:
    #m_mag_log = la.log(m_mag)
    #m_mag_log_norm = (m_mag_log.T - np.mean(m_mag_log, axis=1)).T

    if False:
        from libplot import lp
        lp.plotm(m_mag_log)
        lp.plotm(m_mag_log_norm)
        nx=312; lp.figure(); lp.plot(m_mag_log[nx:nx+3,:].T); lp.grid()
        nx=312; lp.figure(); lp.plot(m_mag_log_norm[nx:nx+3,:].T); lp.grid()

    # Formatting for Acoustic Modelling:
    m_mag_mel_log, m_real_mel, m_imag_mel, v_lf0_smth = format_for_modelling(m_mag, m_real, m_imag, v_f0, fs, nbins_mel=nbins_mel, nbins_phase=nbins_phase)
    fft_len = 2*(np.size(m_mag,1) - 1)
    v_lgain = la.log(v_gain)

    if b_norm_mag:
        v_mean = np.mean(m_mag_mel_log[:,1:], axis=1)
        m_mag_mel_log = (m_mag_mel_log.T - v_mean).T
        v_lgain  = v_mean
        m_mag_mel_log[:,0] = v_lgain

    # Save features:
    if type(out_dir) is str:
        file_id = os.path.basename(wav_file).split(".")[0]
        write_featfile(m_mag_mel_log, out_dir, file_id + '.mag')
        write_featfile(m_real_mel   , out_dir, file_id + '.real')
        write_featfile(m_imag_mel   , out_dir, file_id + '.imag')
        write_featfile(v_lf0_smth   , out_dir, file_id + '.lf0')

        if const_rate_ms<=0.0:
            write_featfile(v_shift  , out_dir, file_id + '.shift')

        return

    return m_mag_mel_log, m_real_mel, m_imag_mel, v_lf0_smth, v_shift, fs, fft_len, v_lgain


def synthesis_from_acoustic_modelling_old(in_feats_dir, filename_token, out_syn_dir, nbins_mel, nbins_phase, fs, fft_len=None, pf_type='no', magphase_type='type1', b_const_rate=False):
    '''
    pf_type: Postfilter type: 'merlin' (Merlin's style), 'magphase' (MagPhase's own postfilter (in development)), or 'no'.
    '''

    #if pf_type=='merlin':
    #    post_filter_merlin()
    # Display:
    print("\nSynthesising file: " + filename_token + '.wav............................')

    # Reading parameter files:
    m_mag_mel_log = lu.read_binfile(in_feats_dir + '/' + filename_token + '.mag' , dim=nbins_mel)
    m_real_mel    = lu.read_binfile(in_feats_dir + '/' + filename_token + '.real', dim=nbins_phase)
    m_imag_mel    = lu.read_binfile(in_feats_dir + '/' + filename_token + '.imag', dim=nbins_phase)
    v_lf0         = lu.read_binfile(in_feats_dir + '/' + filename_token + '.lf0' , dim=1)

    if pf_type=='magphase':
        print('Using MagPhase postfilter!')
        m_mag_mel_log = post_filter(m_mag_mel_log, fs)


    # Waveform generation:
    if magphase_type=='type1':
        v_syn_sig = synthesis_from_compressed_type1(m_mag_mel_log, m_real_mel, m_imag_mel, v_lf0, fs, fft_len=fft_len, b_const_rate=b_const_rate)
    elif magphase_type=='type2':
        v_syn_sig = synthesis_from_compressed_type2(m_mag_mel_log, m_real_mel, m_imag_mel, v_lf0, fs, fft_len=fft_len, const_rate_ms=5)

    la.write_audio_file(out_syn_dir + '/' + filename_token + '.wav', v_syn_sig, fs)
    return

def synthesis_from_acoustic_modelling(in_feats_dir, filename_token, out_syn_dir, nbins_mel, nbins_phase, fs,
                                            fft_len=None, pf_type='no', b_const_rate=False):
    '''
    Synthesises a waveform from compressed MagPhase features.

    Params:
    in_feats_dir:   Directory containing the MagPhase features .mag, .real, .imag, and .lf0
    filename_token: Name of the utterace. E.g., "arctic_a0001"
    out_syn_dir:    Directory where the synthesised waveform will be stored.
    nbins_mel:      Number of coefficents (bins) for the Log Magnitude feature (mag).
    nbins_phase:    Number of coefficents (bins) for the phase features (real and imag).
                    Typical values: 45, 20, 10.
    fs:             Sample rate.
    fft_len:        FFT length. If None, its value is set according to the sample rate.
    pf_type:        "magphase":MagPhase's own postfilter (in development)
                    "merlin":  Merlin's style postfilter.
                    "no":      No postfilter.
    b_const_rate:   If False, variable-frame rate input features (pitch synchronous) [Default]
                    If True,   5ms constant frame rate input features.
    '''

    # Display:
    print("\nSynthesising file: " + filename_token + '.wav............................')

    # Reading parameter files:
    m_mag_mel_log = lu.read_binfile(in_feats_dir + '/' + filename_token + '.mag' , dim=nbins_mel)
    m_real_mel    = lu.read_binfile(in_feats_dir + '/' + filename_token + '.real', dim=nbins_phase)
    m_imag_mel    = lu.read_binfile(in_feats_dir + '/' + filename_token + '.imag', dim=nbins_phase)
    v_lf0         = lu.read_binfile(in_feats_dir + '/' + filename_token + '.lf0' , dim=1)

    if pf_type=='magphase':
        print('Using MagPhase postfilter...')
        m_mag_mel_log = post_filter(m_mag_mel_log, fs)

    elif pf_type=='merlin':
        print('Using Merlin postfilter...')
        m_mag_mel_log = post_filter_merlin(m_mag_mel_log, fs)

    elif pf_type=='no':
        print('No postfilter...')

    # Waveform generation:
    v_syn_sig = synthesis_from_compressed(m_mag_mel_log, m_real_mel, m_imag_mel, v_lf0,
                                                fs, fft_len=fft_len, b_const_rate=b_const_rate)

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
        crsf_cf = 2500 #4500 #3000 # TODO: tune these values.
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



def griffin_lim(m_mag, v_shift, win_func=np.hanning, phase_init='random', niters=30):
    '''
    Pitch synchronous Griffin-Lim algorithm.
    phase_init: 'random' (random phase), 'linear' (linear phase), 'min_phase' (minimum phase), or numpy 2D array (matrix) containing initial phase values.
    '''

    print('Starting Griffin-Lim. It could take a while...')

    v_shift = lu.round_to_int(v_shift)
    nfrms, fft_len_half = m_mag.shape
    fft_len = 2 * (fft_len_half - 1)

    # Initial phase set up:
    if type(phase_init)==str:

        if phase_init=='random':
            m_phase = 2 * np.pi * (np.random.rand(nfrms, fft_len) - 0.5)

        elif phase_init=='linear':
            m_frms_zero = np.zeros((nfrms, fft_len))
            m_frms_zero[:,(fft_len/2)] = 1.0
            m_phase = np.angle(np.fft.fft(m_frms_zero))

        elif phase_init=='min_phase':
            m_mag_cmplx_min_ph = la.build_min_phase_from_mag_spec(m_mag)
            m_phase = np.angle(m_mag_cmplx_min_ph)
            m_phase = la.add_hermitian_half(m_phase, data_type='phase')

    elif type(phase_init)==np.ndarray:
        m_phase = la.add_hermitian_half(phase_init, data_type='phase')

    # protection for indexes 0 and fft_len_half?

    m_mag = la.add_hermitian_half(m_mag)
    v_pm = la.shift_to_pm(v_shift)
    for nxi in xrange(niters):

        # Synthesis:
        m_cmplx_sp = m_mag * np.exp(m_phase * 1j)
        m_frms     = np.fft.ifft(m_cmplx_sp).real
        v_sig = ola(m_frms, v_pm, win_func=None)

        if nxi==(niters-1):
            break

        # Analysis:
        l_frms, v_lens, v_pm_plus, v_shift_dummy, v_rights = windowing(v_sig, v_pm, win_func=win_func)
        m_frms = la.frm_list_to_matrix(l_frms, v_shift, fft_len)
        m_cmplx_sp = np.fft.fft(m_frms, n=fft_len)

        # Update:
        m_phase = np.angle(m_cmplx_sp)

    return v_sig, la.remove_hermitian_half(m_phase)

def post_filter_merlin(m_mag_mel_log, fs, pf_coef=1.4):

    '''
    TODO: Add note about Merlin copyright
    '''

    # Constants:
    fft_len   = 4096
    minph_ord = fft_len / 2 - 1  # minimum phase order (2047)
    alpha = define_alpha(fs)

    # Temp files setup:
    temp_mcep    = lu.ins_pid('temp.mcep')
    temp_mcep_pf = lu.ins_pid('temp.mcep_pf')
    temp_lifter  = lu.ins_pid('temp.lift')
    temp_r0      = lu.ins_pid('temp.r0')
    temp_p_r0    = lu.ins_pid('temp.p_r0')
    temp_b0      = lu.ins_pid('temp.b0')
    temp_p_b0    = lu.ins_pid('temp.p_b0')


    # Save m_mag_mel_log into mcep:
    m_mcep = la.rceps(m_mag_mel_log, in_type='log', out_type='compact')
    lu.write_binfile(m_mcep, temp_mcep)

    ncoeffs = m_mag_mel_log.shape[1]

    # Building lifter:
    lifter = "echo 1 1 " + ("%1.2f " % pf_coef) * (ncoeffs-2)

    # SPTK binaries:
    x2x_bin   = os.path.join(la._sptk_dir, 'x2x')
    freqt_bin = os.path.join(la._sptk_dir, 'freqt')
    c2acr_bin = os.path.join(la._sptk_dir, 'c2acr')
    vopr_bin  = os.path.join(la._sptk_dir, 'vopr')
    mc2b_bin  = os.path.join(la._sptk_dir, 'mc2b')
    bcp_bin   = os.path.join(la._sptk_dir, 'bcp')
    sopr_bin  = os.path.join(la._sptk_dir, 'sopr')
    merge_bin = os.path.join(la._sptk_dir, 'merge')
    b2mc_bin  = os.path.join(la._sptk_dir, 'b2mc')

    # Saving lifter in file:
    curr_cmd = '{lifter} | {x2x} +af > {weight}'.format(lifter=lifter, x2x=x2x_bin, weight=temp_lifter)
    call(curr_cmd, shell=True)
    #--------------------------------------
    # Saving Base r0:
    curr_cmd = '{freqt} -m {order} -a {fw} -M {co} -A 0 < {mgc} | {c2acr} -m {co} -M 0 -l {fl} > {base_r0}'.\
                    format(freqt=freqt_bin, order=ncoeffs-1, fw=alpha, co=minph_ord, mgc=temp_mcep, c2acr=c2acr_bin, fl=fft_len, base_r0=temp_r0)
    call(curr_cmd, shell=True)
    #--------------------------------------
    curr_cmd = '{vopr} -m -n {order} < {mgc} {weight} | {freqt} -m {order} -a {fw} -M {co} -A 0 | {c2acr} -m {co} -M 0 -l {fl} > {base_p_r0}'\
                                .format(vopr=vopr_bin, order=ncoeffs-1, mgc=temp_mcep, weight=temp_lifter, freqt=freqt_bin, fw=alpha, co=minph_ord,
                                                                                                    c2acr=c2acr_bin, fl=fft_len, base_p_r0=temp_p_r0)
    call(curr_cmd, shell=True)

    #--------------------------------------
    curr_cmd = '{vopr} -m -n {order} < {mgc} {weight} | {mc2b} -m {order} -a {fw} | {bcp} -n {order} -s 0 -e 0 > {base_b0}'\
                    .format(vopr=vopr_bin, order=ncoeffs-1, mgc=temp_mcep, weight=temp_lifter, mc2b=mc2b_bin, fw=alpha, bcp=bcp_bin, base_b0=temp_b0)
    call(curr_cmd, shell=True)

    #--------------------------------------
    curr_cmd = '{vopr} -d < {base_r0} {base_p_r0} | {sopr} -LN -d 2 | {vopr} -a {base_b0} > {base_p_b0}'\
                .format(vopr=vopr_bin, base_r0=temp_r0, base_p_r0=temp_p_r0, sopr=sopr_bin, base_b0=temp_b0, base_p_b0=temp_p_b0)
    call(curr_cmd, shell=True)
    #--------------------------------------

    curr_cmd = '{vopr} -m -n {order} < {mgc} {weight} | {mc2b} -m {order} -a {fw} | {bcp} -n {order} -s 1 -e {order} | {merge} -n {order2} -s 0 -N 0 {base_p_b0} | {b2mc} -m {order} -a {fw} > {base_p_mgc}'\
                            .format(vopr=vopr_bin, order=ncoeffs-1, mgc=temp_mcep, weight=temp_lifter, mc2b=mc2b_bin, fw=alpha, bcp=bcp_bin,
                                                    merge=merge_bin, order2=ncoeffs-2, base_p_b0=temp_p_b0, b2mc=b2mc_bin, base_p_mgc=temp_mcep_pf)
    call(curr_cmd, shell=True)

    #--------------------------------------

    # Convert to mel_mag_log:
    m_mcep_pf = lu.read_binfile(temp_mcep_pf, dim=ncoeffs)
    m_mag_mel_log_pf = la.mcep_to_sp_cosmat(m_mcep_pf, ncoeffs, alpha=0.0, out_type='log')

    # Protection agains possible nans:
    m_mag_mel_log_pf[np.isnan(m_mag_mel_log_pf)] = la.MAGIC

    # Temp files removal:
    os.remove(temp_mcep)
    os.remove(temp_mcep_pf)
    os.remove(temp_lifter)
    os.remove(temp_r0)
    os.remove(temp_p_r0)
    os.remove(temp_b0)
    os.remove(temp_p_b0)


    return m_mag_mel_log_pf
