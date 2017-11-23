# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 02:48:28 2015

My personal library for general audio processing.

@author: Felipe Espic
"""
import numpy as np
import os
from subprocess import call
#import warnings
import soundfile as sf
import libutils as lu
from scipy import interpolate
from ConfigParser import SafeConfigParser

# Configuration:
#_curr_dir = os.path.dirname(os.path.realpath(__file__))
#_reaper_bin    = os.path.realpath(_curr_dir + '/../tools/REAPER/build/reaper')
#_sptk_mcep_bin = os.path.realpath(_curr_dir + '/../tools/SPTK-3.9/build/bin/mcep')

MAGIC = -1.0E+10 # logarithm floor (the same as SPTK)

#-------------------------------------------------------------------------------
def parse_config():
    global _reaper_bin, _sptk_mcep_bin
    _curr_dir = os.path.dirname(os.path.realpath(__file__))
    _reaper_bin    = os.path.realpath(_curr_dir + '/../tools/REAPER/build/reaper')
    _sptk_mcep_bin = os.path.realpath(_curr_dir + '/../tools/SPTK-3.9/build/bin/mcep')
    _config = SafeConfigParser()
    _config.read(_curr_dir + '/../config.ini')
    #import ipdb; ipdb.set_trace()
    if not ((_config.get('TOOLS', 'reaper')=='') or (_config.get('TOOLS', 'sptk_mcep')=='')):
        #import ipdb; ipdb.set_trace()
        _reaper_bin    = _config.get('TOOLS', 'reaper')
        _sptk_mcep_bin = _config.get('TOOLS', 'sptk_mcep')
    return
parse_config()

#-------------------------------------------------------------------------------
def gen_mask_simple(v_voi, nbins, cutoff_bin):
    '''
    Basically: 1=deterministic, 0=stochastic
    '''
    m_mask  = np.tile(v_voi, [nbins,1]).T
    m_mask[:,cutoff_bin:] = 0

    return m_mask

#------------------------------------------------------------------------------

def mix_by_mask(m_data_a, m_data_b, m_mask):
    '''
    Basically, in the mask: 1=deterministic, 0=stochastic
    Also: 1=m_data_a, 0=m_data_b
    '''
    m_data = m_mask * m_data_a + (1 - m_mask) * m_data_b

    return m_data

#------------------------------------------------------------------------------
def shift_to_pm(v_shift):
    v_pm = np.cumsum(v_shift)
    return v_pm

#------------------------------------------------------------------------------
def pm_to_shift(v_pm):
    v_shift = np.diff(np.hstack((0,v_pm)))
    return v_shift

#------------------------------------------------------------------------------
def gen_non_symmetric_win(left_len, right_len, win_func):
    # Left window:
    v_left_win = win_func(1+2*left_len)
    v_left_win = v_left_win[0:(left_len+1)]
    
    # Right window:
    v_right_win = win_func(1+2*right_len)
    v_right_win = np.flipud(v_right_win[0:(right_len+1)])
    
    # Constructing window:
    return np.hstack((v_left_win, v_right_win[1:]))    
    
#------------------------------------------------------------------------------
# generated centered assymetric window:
# If totlen is even, it is assumed that the center is the first element of the second half of the vector.
# TODO: case win_func == None
def gen_centr_win(winlen_l, winlen_r, totlen, win_func=None):
   
    v_win_shrt = gen_non_symmetric_win(winlen_l, winlen_r, win_func)  
    win_shrt_len = len(v_win_shrt)
    
    nx_cntr  = np.floor(totlen / 2.0).astype(int)
    nzeros_l = nx_cntr - winlen_l    
    
    v_win = np.zeros(totlen)
    v_win[nzeros_l:nzeros_l+win_shrt_len] = v_win_shrt
    return v_win

#------------------------------------------------------------------------------
def ola(m_frm, shift):
    shift = int(shift)
    nfrms, frmlen = m_frm.shape

    sig_len = (nfrms - 1) * shift + frmlen
    v_sig   = np.zeros(sig_len)
    strt    = 0
    for nxf in xrange(nfrms):

        # Add frames:
        v_sig[strt:(strt+frmlen)] += m_frm[nxf,:]
        strt += shift

    return v_sig

#------------------------------------------------------------------------------
def frm_list_to_matrix(l_frames, v_shift, nFFT):
    nFFThalf = nFFT / 2 + 1
    nfrms    = len(v_shift)
    m_frm    = np.zeros((nfrms, nFFT))
    for i in xrange(nfrms):
        rel_shift  = nFFThalf - v_shift[i] - 1
        m_frm[i,:] = frame_shift(l_frames[i], rel_shift, nFFT)  
    
    return m_frm

#------------------------------------------------------------------------------
def frame_shift(v_frm, shift, out_len):
    right_len = out_len - (shift + len(v_frm))
    v_frm_out = np.hstack(( np.zeros(shift) , v_frm, np.zeros(right_len)))
    return v_frm_out
    
#------------------------------------------------------------------------------
# "Cosine window": cos_win**2 = hannnig
# power: 1=> coswin, 2=> hanning
def cos_win(N):
    v_x   = np.linspace(0,np.pi,N)
    v_win = np.sin(v_x)
    return v_win

#------------------------------------------------------------------------------
def hz_to_bin(v_hz, nFFT, fs):    
    return v_hz * nFFT / float(fs)

def bin_to_hz(v_bin, nFFT, fs):         
    return v_bin * fs / float(nFFT)

#------------------------------------------------------------------------------
# m_sp_l: spectrum on the left. m_sp_r: spectrum on the right
# TODO: Processing fo other freq scales, such as Mel.
def spectral_crossfade(m_sp_l, m_sp_r, cut_off, bw, fs, freq_scale='hz'):

    # Hz to bin:
    nFFThalf = m_sp_l.shape[1]
    nFFT     = (nFFThalf - 1) * 2    
    bin_l    = lu.round_to_int(hz_to_bin(cut_off - bw/2, nFFT, fs))     
    bin_r    = lu.round_to_int(hz_to_bin(cut_off + bw/2, nFFT, fs))

    # Gen short windows:
    bw_bin       = bin_r - bin_l
    v_win_shrt   = np.hanning(2*bw_bin + 1)
    v_win_shrt_l = v_win_shrt[bw_bin:]
    v_win_shrt_r = v_win_shrt[:bw_bin+1]
    
    # Gen long windows:
    v_win_l = np.hstack((np.ones(bin_l),  v_win_shrt_l , np.zeros(nFFThalf - bin_r - 1)))
    v_win_r = np.hstack((np.zeros(bin_l), v_win_shrt_r , np.ones(nFFThalf - bin_r - 1)))
    
    # Apply windows:
    m_sp_l_win = m_sp_l * v_win_l[None,:]
    m_sp_r_win = m_sp_r * v_win_r[None,:]
    m_sp       = m_sp_l_win + m_sp_r_win
    
    return m_sp
    

#------------------------------------------------------------------------------
def rceps_to_min_phase_rceps(m_rceps):
    '''
    # m_rceps: Complete real cepstrum (length=nfft)
    '''
    nFFThalf = m_rceps.shape[1] / 2 + 1
    m_rceps[:,1:(nFFThalf-1)] *= 2

    return m_rceps[:nFFThalf]


#------------------------------------------------------------------------------
# nc: number of coeffs
# fade_to_total: ratio between the length of the fade out over the total ncoeffs
def spectral_smoothing_rceps(m_sp_log, nc_total=60, fade_to_total=0.2):
    '''
    m_sp_log could be in any base log or decibels.
    '''

    nc_fade = lu.round_to_int(fade_to_total * nc_total)

    # Adding hermitian half:
    m_sp_log_ext = add_hermitian_half(m_sp_log)

    # Getting Cepstrum:
    m_rceps = np.fft.ifft(m_sp_log_ext).real

    m_rceps_minph = rceps_to_min_phase_rceps(m_rceps)
    #v_ener_orig_rms = np.sqrt(np.mean(m_rceps_minph**2,axis=1))
    
    # Create window:
    v_win_shrt = np.hanning(2*nc_fade+3)
    v_win_shrt = v_win_shrt[nc_fade+2:-1]    
        
    # Windowing:    
    m_rceps_minph[:,nc_total:] = 0
    m_rceps_minph[:,nc_total-nc_fade:nc_total] *= v_win_shrt

    # Energy compensation:
    #v_ener_after_rms = np.sqrt(np.mean(m_rceps_minph**2,axis=1))
    #v_ener_fact      = v_ener_orig_rms / v_ener_after_rms
    #m_rceps_minph    = m_rceps_minph * v_ener_fact[:,None]
    
    # Go back to spectrum:
    nfft        = m_rceps.shape[1]
    m_sp_log_sm = np.fft.fft(m_rceps_minph, n=nfft).real
    m_sp_log_sm = remove_hermitian_half(m_sp_log_sm)
    #m_sp_sm = np.exp(m_sp_sm)
    
    return m_sp_log_sm

#------------------------------------------------------------------------------
def log(m_x):
    '''
    Protected log: Uses MAGIC number to floor the logarithm.
    '''    
    m_y = np.log(m_x) 
    m_y[np.isinf(m_y)] = MAGIC
    return m_y    
    
#------------------------------------------------------------------------------
# out_type: 'compact' or 'whole'
def rceps(m_data, in_type='log', out_type='compact'):
    """
    in_type: 'abs', 'log' (any log base), 'td' (time domain).
    TODO: 'td' case not implemented yet!!
    """
    ncoeffs = m_data.shape[1]
    if in_type == 'abs':
        m_data = log(m_data)    
        
    m_data  = add_hermitian_half(m_data, data_type='magnitude')
    m_rceps = np.fft.ifft(m_data).real

    # Amplify coeffs in the middle:
    if out_type == 'compact':        
        m_rceps[:,1:(ncoeffs-2)] *= 2
        m_rceps = m_rceps[:,:ncoeffs]
    
    return m_rceps 

#------------------------------------------------------------------------------
# interp_type: e.g., 'linear', 'slinear', 'zeros'
def interp_unv_regions(m_data, v_voi, voi_cond='>0', interp_type='linear'):

    vb_voiced   = eval('v_voi ' + voi_cond)
    
    if interp_type == 'zeros':
        m_data_intrp = m_data * vb_voiced[:,None]

    else:
        v_voiced_nx = np.nonzero(vb_voiced)[0]
    
        m_strt_and_end_voi_frms = np.vstack((m_data[v_voiced_nx[0],:] , m_data[v_voiced_nx[-1],:]))        
        t_strt_and_end_voi_frms = tuple(map(tuple, m_strt_and_end_voi_frms))
        
        func_intrp  = interpolate.interp1d(v_voiced_nx, m_data[vb_voiced,:], bounds_error=False , axis=0, fill_value=t_strt_and_end_voi_frms, kind=interp_type)
        
        nFrms = np.size(m_data, axis=0)
        m_data_intrp = func_intrp(np.arange(nFrms))
    
    return m_data_intrp

'''
#------------------------------------------------------------------------------
# Generates time-domain non-symmetric "flat top" windows
# Also, it can be used for generating non-symetric windows ("non-flat top")
# func_win: e.g., numpy.hanning
# flat_to_len_ratio: flat_length / total_length. Number [0,1]
def gen_wider_window(func_win,len_l, len_r, flat_to_len_ratio):
    fade_to_len_ratio = 1 - flat_to_len_ratio  
    
    len_l = lu.round_to_int(len_l)
    len_r = lu.round_to_int(len_r)
    
    len_l_fade = lu.round_to_int(fade_to_len_ratio * len_l)     
    len_r_fade = lu.round_to_int(fade_to_len_ratio * len_r) 
        
    v_win_l   = func_win(2 * len_l_fade + 1)
    v_win_l   = v_win_l[:len_l_fade]
    v_win_r   = func_win(2 * len_r_fade + 1)
    v_win_r   = v_win_r[len_r_fade+1:]
    len_total = len_l + len_r
    len_flat  = len_total - (len_l_fade + len_r_fade)
    v_win     = np.hstack(( v_win_l, np.ones(len_flat) , v_win_r ))
        
    return v_win
'''

# Read audio file:-------------------------------------------------------------
def read_audio_file(filepath, **kargs):
    '''
    Wrapper function. For now, just to keep consistency with the library
    '''    
    return sf.read(filepath, **kargs)
    
# Write wav file:--------------------------------------------------------------
# The format is picked automatically from the file extension. ('WAV', 'FLAC', 'OGG', 'AIFF', 'WAVEX', 'RAW', or 'MAT5')
# v_signal be mono (TODO: stereo, comming soon), values [-1,1] are expected if no normalisation is selected.
def write_audio_file(filepath, v_signal, fs, norm=0.98):
    '''
    norm: If None, no normalisation is applied. If it is a float number,
          it is the target value (absolute) for the normalisation.
    '''
    
    # Normalisation:
    if norm is not None:
        v_signal = norm * v_signal / np.max(np.abs(v_signal)) # default
        
    # Write:    
    sf.write(filepath, v_signal, fs)
    
    return

# def write_audio_file(filepath, v_signal, fs, **kargs):

#     # Parsing input:
#     if 'norm' in kargs:
#         if kargs['norm'] == False:
#             pass

#         elif kargs['norm'] == 'max':
#             v_signal = v_signal / np.max(np.abs(v_signal))

#         del(kargs['norm'])
#     else:
#         v_signal = v_signal / np.max(np.abs(v_signal)) # default

#     # Write:
#     sf.write(filepath, v_signal, fs, **kargs)

#     return


'''
# 1-D Smoothing by convolution: (from ScyPy Cookbook - not checked yet!)-----------------------------
def smooth_by_conv(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.
    output:
        the smoothed signal
        
    example:
    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """ 
     
    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."
    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."
        
    if window_len<3:
        return x
    
    
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
    
    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')
    
    y=np.convolve(w/w.sum(),s,mode='valid')
    return y 
'''
#------------------------------------------------------------------------------
# data_type: 'magnitude', 'phase' or 'zeros' (for zero padding), 'complex'
def add_hermitian_half(m_data, data_type='mag'):
           
    if (data_type == 'mag') or (data_type == 'magnitude'):
        m_data = np.hstack((m_data , np.fliplr(m_data[:,1:-1])))
        
    elif data_type == 'phase':        
        m_data[:,0]  = 0            
        m_data[:,-1] = 0   
        m_data = np.hstack((m_data , -np.fliplr(m_data[:,1:-1])))

    elif data_type == 'zeros':
        nfrms, nFFThalf = m_data.shape
        m_data = np.hstack((m_data , np.zeros((nfrms,nFFThalf-2))))
        
    elif data_type == 'complex':
        m_data_real = add_hermitian_half(m_data.real)
        m_data_imag = add_hermitian_half(m_data.imag, data_type='phase')
        m_data      = m_data_real + m_data_imag * 1j
    
    return m_data

# Remove hermitian half of fft-based data:-------------------------------------
# Works for either even or odd fft lenghts.
def remove_hermitian_half(m_data):
    dp = lu.DimProtect(m_data)
    
    nFFThalf   = int(np.floor(np.size(m_data,1) / 2)) + 1
    m_data_rem = m_data[:,:nFFThalf].copy()

    dp.end(m_data_rem)
    return m_data_rem
    
#-----------------------------------------------------
def read_est_file(est_file):
    '''
    Generic function to read est files. So far, it reads the first two columns of est files. (TODO: expand)
    '''

    # Get EST_Header_End line number: (TODO: more efficient)
    with open(est_file) as fid:
        header_size = 1 # init
        for line in fid:
            if line == 'EST_Header_End\n':
                break
            header_size += 1

    m_data = np.loadtxt(est_file, skiprows=header_size, usecols=[0,1])
    return m_data

#------------------------------------------------------------------------------
# check_len_smpls= signal length. If provided, it checks and fixes for some pm out of bounds (REAPER bug)
# fs: Must be provided if check_len_smpls is given
def read_reaper_est_file(est_file, check_len_smpls=-1, fs=-1, skiprows=7, usecols=[0,1]):

    # Checking input params:
    if (check_len_smpls > 0) and (fs == -1):
        raise ValueError('If check_len_smpls given, fs must be provided as well.')

    # Read text: TODO: improve skiprows
    m_data = np.loadtxt(est_file, skiprows=skiprows, usecols=usecols)
    m_data = np.atleast_2d(m_data)
    v_pm_sec  = m_data[:,0]
    v_voi = m_data[:,1]

    # Protection against REAPER bugs 1:
    vb_correct = np.hstack(( True, np.diff(v_pm_sec) > 0))
    v_pm_sec  = v_pm_sec[vb_correct]
    v_voi = v_voi[vb_correct]

    # Protection against REAPER bugs 2 (maybe it needs a better protection):
    if (check_len_smpls > 0) and ( (v_pm_sec[-1] * fs) >= (check_len_smpls-1) ):
        v_pm_sec  = v_pm_sec[:-1]
        v_voi = v_voi[:-1]
    return v_pm_sec, v_voi

# REAPER wrapper:--------------------------------------------------------------
def reaper(in_wav_file, out_est_file):
    print("Extracting epochs with REAPER...")
    global _reaper_bin
    cmd =  _reaper_bin + " -s -x 400 -m 50 -a -u 0.005 -i %s -p %s" % (in_wav_file, out_est_file)
    call(cmd, shell=True)
    return
    
#------------------------------------------------------------------------------
def f0_to_lf0(v_f0):
       
    old_settings = np.seterr(divide='ignore') # ignore warning
    v_lf0 = np.log(v_f0)
    np.seterr(**old_settings)  # reset to default
    
    v_lf0[np.isinf(v_lf0)] = MAGIC
    return v_lf0

# Get pitch marks from signal using REAPER:------------------------------------

def get_pitch_marks(v_sig, fs):
    
    temp_wav = lu.ins_pid('temp.wav')
    temp_pm  = lu.ins_pid('temp.pm')
        
    sf.write(temp_wav, v_sig, fs)
    reaper(temp_wav, temp_pm)
    v_pm = np.loadtxt(temp_pm, skiprows=7)
    v_pm = v_pm[:,0]
    
    # Protection against REAPER bugs 1:
    vb_correct = np.hstack(( True, np.diff(v_pm) > 0))
    v_pm = v_pm[vb_correct]
    
    # Protection against REAPER bugs 2 (maybe I need a better protection):
    if (v_pm[-1] * fs) >= (np.size(v_sig)-1):
        v_pm = v_pm[:-1]
    
    # Removing temp files:
    os.remove(temp_wav)
    os.remove(temp_pm)
    
    return v_pm


# Next power of two:-----------------------------------------------------------
def next_pow_of_two(x):
    # Protection:    
    if x < 2: 
        x = 2
    # Safer for older numpy versions:
    x = 2**np.ceil(np.log2(x)).astype(int)
    
    return x

#---------------------------------------------------------------------------
def windowing(v_sig, winlen, shift, winfunc=np.hanning, extend='none'):
    '''
    Typical constant frame rate windowing function
    winlen and shift (hopsize) in samples.
    extend: 'none', 'both', 'beg', 'end' . Extension of v_sig towards its beginning and/or end.
    '''
    shift = int(shift)
    vWin   = winfunc(winlen)
    frmLen = len(vWin)

    if extend=='both' or extend=='beg':
        nZerosBeg = int(np.floor(frmLen/2))
        vZerosBeg = np.zeros(nZerosBeg)
        v_sig     = np.concatenate((vZerosBeg, v_sig))

    if extend=='both' or extend=='end':
        nZerosEnd = frmLen
        vZerosEnd = np.zeros(nZerosEnd)
        v_sig     = np.concatenate((v_sig, vZerosEnd))

    nFrms  = np.floor(1 + (v_sig.shape[0] - winlen) / float(shift)).astype(int)
    mSig   = np.zeros((nFrms, frmLen))
    nxStrt = 0
    for t in xrange(nFrms):
        #print(t)
        mSig[t,:] = v_sig[nxStrt:(nxStrt+frmLen)] * vWin
        nxStrt = nxStrt + shift   
    
    return mSig

# This function is provided to to avoid confusion about how to compute the exact 
# number of frames from shiftMs and fs    
def GetNFramesFromSigLen(sigLen, shiftMs, fs):
    
    shift = np.round(fs * shiftMs / 1000)
    nFrms = np.ceil(1 + ((sigLen - 1) / shift))
    nFrms = int(nFrms)
    
    return nFrms


#==============================================================================
# Converts mcep to lin sp, without doing any  Mel warping.
def mcep_to_lin_sp_log(mgc_mat, nFFT):
    
    nFrms, n_coeffs = mgc_mat.shape
    nFFTHalf = 1 + nFFT/2
    
    mgc_mat = np.concatenate((mgc_mat, np.zeros((nFrms, (nFFT/2 - n_coeffs + 1)))),1)
    mgc_mat = np.concatenate((mgc_mat, np.fliplr(mgc_mat[:,1:-1])),1)
    sp_log  = (np.fft.fft(mgc_mat, nFFT,1)).real
    sp_log  = sp_log[:,0:nFFTHalf]

    return sp_log 

    
#Gets RMS from matrix no matter the number of bins m_data has, 
#it figures out according to the FFT length.
# For example, nFFT = 128 , nBins_data= 60 (instead of 65 or 128)
def get_rms(m_data, nFFT):
    m_data2 = m_data**2
    m_data2[:,1:(nFFT/2)] = 2 * m_data2[:,1:(nFFT/2)]    
    v_rms = np.sqrt(np.sum(m_data2[:,0:(nFFT/2+1)],1) / nFFT)    
    return v_rms   
    
# Converts spectrum to MCEPs using SPTK toolkit--------------------------------  
# if alpha=0, no spectral warping
# m_sp: absolute and non redundant spectrum
# in_type: Type of input spectrum. if 3 => |f(w)|. If 1 => 20*log|f(w)|. If 2 => ln|f(w)|
# fft_len: If 0 => automatic computed from input data, If > 0 , is the value of the fft length
def sp_to_mcep(m_sp, n_coeffs=60, alpha=0.77, in_type=3, fft_len=0):

    #Pre:
    temp_sp  =  lu.ins_pid('temp.sp')
    temp_mgc =  lu.ins_pid('temp.mgc')
    
    # Writing input data:
    lu.write_binfile(m_sp, temp_sp)

    if fft_len is 0: # case fft automatic
        fft_len = 2*(np.size(m_sp,1) - 1)

    # MCEP:      
    curr_cmd = _sptk_mcep_bin + " -a %1.2f -m %d -l %d -e 1.0E-8 -j 0 -f 0.0 -q %d %s > %s" % (alpha, n_coeffs-1, fft_len, in_type, temp_sp, temp_mgc)
    call(curr_cmd, shell=True)
    
    # Read MGC File:
    m_mgc = lu.read_binfile(temp_mgc , n_coeffs)
    
    # Deleting temp files:
    os.remove(temp_sp)
    os.remove(temp_mgc)
    
    #$sptk/mcep -a $alpha -m $mcsize -l $nFFT -e 1.0E-8 -j 0 -f 0.0 -q 3 $sp_dir/$sentence.sp > $mgc_dir/$sentence.mgc
    
    return m_mgc

'''
# MCEP to SP using SPTK toolkit.-----------------------------------------------
# m_sp is absolute and non redundant spectrum
# out_type = type of output spectrum. If out_type==0 -> 20*log|H(z)|. If out_type==2 -> |H(z)| . If out_type==1 -> ln|H(z)|
def mcep_to_sp_sptk(m_mgc, nFFT, alpha=0.77, out_type=2): 
  
    n_coeffs = m_mgc.shape[1]    

    temp_mgc =  ins_pid('temp.mgc') 
    temp_sp  =  ins_pid('temp.sp')
    
    lu.write_binfile(m_mgc,temp_mgc)

    # MGC to Spec:
    curr_cmd = _curr_dir + "/SPTK-3.7/bin/mgc2sp -a %1.2f -g 0 -m %d -l %d -o %d %s > %s" % (alpha, n_coeffs-1, nFFT, out_type, temp_mgc, temp_sp)
    call(curr_cmd, shell=True) 

    m_sp = lu.read_binfile(temp_sp, dim=1+nFFT/2)
    if np.size(m_sp,0) == 1: # protection when it is only one frame
        m_sp = m_sp[0]
    
    os.remove(temp_mgc)
    os.remove(temp_sp)
   
    return m_sp
'''
#============================================================================== 
# out_type: 'db', 'log', 'abs' (absolute)    
def mcep_to_sp_cosmat(m_mcep, n_spbins, alpha=0.77, out_type='abs'):
    '''
    mcep to sp using dot product with cosine matrix.
    '''
    # Warping axis:
    n_cepcoeffs = m_mcep.shape[1]
    v_bins_out  = np.linspace(0, np.pi, num=n_spbins)
    v_bins_warp = np.arctan(  (1-alpha**2) * np.sin(v_bins_out) / ((1+alpha**2)*np.cos(v_bins_out) - 2*alpha) ) 
    v_bins_warp[v_bins_warp < 0] += np.pi
    
    # Building matrix:
    m_trans = np.zeros((n_cepcoeffs, n_spbins))
    for nxin in xrange(n_cepcoeffs):
        for nxout in xrange(n_spbins):
            m_trans[nxin, nxout] = np.cos( v_bins_warp[nxout] * nxin )        
            
    # Apply transformation:
    m_sp = np.dot(m_mcep, m_trans)
    
    if out_type == 'abs':
        m_sp = np.exp(m_sp)
    elif out_type == 'db':
        m_sp = m_sp * (20 / np.log(10))
    elif out_type == 'log':
        pass
    
    return m_sp

# Absolute to Decibels:--------------------------------------------------------
# b_inv: inverse function
def db(m_data, b_inv=False):
    if b_inv==False:
        return 20 * np.log10(m_data) 
    elif b_inv==True:
        return 10 ** (m_data / 20)

            
# in_type: Type of input spectrum. if 3 => |f(w)|. If 1 => 20*log|f(w)|. If 2 => ln|f(w)|        
def sp_mel_warp(m_sp, nbins_out, alpha=0.77, in_type=3):
    '''
    Info:
    in_type: Type of input spectrum. if 3 => |f(w)|. If 1 => 20*log|f(w)|. If 2 => ln|f(w)|        
    '''
    
    # sp to mcep:
    m_mcep = sp_to_mcep(m_sp, n_coeffs=nbins_out, alpha=alpha, in_type=in_type)
    
    # mcep to sp:
    if in_type == 3:
        out_type = 'abs'
    elif in_type == 1:
        out_type = 'db'
    elif in_type == 2:
        out_type = 'log'
        
    m_sp_wrp = mcep_to_sp_cosmat(m_mcep, nbins_out, alpha=0.0, out_type=out_type)
    return m_sp_wrp
    

#==============================================================================
# in_type: 'abs', 'log'
# TODO: 'db'
def sp_mel_unwarp(m_sp_mel, nbins_out, alpha=0.77, in_type='log'):
    
    ncoeffs = m_sp_mel.shape[1]
    
    if in_type == 'abs':
        m_sp_mel = np.log(m_sp_mel)
    
    #sp to mcep:
    m_sp_mel = add_hermitian_half(m_sp_mel, data_type='magnitude')
    m_mcep   = np.fft.ifft(m_sp_mel).real
    
    # Amplify coeffs in the middle:    
    m_mcep[:,1:(ncoeffs-2)] *= 2
       
    #mcep to sp:    
    m_sp_unwr = mcep_to_sp_cosmat(m_mcep[:,:ncoeffs], nbins_out, alpha=alpha, out_type=in_type)
    
    return m_sp_unwr


def convert_label_state_align_to_var_frame_rate(in_lab_st_file, v_dur_state, out_lab_st_file):
    # Constants:
    shift_ms = 5.0

    # Read input files:
    mstr_labs_st = np.loadtxt(in_lab_st_file, dtype='string', delimiter=" ", comments=None, usecols=(2,))

    v_dur_ms = v_dur_state * shift_ms
    v_dur_ns = v_dur_ms * 10000
    v_dur_ns = np.hstack((0,v_dur_ns))
    v_dur_ns_cum = np.cumsum(v_dur_ns)
    m_dur_ns_cum = np.vstack((v_dur_ns_cum[:-1], v_dur_ns_cum[1:])).T.astype(int)

    # To string array:
    mstr_dur_ns_cum = np.char.mod('%d', m_dur_ns_cum)

    # Concatenate data:
    mstr_out_labs_st = np.hstack((mstr_dur_ns_cum, mstr_labs_st[:,None]))

    # Save file:
    np.savetxt(out_lab_st_file, mstr_out_labs_st,  fmt='%s')
    return


