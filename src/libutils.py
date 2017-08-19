# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 23:16:51 2016

@author: Felipe Espic
"""

#==============================================================================
# IMPORTS
#==============================================================================
import numpy as np
import os
import glob
import time
from multiprocessing import Pool

#==============================================================================
# FUNCTIONS
#==============================================================================

# Multithreading:----------------------------------------------------------------------------
def func_wrapper(args):
    '''
    Helper function used by "run_multithreaded"
    '''
    func = args[0]
    args_defi = args[1:]
    func(*args_defi)
    return

def run_multithreaded(*args):

    '''
    args[0]: target function
    args[1:]: Arguments for the target function in order. When an argument is iterable (only lists supported so far), concatenate it into a list.
              If an argument is constant, just pass its value. Then, the function will figure out to fill (repeat the constant values per run/iteration)
    '''
    # Getting target function:
    func = args[0]

    # Getting number of runs:
    for nxa in xrange(1,len(args)):
        if type(args[nxa]) is list:
            nruns = len(args[nxa])
            break

    # Building iterable list:
    l_iterable_args = []
    for nxr in xrange(nruns):
        l_curr_args = [func]
        for nxa in xrange(1,len(args)):
            if type(args[nxa]) is list:
                l_curr_args.append(args[nxa][nxr])
            else:
                l_curr_args.append(args[nxa])

        l_iterable_args.append(tuple(l_curr_args))

    # Run multiprocess:
    pool    = Pool()
    results = pool.map(func_wrapper, l_iterable_args)
    return results

#---------------------------------------------------------------------------------

def gen_list_of_file_paths(files_dir, v_file_tkns, suffix):
    '''
    v_file_tkns: Could be a list of strings or a numpy array of strings.
    '''

    nfiles = len(v_file_tkns)
    l_file_paths = []
    for nxf in xrange(nfiles):
        l_file_paths.append(files_dir  + '/' + v_file_tkns[nxf] + suffix)

    return l_file_paths


#---------------------------------------------------------------------------------------------------

def indexes_to_one_zero_vector(v_nxs, length):
    '''
    v_nxs: Vector of indexes.
    length: Length of output vector.
    '''
    v_nxs_oz = np.zeros(length)
    v_nxs_oz[v_nxs.astype(int)] = 1

    return v_nxs_oz


# Read scp file:===============================================================
def read_scp_file(filename):
    return read_text_file2(filename, dtype='string', comments='#') 
   
# Read text file2:=============================================================
# Uses numpy.genfromtxt to read files, and protects against the "bug" for data with only one element.
def read_text_file2(*args, **kargs):
    data = np.genfromtxt(*args, **kargs)
    data = np.atleast_1d(data)    
    return data

# Get file list from path:======================================================
# e.g., files_path = "path/to/files/*.ext"
def get_file_list(files_path):
    files_list = glob.glob(files_path)
    n_files    = len(files_list)
    return files_list, n_files


def read_binfile(filename, dim=60):
    fid = open(filename, 'rb')
    v_data = np.fromfile(fid, dtype=np.float32)
    fid.close()
    if np.mod(v_data.size, dim) != 0:
        raise ValueError('Dimension provided not compatible with file size.')
    m_data = v_data.reshape((-1, dim)).astype('float64') # This is to keep compatibility with numpy default dtype.
    m_data = np.squeeze(m_data)
    return  m_data

def write_binfile(m_data, filename):
    m_data = np.array(m_data, 'float32') # Ensuring float32 output
    fid = open(filename, 'wb')
    m_data.tofile(fid)
    fid.close()
    return

# Rounds a number and converts into int (e.g., to be used as an index):========
# before, this function was called "round_int"
def round_to_int(float_num):    
    float_num = np.round(float_num).astype(int)
    return float_num

# Extract path, name and ext of a full path:===================================    
def fileparts(fullpath):
    path_with_token, ext = os.path.splitext(fullpath)            
    path,  filename      = os.path.split(fullpath)            
    filetoken            = os.path.basename(path_with_token)
    return [path, filetoken, ext, path_with_token]      

def get_filename(filepath):
    filename = os.path.splitext(os.path.basename(filepath))[0]
    return filename

def mkdir(l_dir):
    '''
    l_dir could be a directory or a list of directories.
    '''
    # To list:
    if type(l_dir)==str:
        l_dir = [l_dir]

    for directory in l_dir:
        if not os.path.exists(directory):
            os.mkdir(directory)
    return

class DimProtect(object):
    def __init__(self, *args):
        self.orig_ndim = args[0].ndim # dim depends on the first array passed
        if self.orig_ndim==1:
            for data in args:
                data.resize((1, data.size))
        return

    def end(self, *args):
        if self.orig_ndim==1:
            for data in args:
                data.resize(data.shape[1])
        return

def add_rel_path(rel_path):
    import sys, os, inspect
    caller_file = inspect.stack()[1][1]
    caller_dir  = os.path.dirname(caller_file)
    dir_to_add  = os.path.realpath(caller_dir + rel_path)
    sys.path.append(dir_to_add)

# Inserts pid to file name. This is useful when using temp files.--------------
# Example: path/file.wav -> path/file_pid.wav
def ins_pid(filepath):
    filename, ext = os.path.splitext(filepath)
    filename = "%s_%d%s" % (filename, os.getpid(), ext)
    return filename

# Inserts date and time to file name. This is useful for output files----------
# Example: path/file.wav -> path/file_prefix_date_time.wav
def ins_date_time(filepath, prefix=""):
    filename, ext = os.path.splitext(filepath)
    filename = "%s_%s_%s%s" % (filename, prefix, time.strftime("%Y%m%d_%H%M"), ext)
    return filename

