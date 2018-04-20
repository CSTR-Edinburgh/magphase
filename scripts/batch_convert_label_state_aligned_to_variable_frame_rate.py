#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Felipe Espic

DESCRIPTION:
As Merlin works at a constant frame rate and this vocoder runs at a variable frame rate, it is needed to trick Merlin by warping the time durations in the label files.
This script converts the original label files to the "variable rate frame" labels, thus compensating the variable-to-constant frame rate difference.

INSTRUCTIONS:
This demo should work out of the box. Just run it by typing: python <script name>
If you want to use this demo with real data to work with Merlin, just modify the directories, input files, accordingly.
See the main function below for details.

NOTE: The file crashlist_file will store the list of utterances that were not possible to convert.
This could happen if for example some phonemes had no frames assigned. This rarelly occurs.
"""

import sys, os
curr_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.realpath(curr_dir + '/../../src'))

import libutils as lu
import libaudio as la
import magphase as mp


if __name__ == '__main__':  
    
    # CONSTANTS: So far, the vocoder has been tested only with the following constants:===
    fs = 48000

    # INPUT:==============================================================================
    files_scp       = '../demos/data_48k/file_id.scp'   # List of file names (tokens). Format used by Merlin.
    in_lab_st_dir   = '../demos/data_48k/labs'          # Original state aligned label files directory (in the format used by Merlin).
    in_shift_dir    = '../demos/data_48k/params_nat'    # Directory containing .shift files. You need to run feature extraction before running this script
                                                        # (e.g., batch_feature_extrction_for_tts.py)
    out_lab_st_dir  = '../demos/data_48k/labs_var_rate' # Directory that will contain the converted "variable frame rate" state aligned label files.
    b_prevent_zeros = False                             # True if you want to make sure that all the phonemes have one frame at least.
                                                        # (not recommended, only usful when there are too many utterances crashed)


    # PROCESSING:=========================================================================
    lu.mkdir(out_lab_st_dir)
    v_fileTokns = lu.read_text_file2(files_scp, dtype='string', comments='#')
    n_files = len(v_fileTokns)
    
    crashlist_file = lu.ins_pid('crash_file_list.scp')
    for ftkn in v_fileTokns:
        
        # Display:
        print('\nAnalysing file: ' + ftkn + '................................')
        
        # Input files:
        in_lab_st_file  = in_lab_st_dir  + '/' + ftkn + '.lab'
        out_lab_st_file = out_lab_st_dir + '/' + ftkn + '.lab'
        in_shift_file   = in_shift_dir   + '/' + ftkn + '.shift'

        try:
            v_shift  = lu.read_binfile(in_shift_file, dim=1)
            v_n_frms = mp.get_num_of_frms_per_state(v_shift, in_lab_st_file, fs, b_prevent_zeros=b_prevent_zeros, n_states_x_phone=5, nfrms_tolerance=6)

            # Extraction:
            la.convert_label_state_align_to_var_frame_rate(in_lab_st_file, v_n_frms, out_lab_st_file)
        
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            with open(crashlist_file, "a") as crashlistlog:
                crashlistlog.write(ftkn + '\n')

    print('Done!')
        
