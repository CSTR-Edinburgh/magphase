#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

@author: felipe
"""
import numpy as np
import shutil

import sys, os
curr_dir = os.getcwd()
sys.path.append(os.path.realpath(curr_dir + '/../src'))
import libutils as lu
import magphase as mp
import configparser # Install it with pip (it's not the same as 'ConfigParser' (old version))
import subprocess


# Debug:
class Struct:
    pass


def copytree(src_dir, l_items, dst_dir, symlinks=False, ignore=None):
    for item in l_items:
        s = os.path.join(src_dir, item)
        d = os.path.join(dst_dir, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)

def feat_extraction(in_wav_dir, file_name_token, out_feats_dir, d_opts):

    # Display:
    print("\nAnalysing file: " + file_name_token + '.wav............................')

    # File setup:
    wav_file = os.path.join(in_wav_dir, file_name_token + '.wav')

    mp.analysis_compressed_type1_with_phase_comp(wav_file, fft_len=None, out_dir=out_feats_dir, nbins_mel=60,
                                                                    nbins_phase=d_opts['nbins_phase'],
                                                                    b_const_rate=d_opts['b_const_rate'],
                                                                                        b_mag_fbank_mel=False)

    return

if __name__ == '__main__':
    # CONSTANTS:============================================================================
    merlin_path = '/afs/inf.ed.ac.uk/group/cstr/projects/Felipe_Espic/Projects/DirectFFTWaveModelling/magphase_proj/merlin'
    base_exper_path = '/afs/inf.ed.ac.uk/group/cstr/projects/Felipe_Espic/Projects/DirectFFTWaveModelling/magphase_proj/merlin/egs/nick/nick_magphase_type1_var_rate'

    l_files_and_dirs_to_copy = ['conf',
                                'file_id_list_20.scp',
                                'file_id_list_fixed.scp',
                                'file_id_list.scp',
                                'questions_dnn_481.hed',
                                'scripts',
                                'data/label_state_align',
                                'data/label_state_align_from_var_rate']

    # Feature Extraction:
    in_wav_dir         = '/afs/inf.ed.ac.uk/group/cstr/projects/Felipe_Espic/Databases/Nick-Zhizheng_dnn_baseline_practice/data/wav'
    acoustic_feats_dir = 'data/acoustic_feats'




    # Merlin's config file:
    question_file_name = 'questions_dnn_481.hed'

    # INPUT:================================================================================#
    # Setup:
    b_setup_files   = False
    b_feat_extr     = False
    b_config_merlin = False
    b_run_merlin    = False
    b_wavgen        = True

    # General:
    exper_path = '/afs/inf.ed.ac.uk/group/cstr/projects/Felipe_Espic/Projects/DirectFFTWaveModelling/magphase_proj/merlin/egs/nick/01_nick_magphase_type1_var_rate_new_ph_45'

    b_feat_ext_multiproc = False

    d_mp_opts = {'nbins_phase' : 20,
                 'b_const_rate': False,
                 'l_pf_type'   : [ 'no', 'merlin', 'magphase'] # 'magphase', 'merlin', 'no'
                 }

    exper_mode = 'trial' # 'full', 'trial'

    NORMLAB  = True
    MAKECMP  = True
    NORMCMP  = True
    TRAINDNN = True
    DNNGEN   = True
    GENWAV   = False
    CALMCD   = True

    # PROCESS:================================================================================
    # Pre setup:
    if exper_mode=='full':
        file_id_list = 'file_id_list_fixed.scp'
    elif exper_mode=='trial':
        file_id_list = 'file_id_list_20.scp'

    # Read file list:
    l_file_tokns = lu.read_text_file2(os.path.join(exper_path, file_id_list), dtype='string', comments='#').tolist()

    #-----------------------------------------------------------------------------------------

    if b_setup_files:
        # Copy files and directories from base to current experiment:
        copytree(base_exper_path, l_files_and_dirs_to_copy, exper_path)
        os.rename(os.path.join(exper_path, 'conf/config.conf'), os.path.join(exper_path, 'conf/config_base.conf'))

        # Save backup of this file and used magphase code:
        shutil.copytree(os.path.dirname(mp.__file__), os.path.join(exper_path, 'backup_magphase_code'))
        shutil.copy2(__file__, os.path.join(exper_path, 'conf'))

    if b_feat_extr:
        # Extract features:
        lu.mkdir(os.path.join(exper_path, acoustic_feats_dir))

        if b_feat_ext_multiproc:
            lu.run_multithreaded(feat_extraction, in_wav_dir, l_file_tokns, acoustic_feats_dir, d_mp_opts)
        else:
            for file_name_token in l_file_tokns:
                feat_extraction(in_wav_dir, file_name_token, os.path.join(exper_path,acoustic_feats_dir), d_mp_opts)

    if b_config_merlin or b_wavgen:
        # Edit Merlin's config file:
        parser = configparser.ConfigParser()
        parser.optionxform = str
        parser.read([os.path.join(exper_path, 'conf/config_base.conf')])

        parser['DEFAULT']['TOPLEVEL'] = exper_path
        parser['Paths']['file_id_list']   = "%(work)s/" + file_id_list
        parser['Labels']['question_file_name'] = os.path.join(exper_path , question_file_name)

        parser['Outputs']['real' ] = '%d' %  d_mp_opts['nbins_phase']
        parser['Outputs']['imag' ] = '%d' %  d_mp_opts['nbins_phase']
        parser['Outputs']['dreal'] = '%d' % (d_mp_opts['nbins_phase']*3)
        parser['Outputs']['dimag'] = '%d' % (d_mp_opts['nbins_phase']*3)


        if exper_mode=='full':
            parser['Architecture']['model_file_name']   = 'feed_forward_4_tanh'
            parser['Architecture']['hidden_layer_size'] = "[1024, 1024, 1024, 1024]"
            parser['Architecture']['hidden_layer_type'] = "['TANH', 'TANH', 'TANH', 'TANH']"
            parser['Architecture']['warmup_epoch']      = '10'
            parser['Architecture']['training_epochs']   = '25'

            parser['Data']['train_file_number'] = '2400'
            parser['Data']['valid_file_number'] = '70'
            parser['Data']['test_file_number']  = '71'

        elif exper_mode=='trial':
            parser['Architecture']['model_file_name']   = 'feed_forward_1_tanh'
            parser['Architecture']['hidden_layer_size'] = "[256]"
            parser['Architecture']['hidden_layer_type'] = "['TANH']"
            parser['Architecture']['warmup_epoch']      = '2'
            parser['Architecture']['training_epochs']   = '3'

            parser['Data']['train_file_number'] = '12'
            parser['Data']['valid_file_number'] = '4'
            parser['Data']['test_file_number']  = '4'


        parser['Processes']['NORMLAB' ] = '%s' % NORMLAB
        parser['Processes']['MAKECMP' ] = '%s' % MAKECMP
        parser['Processes']['NORMCMP' ] = '%s' % NORMCMP
        parser['Processes']['TRAINDNN'] = '%s' % TRAINDNN
        parser['Processes']['DNNGEN'  ] = '%s' % DNNGEN
        parser['Processes']['GENWAV'  ] = '%s' % GENWAV
        parser['Processes']['CALMCD'  ] = '%s' % CALMCD

        with open(os.path.join(exper_path ,'conf/config.conf'), 'wb') as file:
            parser.write(file)


    # Run Merlin:
    if b_run_merlin:
        subprocess.call([ os.path.join(exper_path , 'scripts/submit.sh'),
                          os.path.join(merlin_path, 'src/run_merlin.py'),
                          os.path.join(exper_path , 'conf/config.conf')])


    # Waveform generation:
    if b_wavgen:
        gen_feats_path = os.path.join(exper_path, 'gen', parser['Architecture']['model_file_name'])
        n_testfiles = int(parser['Data'    ]['test_file_number'])
        nbins_mag   = int(parser['Outputs' ]['mag'])
        fs          = int(parser['Waveform']['samplerate'])

        for file_tokn in l_file_tokns[-n_testfiles:]:
            for pf_type in d_mp_opts['l_pf_type']:
                gen_wav_path = gen_feats_path + '_pf_' + pf_type
                lu.mkdir(gen_wav_path)
                mp.synthesis_from_acoustic_modelling(gen_feats_path, file_tokn, gen_wav_path, nbins_mag,
                                                                        d_mp_opts['nbins_phase'], fs, pf_type=pf_type)





