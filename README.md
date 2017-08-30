# MagPhase Vocoder
Speech analysis/synthesis system for TTS and related applications.

This software is based on the method described in the paper:

[F. Espic, C. Valentini-Botinhao, and S. King, “Direct Modelling of Magnitude and Phase Spectra for Statistical Parametric Speech Synthesis,” in Proc. Interspeech, Stockholm, Sweden, August, 2017.](http://www.isca-speech.org/archive/Interspeech_2017/pdfs/1647.PDF)

@ Author: [Felipe Espic](http://felipeespic.com)

More information at http://www.felipeespic.com/magphase/

## I. Description
This is a speech waveform analysis/synthesis system used in Statistical Parametric Speech Synthesis (SPSS).

The analysis module extracts four feature streams describing magnitude spectra, phase spectra, and F0. These features can be used to train a regression model (e.g., DNN, LSTM, HMM. etc.) so then, predicted values can be generated.
The synthesis module takes these features at the input to generate the final synthesised waveform.

Key points:
* Avoids estimation steps as far as possible (no aperiodicities, spectral envelope, or harmonics estimation, etc.)
* Robust extraction and modelling of phase spectra (Conventional vocoders just create artificial phase at the output).
* No phase unwrapping required.
* Uses fast operations during synthesis (e.g., FFT, PSOLA).
* Remarkably reduces typical "buzziness" and "phasiness".
* Many other applications and improvements not explored yet.

## II. License:
See the LICENCE file for details.

## III. Prerequisites:
* OS: Linux (MacOSx comming soon)
* Python 2.7
* Standard Python packages: numpy, scipy, soundfile, matplotlib

## IV. Install:
1. Install Pyhton 2.7 and the packages required using the package manager of your distro or by using the command pip (recomended).
e.g.,
```
sudo pip install numpy scipy soundfile matplotlib
```
2. Download and compile SPTK and REAPER:
```
cd tools
./download_and_compile_tools.sh
```
This will compile SPTK and REAPER automatically for you...and that's it!

## V. Usage:
Just go to ```/demos```, read the instructions inside the demo scripts, which are very discriptive.
They should run out of the box by running ```python <demo_script>```.

We recomend that you play firstly with ```demo_copy_synthesis_high_res.py``` , and then ```demo_copy_synthesis_low_dim.py```
They both perform analysis/synthesis routines.

Then, you can modify the demo scripts to suit your needs.

**NOTE:** Just remember to run the scripts from their locations.

## VI. Using with the Merlin toolkit:
We provide demo scripts  in ```/demos/run_for_merlin```. Firstly, run the demos in order to learn how they work, and then you can adapt them to work with the Merlin toolkit and real data.

When working with real data and the Merlin toolkit, follow these steps:

1. Before training, run the scripts (with the paths pointing to your data): ```0_batch_feature_extraction_for_merlin.py``` , and then ```1_batch_convert_label_state_aligned_to_variable_frame_rate.py```

2. Modify the Merlin config file to use the extracted MagPhase features:
Use the features mag, real, imag, and lf0 as output parameters for the NN (instead of the defauls: mgc, bap, and lf0).

3. Modify the Merlin config file to use the variable rate label by pointing to the ```/labs_var_rate``` directory.

4. Run all the Merlin steps up to DNNGEN (Do not run WAVGEN).

5. Run the script ```2_batch_wave_generation.py``` to generate the waveforms from the parameters predicted by Merlin (usually stored in ```<experiment_dir>/data/gen/```)
