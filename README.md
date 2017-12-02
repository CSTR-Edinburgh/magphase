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


## II. Colaboration:
We need help to improve this software. You can colaborate by:

* **Building TTS voices using Merlin and MagPhase and compare with other vocoders, e.g., WORLD.** Then, please tell us your results. We have tested MagPhase only with a few voices and it's needed to cover a wider range. We have recently fixed some bugs that have came out thanks to people reporting their results using new data.


* **Implementing native variable frame rate support in Merlin.** MagPhase works in a variable frame rate fashion (pitch synchronous). So far, in order to integrate Merlin with MagPhase, we have been applying a suboptimal [workaround](https://github.com/CSTR-Edinburgh/merlin/blob/master/egs/slt_arctic/s2/scripts/convert_label_state_align_to_variable_frame_rate.py). We stronlgy believe that the performance will be highly increased if Merlin's supported variable frame rate. Plase, let us know if you are interested on colaborating on this.


## III. License:
See the LICENCE file for details.

## IV. Requirements:
* OS: Linux (MacOSx coming soon)
* Python 2.7
* Standard Python packages: numpy, scipy, soundfile, matplotlib

## V. Install:
1. Install Pyhton 2.7 and the packages required using the package manager of your distro or by using the command pip (recomended).
e.g.,
```
pip install numpy scipy soundfile matplotlib
```

2. If you have SPTK and REAPER already installed in your system (e.g. in a Merlin installation), edit the **config.ini** file providing the paths for **reaper** and the SPTK's **mcep** binaries. Otherwise, download and compile SPTK and REAPER by:
```
cd tools
./download_and_compile_tools.sh
```
This will compile and configure SPTK and REAPER automatically for you...and that's it!

## VI. Usage:
Just go to ```/demos```, read the instructions inside the demo scripts, which are very discriptive.
They should run out of the box by running ```python <demo_script>```.

We recomend that you play firstly with ```demo_copy_synthesis_lossless.py``` , and then ```demo_copy_synthesis_low_dim.py```
They both perform analysis/synthesis routines.

Then, you can modify the demo scripts to suit your needs.

**NOTE:** Just remember to run the scripts from their locations.

## VII. Using MagPhase with the Merlin toolkit:
We provide two demos distributed with the Merlin's official distribution. These  show examples of the of Merlin with MagPhase integration:
* Text-To-Speech: [Merlin's slt_arctic demo](https://github.com/CSTR-Edinburgh/merlin/tree/master/egs/slt_arctic/s2) (small and full subset versions)

* Voice conversion: [Merlin's voice conversion demo](https://github.com/CSTR-Edinburgh/merlin/tree/master/egs/voice_conversion/s2) (roughly tested)
