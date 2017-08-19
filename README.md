# MagPhase Vocoder:
MagPhase Vocoder: Speech analysis/synthesis system intended for TTS that extracts and generates from magnitude and phase spectra as speech features.

More information at http://www.felipeespic.com/magphase/

## I. License:
See the LICENCE file for details.

## I. Prerequisites:
- OS: Linux (MacOSx soon)
- Python 2.7
- Standard Python packages: numpy, scipy, soundfile, matplotlib

## II. Install:
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


## III. Instructions:
Just go to ```/demos```, read the instructions inside the demo scripts, which are very discriptive.
They should run out of the box by running ```python <demo_script>```.

We recomend that you play firstly with ```demo_copy_synthesis_high_res.py``` , and then ```demo_copy_synthesis_low_dim.py```
They both perform analysis/synthesis routines.

**NOTE:** Just remember to run the scripts from their locations.

Then, you can modify the demo scripts to suit your needs.


## IV. Using with the Merling toolkit:
We provide demo scripts  in ```/demos/run_for_merlin```. Firstly, run the demos in order to learn how they work, and then you can adapt them to work with the Merlin toolkit and real data.

When working with real data and the Merlin toolkit, follow these steps:

1. Before training, run the scripts (with the paths pointing to your data): ```0_batch_feature_extraction_for_merlin.py``` , and then ```1_batch_convert_label_state_aligned_to_variable_frame_rate.py```

2. Modify the Merlin config file to use the extracted MagPhase features:
Use the features mag, real, imag, and lf0 as output parameters for the NN (instead of the defauls: mgc, bap, and lf0).

3. Modify the Merlin config file to use the variable rate label by pointing to the ```/labs_var_rate``` directory.

4. Run all the Merlin steps up to DNNGEN (Do not run WAVGEN).

5. Run the script ```2_batch_wave_generation.py``` to generate the waveforms from the parameters predicted by Merlin (usually stored in ```<experiment_dir>/data/gen/```)


