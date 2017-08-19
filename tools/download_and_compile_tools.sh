#!/bin/bash

# 1. Get and compile SPTK
echo "Downloading SPTK-3.9:======================================================="
wget http://downloads.sourceforge.net/sp-tk/SPTK-3.9.tar.gz
tar xzf SPTK-3.9.tar.gz
rm SPTK-3.9.tar.gz

echo "Compiling SPTK:============================================================="
(
    cd SPTK-3.9;
    ./configure --prefix=$PWD/build;
    make;
    make install
)

# Get and compile REAPER:
echo "Downloading REAPER:========================================================"
git clone https://github.com/google/REAPER.git

echo "Compiling REAPER:=========================================================="
cd REAPER
mkdir build   # In the REAPER top-level directory
cd build
cmake ..
make

echo "All tools successfully compiled!!"
