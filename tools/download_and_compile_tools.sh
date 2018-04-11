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
    make -j;
    make install
)

# Get and compile REAPER:
echo "Downloading REAPER:========================================================"
git clone https://github.com/google/REAPER.git

echo "Compiling REAPER:=========================================================="
(
    cd REAPER
    mkdir build   # In the REAPER top-level directory
    cd build
    cmake ..
    make
)
# Remove unnecessary files:
echo "Removing temporary files:=================================================="
mkdir -p ./bin
cp ./REAPER/build/reaper ./bin

cp ./SPTK-3.9/build/bin/mcep ./bin
cp ./SPTK-3.9/build/bin/x2x ./bin
cp ./SPTK-3.9/build/bin/freqt ./bin
cp ./SPTK-3.9/build/bin/c2acr ./bin
cp ./SPTK-3.9/build/bin/vopr ./bin
cp ./SPTK-3.9/build/bin/mc2b ./bin
cp ./SPTK-3.9/build/bin/bcp ./bin
cp ./SPTK-3.9/build/bin/sopr ./bin
cp ./SPTK-3.9/build/bin/merge ./bin
cp ./SPTK-3.9/build/bin/b2mc ./bin

rm -rf ./REAPER
rm -rf ./SPTK-3.9


echo "All tools successfully compiled!"
