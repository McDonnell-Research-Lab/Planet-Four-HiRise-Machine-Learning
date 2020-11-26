#!/usr/bin/env bash

conda init
source ~/.bashrc
conda create -n HiRISE_ML python=3.7 -y

eval "$(conda shell.bash hook)"

source ~/anaconda3/etc/profile.d/conda.sh
conda activate HiRISE_ML

conda install -y -c anaconda jupyter 
pip install opencv-python
pip install "Glymur==0.9.1"
conda install -y -c anaconda openjpeg
conda install -y -c anaconda scikit-learn
conda install -y -c michaelaye planetfour-catalog
pip install p4tools

conda install -y -c anaconda tensorflow-gpu==2.1