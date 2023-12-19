#!/bin/bash 

conda install gxx_linux-64 gcc_linux-64 swig
conda config --add channels conda-forge
conda config --set channel_priority strict
conda install auto-sklearn

