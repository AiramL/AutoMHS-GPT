#!/bin/bash
sudo apt-get install linux-headers-$(uname -r) -y
sudo add-apt-repository contrib
sudo apt-key del 7fa2af80
wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get --allow-releaseinfo-change update
sudo apt-get -y install cuda
python -m pip install --upgrade setuptools pip wheel
python -m pip install nvidia-pyindex
python -m pip install nvidia-cuda-runtime-cu12

