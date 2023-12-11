#!/bin/bash

mkdir ../datasets/VeReMi_Extension
cd ../datasets/VeReMi_Extension
wget https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/k62n4z9gdz-1.zip
unzip k62n4z9gdz-1.zip
tar zvxf mixalldata_clean.tar 
