#!/bin/bash

apt install python3-pip python3-dev python-virtualenv -y

mkdir -p tensorflow
cd tensorflow
virtualenv --system-site-packages -p python3 venv

source ./venv/bin/activate
pip3 install --upgrade pip
pip3 install tensorflow==1.5

