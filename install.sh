#!/usr/bin/env sh
HOME=`pwd`

sudo apt-get update
sudo apt-get install python3-pybind11
sudo apt-get install ninja-build

# Chamfer Distance
cd extensions/chamfer_dist
rm -rf build/ dist/ *.so
python setup.py install --user
ln -s /home/ubuntu/.local/lib/python3.10/site-packages/chamfer-2.0.0-py3.10-linux-x86_64.egg/chamfer.cpython-310-x86_64-linux-gnu.so ./chamfer.so

# EMD
cd $HOME/extensions/emd
python setup.py install --user
