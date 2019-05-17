#!/bin/bash
#
# Build MIGraphX and MIGraphX driver components
MIGRAPHX_DIR=/src/AMDMIGraphX
MIGRAPHX_DRIVER_DIR=/src/migraphx_sample/migraphx_driver

# Not sure why but rbuild gets confused with protobuf, so try building some
# prerequisites manually
cd /src/
git clone https://github.com/pybind/pybind11
pip3 install pytest
cd pybind11
git checkout v2.2.4
mkdir build
cd build
cmake ..
make -j
make install

cd /src/
git clone https://github.com/protocolbuffers/protobuf
cd protobuf
git checkout v3.2.1
git submodule update --init --recursive
./autogen.sh
./configure
make -j
make install

cd /src
wget https://bitbucket.org/blaze-lib/blaze/downloads/blaze-3.5.tar.gz
tar xf blaze-3.5.tar.gz
cd blaze-3.5
cp -r blaze /usr/local/include
rm blaze-3.5.tar.gz

echo "Building MIGraphX"
cd $MIGRAPHX_DIR

# rbuild gets confused about protobuf...
#rbuild build -d depend --cxx=/opt/rocm/bin/hcc 2>&1 | tee build.log
mkdir build
cd build
env CXX=/opt/rocm/bin/hcc CXXFLAGS="-O3" cmake ..
make -j

echo "Building migraphx_driver"
cd $MIGRAPHX_DRIVER_DIR
mkdir build
cd build
cmake ..
make
