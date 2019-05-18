#!/bin/bash
#
# This script asks configuration information to build a Dockerfile for
# building running MIGraphX + sample programs.
DOCKERFILE=${DOCKERFILE:="Dockerfile"}

if [ -f ${DOCKERFILE} ]; then
    echo File ${DOCKERFILE} exists, please move out of way before running this script.
    exit 1
fi

read -p "Enter dockerfile base [rocm/dev-ubuntu-16.04:2.3]: " docker_base
docker_base=${docker_base:="rocm/dev-ubuntu-16.04:2.3"}
echo "FROM $docker_base" > ${DOCKERFILE}

# System pre-requisites
cat >> $DOCKERFILE <<EOF
RUN apt update && apt install -y git cmake python-pip python3-pip zlib1g-dev unzip autogen autoconf libtool wget
RUN apt update && apt install -y libnuma-dev rocm-cmake rocm-libs miopen-hip
RUN apt update && apt install -y libopencv-dev
RUN pip install https://github.com/RadeonOpenCompute/rbuild/archive/master.tar.gz
RUN pip install cget
EOF

# Copy in MIGraphX and migraphx_sample directories
cat >> $DOCKERFILE <<EOF
RUN mkdir /src
RUN cd /src && git clone https://github.com/ROCmSoftwarePlatform/AMDMIGraphX
RUN cd /src && git clone https://github.com/mvermeulen/migraphx_sample 
ENV LD_LIBRARY_PATH=/usr/local/lib:
COPY copyfiles/build.sh /src
COPY copyfiles/include/half.hpp /usr/local/include/
EOF

# Copy in PyTorch/Tensorflow components needed for simple graphs
# NOTE: Could also install AMD Versions rather than CPU versions...
cat >> $DOCKERFILE <<EOF
RUN pip3 install torch torchvision
RUN pip3 install tensorflow
EOF
