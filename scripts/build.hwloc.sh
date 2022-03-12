#!/bin/bash -e

HWLOC_VERSION="2.7.0"
HWLOC_V="v2.7"

mkdir -p deps
mkdir -p deps/include
mkdir -p deps/lib

mkdir -p build && cd build

wget https://download.open-mpi.org/release/hwloc/$HWLOC_V/hwloc-${HWLOC_VERSION}.tar.gz -O hwloc-${HWLOC_VERSION}.tar.gz
tar -xzf hwloc-${HWLOC_VERSION}.tar.gz

cd hwloc-${HWLOC_VERSION}
./configure --disable-shared --enable-static --disable-io --disable-libudev --disable-libxml2
make -j$(nproc || sysctl -n hw.ncpu || sysctl -n hw.logicalcpu)
cp -fr include ../../deps
cp hwloc/.libs/libhwloc.a ../../deps/lib
cd ..
