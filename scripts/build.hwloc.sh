#!/bin/bash -e

HWLOC_VERSION="2.2.0"

mkdir -p deps
mkdir -p deps/include
mkdir -p deps/lib

mkdir -p build && cd build

wget https://download.open-mpi.org/release/hwloc/v2.2/hwloc-${HWLOC_VERSION}.tar.bz2 -O hwloc-${HWLOC_VERSION}.tar.bz2
tar -xjf hwloc-${HWLOC_VERSION}.tar.bz2

cd hwloc-${HWLOC_VERSION}
./configure --disable-shared --enable-static --disable-io --disable-libudev --disable-libxml2
make -j$(nproc)
cp -fr include/ ../../deps
cp hwloc/.libs/libhwloc.a ../../deps/lib
cd ..