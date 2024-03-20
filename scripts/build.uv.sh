#!/bin/sh -e

UV_VERSION="1.48.0"

mkdir -p deps
mkdir -p deps/include
mkdir -p deps/lib

mkdir -p build && cd build

wget https://dist.libuv.org/dist/v${UV_VERSION}/libuv-v${UV_VERSION}.tar.gz -O v${UV_VERSION}.tar.gz
tar -xzf v${UV_VERSION}.tar.gz

cd libuv-v${UV_VERSION}
sh autogen.sh
./configure --disable-shared
make -j$(nproc || sysctl -n hw.ncpu || sysctl -n hw.logicalcpu)
cp -fr include ../../deps
cp .libs/libuv.a ../../deps/lib
cd ..
