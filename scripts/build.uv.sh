#!/bin/bash -e

UV_VERSION="1.38.0"

mkdir -p deps
mkdir -p deps/include
mkdir -p deps/lib

mkdir -p build && cd build

wget https://github.com/libuv/libuv/archive/v${UV_VERSION}.tar.gz -O v${UV_VERSION}.tar.gz
tar -xzf v${UV_VERSION}.tar.gz

cd libuv-${UV_VERSION}
sh autogen.sh
./configure --disable-shared
make -j$(nproc)
cp -fr include/ ../../deps
cp .libs/libuv.a ../../deps/lib
cd ..