#!/bin/bash -e

UV_VERSION="1.34.0"
OPENSSL_VERSION="1.1.1d"
HWLOC_VERSION="2.1.0"

mkdir deps
mkdir deps/include
mkdir deps/lib

mkdir build && cd build

wget https://github.com/libuv/libuv/archive/v${UV_VERSION}.tar.gz
tar -xzf v${UV_VERSION}.tar.gz

wget https://download.open-mpi.org/release/hwloc/v2.1/hwloc-${HWLOC_VERSION}.tar.bz2
tar -xjf hwloc-${HWLOC_VERSION}.tar.bz2

wget https://www.openssl.org/source/openssl-${OPENSSL_VERSION}.tar.gz
tar -xzf openssl-${OPENSSL_VERSION}.tar.gz

cd libuv-${UV_VERSION}
sh autogen.sh
./configure --disable-shared
make -j$(nproc)
cp -fr include/ ../../deps
cp .libs/libuv.a ../../deps/lib
cd ..

cd hwloc-${HWLOC_VERSION}
./configure --disable-shared --enable-static --disable-io --disable-libudev --disable-libxml2
make -j$(nproc)
cp -fr include/ ../../deps
cp hwloc/.libs/libhwloc.a ../../deps/lib
cd ..

cd openssl-${OPENSSL_VERSION}
./config -no-shared -no-asm -no-zlib -no-comp -no-dgram -no-filenames -no-cms
make -j$(nproc)
cp -fr include/ ../../deps
cp libcrypto.a ../../deps/lib
cp libssl.a ../../deps/lib
cd ../..