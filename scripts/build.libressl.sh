#!/bin/sh -e

LIBRESSL_VERSION="3.5.2"

mkdir -p deps
mkdir -p deps/include
mkdir -p deps/lib

mkdir -p build && cd build

wget https://ftp.openbsd.org/pub/OpenBSD/LibreSSL/libressl-${LIBRESSL_VERSION}.tar.gz -O libressl-${LIBRESSL_VERSION}.tar.gz
tar -xzf libressl-${LIBRESSL_VERSION}.tar.gz

cd libressl-${LIBRESSL_VERSION}
./configure --disable-shared
make -j$(nproc || sysctl -n hw.ncpu || sysctl -n hw.logicalcpu)
cp -fr include ../../deps
cp crypto/.libs/libcrypto.a ../../deps/lib
cp ssl/.libs/libssl.a ../../deps/lib
cd ..
