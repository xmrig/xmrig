#!/bin/sh -e

OPENSSL_VERSION="3.5.9"

mkdir -p deps
mkdir -p deps/include
mkdir -p deps/lib

mkdir -p build && cd build

wget https://github.com/openssl/openssl/releases/download/openssl-${OPENSSL_VERSION}/openssl-${OPENSSL_VERSION}.tar.gz -O openssl-${OPENSSL_VERSION}.tar.gz
tar -xzf openssl-${OPENSSL_VERSION}.tar.gz

cd openssl-${OPENSSL_VERSION}
./config no-shared no-module no-dso no-pinshared no-apps no-tests no-docs no-makedepend \
    no-legacy no-engine no-autoload-config no-ssl3 no-dtls no-srp no-comp no-quic \
    no-aria no-bf no-blake2 no-camellia no-cast no-des no-idea no-md4 no-mdc2 no-ocb no-rc2 no-rc4 \
    no-rmd160 no-seed no-siv no-sm2 no-sm3 no-sm4 no-whirlpool no-argon2 no-scrypt no-ec2m no-dsa \
    no-gost no-ml-dsa no-slh-dsa no-cms no-ts no-ct no-cmp no-srtp no-rfc3779 no-http no-ui-console \
    no-integrity-only-ciphers no-tls-deprecated-ec no-multiblock \
    -ffunction-sections -fdata-sections -O2
make -j$(nproc || sysctl -n hw.ncpu || sysctl -n hw.logicalcpu)
cp -fr include ../../deps
cp libcrypto.a ../../deps/lib
cp libssl.a ../../deps/lib
cd ..
