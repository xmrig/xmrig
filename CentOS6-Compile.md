# CentOS6 installation guide

## prepare

+ install libuv1

```bash
wget https://github.com/libuv/libuv/archive/v1.23.2.tar.gz
tar zxvf v1.23.2.tar.gz
cd libuv-1.23.2
sh autogen.sh
./configure
make
make check # maybe ignored
make install
```

+ repalce all Ofast to O2 in $PROJECT/cmake/flags.cmake

+ e.g.

```bash
sed -i 's/Ofast/O2/g' ./cmake/flags.cmake
```

+ You must install gcc about 5.4+ (I've test 6.4)

## install

+ Cmake project using newest gcc with libuv

```bash
mkdir build
cd build
export CC=/usr/local/bin/gcc
export CXX=/usr/local/bin/g++
cmake .. -DUV_INCLUDE_DIR=/usr/local/include -DUV_LIBRARY=/usr/local/lib/libuv.a
make
```

+ It works