#!/bin/bash
yum update -y
yum install -y cmake make git openssl-devel libmicrohttpd-devel
rpm -i https://github.com/sipcapture/captagent/raw/master/dependency/centos/6/libuv-1.8.0-1.el6.x86_64.rpm
rpm -i https://github.com/sipcapture/captagent/raw/master/dependency/centos/6/libuv-devel-1.8.0-1.el6.x86_64.rpm
wget http://people.centos.org/tru/devtools-2/devtools-2.repo -O /etc/yum.repos.d/devtools-2.repo
yum upgrade -y
yum install -y devtoolset-2-gcc devtoolset-2-binutils devtoolset-2-gcc-c++

git checkout $1 &&\
scl enable devtoolset-2 "cmake ." &&\
scl enable devtoolset-2 "make" &&\
cp src/config.json . &&\
tar cfz xmrig-$1-lin64.tar.gz xmrig config.json
