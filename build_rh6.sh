#!/bin/bash
yum update -y
yum install -y cmake make git
rpm -i https://github.com/sipcapture/captagent/raw/master/dependency/centos/6/libuv-1.8.0-1.el6.x86_64.rpm
rpm -i https://github.com/sipcapture/captagent/raw/master/dependency/centos/6/libuv-devel-1.8.0-1.el6.x86_64.rpm
wget http://people.centos.org/tru/devtools-2/devtools-2.repo -O /etc/yum.repos.d/devtools-2.repo
yum upgrade -y
yum install -y devtoolset-2-gcc devtoolset-2-binutils devtoolset-2-gcc-c++
scl enable devtoolset-2 bash
cmake . -DWITH_HTTPD=OFF
