#!/bin/sh -e

# https://xmrig.com/docs/miner/hugepages#onegb-huge-pages

sysctl -w vm.nr_hugepages="$(nproc)"

find /sys/devices/system/node/node* -maxdepth 0 -type d -exec sh -c 'echo 3 > "$1/hugepages/hugepages-1048576kB/nr_hugepages"' _ {} \;

echo "1GB pages successfully enabled"
