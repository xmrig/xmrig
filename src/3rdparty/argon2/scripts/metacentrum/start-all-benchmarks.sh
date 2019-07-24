#!/bin/bash

dirname="$(dirname "$0")"

cd "$dirname" || exit 1

./start-benchmark.sh luna
./start-benchmark.sh lex '' '' '' '' '' backfill
./start-benchmark.sh mandos
./start-benchmark.sh zubat
PBS_SERVER=wagap.cerit-sc.cz \
    ./start-benchmark.sh zapat '' '' '' '' '' default@wagap.cerit-sc.cz
