#!/bin/bash

max_t_cost="$1"
max_m_cost="$2"
max_lanes="$3"

if [ -z "$max_t_cost" ]; then
    echo "ERROR: Maximum time cost must be specified!" 1>&2
    exit 1
fi

if [ -z "$max_m_cost" ]; then
    echo "ERROR: Maximum memory cost must be specified!" 1>&2
    exit 1
fi

if [ -z "$max_lanes" ]; then
    echo "ERROR: Maximum number of lanes must be specified!" 1>&2
    exit 1
fi

dirname="$(dirname "$0")"

cd "$dirname/.." || exit 1

echo "t_cost,m_cost,lanes,ms_i,ms_d,ms_id"
stdbuf -oL ./argon2-bench2 $max_t_cost $max_m_cost $max_lanes |
stdbuf -oL tail -n +2 |
while read line; do
    print_comma=0
    for x in $line; do
        if [ $print_comma -eq 1 ]; then
            echo -n ","
        else
            print_comma=1
        fi
        echo -n "$x"
    done
    echo
done
