#!/usr/bin/env bash

if [ -z "$MINER_DIR" ]
then
. ./h-manifest.conf
args=$(getopt -q -u -l "api-port:" -o "l:" -- `cat ./$CUSTOM_NAME.conf`)
else
. $MINER_DIR/$CUSTOM_MINER/h-manifest.conf
args=$(getopt -q -u -l "api-port:" -o "l:" -- `cat $MINER_DIR/$CUSTOM_MINER/$CUSTOM_NAME.conf`)
fi

eval set -- "$args"

while [ $# -ge 1 ]; do
        case "$1" in
                --)
                shift
                break
                ;;
                --api-port)
                CUSTOM_API_PORT="$2"
                shift
                ;;
        esac
        shift
done

if [ -z "$MINER_DIR" ]
then
echo $CUSTOM_API_PORT
fi
