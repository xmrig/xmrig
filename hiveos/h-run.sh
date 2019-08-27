#!/usr/bin/env bash

. h-manifest.conf

CUSTOM_API_PORT=`./parse-api-port.sh`

#try to release TIME_WAIT sockets
while true; do
	for con in `netstat -anp | grep TIME_WAIT | grep ${CUSTOM_API_PORT} | awk '{print $5}'`; do
		killcx $con lo
	done
	netstat -anp | grep TIME_WAIT | grep ${CUSTOM_API_PORT} &&
		continue ||
		break
done

mkdir -p $CUSTOM_LOG_FOLDER
echo -e "Running ${CYAN}iximiner${NOCOLOR}" | tee ${CUSTOM_LOG_FOLDER}/${CUSTOM_NAME}.log

./ninjarig $(< $CUSTOM_NAME.conf)$@ 2>&1 | tee ${CUSTOM_LOG_FOLDER}/${CUSTOM_NAME}.log
