#!/usr/bin/env bash

. $MINER_DIR/$CUSTOM_MINER/h-manifest.conf

. $MINER_DIR/$CUSTOM_MINER/parse-api-port.sh

json_data=`curl -s http://localhost:${CUSTOM_API_PORT}`

algo=`echo $json_data | jq -c ".algo" | sed 's/"//g'`
hashrates=`echo $json_data | jq -c ".hashers[].hashrate[0]"`
shares=`echo $json_data | jq -c ".results.shares_good"`
totalShares=`echo $json_data | jq -c ".results.shares_total"`
total_hashrate=`echo $json_data | jq -c ".hashrate.total[0]"`
total_hashrate=`echo $total_hashrate/1000 | jq -nf /dev/stdin`
rejects=$((totalShares-shares))

gpu_data=`gpu-stats`
busids_data=`echo $gpu_data | jq -r ".busids[]"`
busids=($busids_data)
temp_data=`echo $gpu_data | jq -r ".temp[]"`
temp_local=($temp_data)
fan_data=`echo $gpu_data | jq -r ".fan[]"`
fan_local=($fan_data)
device_bus_data=`echo $json_data | jq -c ".hashers[].bus_id"`
device_bus=($device_bus_data)
stats_temp=""
stats_fan=""
bus_numbers=""
for i in "${!device_bus[@]}"; do
  found=0
  for j in "${!busids[@]}"; do
    if [ "${device_bus[$i],,}" == "\"${busids[$j],,}\"" ]; then
	stats_temp="$stats_temp ${temp_local[$j]}"
	stats_fan="$stats_fan ${fan_local[$j]}"
	bus_number=$(echo ${busids[$j]} | cut -d ':' -f 1 | awk '{printf("%d\n", "0x"$1)}')
	bus_numbers="$bus_numbers $bus_number"
        found=1
	break
    fi
  done
  if [ $found -eq 0 ]; then
    stats_temp="$stats_temp 0"
    stats_fan="$stats_fan 0"
    bus_numbers="$bus_numbers 0"
  fi
done

khs=$total_hashrate
hashrates=$hashrates
stats=$(jq -nc \
	--argjson hashrates "`echo "$hashrates" | tr " " "\n" | jq -cs '.'`" \
	--argjson hs "`echo "$hashrates" | tr " " "\n" | jq -cs '.'`" \
	--arg hs_units "hs" \
	--argjson temp "`echo "$stats_temp" | tr " " "\n" | jq -cs '.'`" \
	--argjson fan "`echo "$stats_fan" | tr " " "\n" | jq -cs '.'`" \
	--arg ac "$shares" --arg rj "$rejects" \
	--argjson bus_numbers "`echo "$bus_numbers" | tr " " "\n" | jq -cs '.'`" \
	--arg algo $algo \
	'{$hashrates, $hs, $hs_units, $temp, $fan, ar: [$ac, $rj], $bus_numbers, $algo}')
