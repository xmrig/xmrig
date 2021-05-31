#!/bin/bash

MSR_FILE=/sys/module/msr/parameters/allow_writes

if test -e "$MSR_FILE"; then
	echo on > $MSR_FILE
else
	modprobe msr allow_writes=on
fi

if cat /proc/cpuinfo | grep -E 'AMD Ryzen|AMD EPYC' > /dev/null;
	then
	if cat /proc/cpuinfo | grep "cpu family[[:space:]]:[[:space:]]25" > /dev/null;
		then
			echo "Detected Zen3 CPU"
			wrmsr -a 0xc0011020 0x4480000000000
			wrmsr -a 0xc0011021 0x1c000200000040
			wrmsr -a 0xc0011022 0xc000000401500000
			wrmsr -a 0xc001102b 0x2000cc14
			echo "MSR register values for Zen3 applied"
		else
			echo "Detected Zen1/Zen2 CPU"
			wrmsr -a 0xc0011020 0
			wrmsr -a 0xc0011021 0x40
			wrmsr -a 0xc0011022 0x1510000
			wrmsr -a 0xc001102b 0x2000cc16
			echo "MSR register values for Zen1/Zen2 applied"
		fi
elif cat /proc/cpuinfo | grep "Intel" > /dev/null;
	then
		echo "Detected Intel CPU"
		wrmsr -a 0x1a4 0xf
		echo "MSR register values for Intel applied"
else
	echo "No supported CPU detected"
fi
