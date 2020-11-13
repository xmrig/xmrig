#!/bin/bash

modprobe msr

if cat /proc/cpuinfo | grep "AMD Ryzen" > /dev/null;
	then
	if cat /proc/cpuinfo | grep "cpu family[[:space:]]:[[:space:]]25" > /dev/null;
		then
			echo "Detected Ryzen (Zen3)"
			wrmsr -a 0xc0011020 0x4480000000000
			wrmsr -a 0xc0011021 0x1c000200000040
			wrmsr -a 0xc0011022 0xc000000401500000
			wrmsr -a 0xc001102b 0x2000cc14
			echo "MSR register values for Ryzen (Zen3) applied"
		else
			echo "Detected Ryzen (Zen1/Zen2)"
			wrmsr -a 0xc0011020 0
			wrmsr -a 0xc0011021 0x40
			wrmsr -a 0xc0011022 0x1510000
			wrmsr -a 0xc001102b 0x2000cc16
			echo "MSR register values for Ryzen (Zen1/Zen2) applied"
		fi
elif cat /proc/cpuinfo | grep "Intel" > /dev/null;
	then
		echo "Detected Intel"
		wrmsr -a 0x1a4 0xf
		echo "MSR register values for Intel applied"
else
	echo "No supported CPU detected"
fi
