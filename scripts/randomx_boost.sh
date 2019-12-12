#!/bin/bash

modprobe msr

if cat /proc/cpuinfo | grep "AMD Ryzen" > /dev/null;
	then
		echo "Detected Ryzen"
		wrmsr -a 0xc0011022 0x510000
		wrmsr -a 0xc001102b 0x1808cc16
		wrmsr -a 0xc0011020 0
		wrmsr -a 0xc0011021 0x40
		echo "MSR register values for Ryzen applied"
elif cat /proc/cpuinfo | grep "Intel" > /dev/null;
	then
		echo "Detected Intel"
		wrmsr -a 0x1a4 6
		echo "MSR register values for Intel applied"
else
	echo "No supported CPU detected"
fi
