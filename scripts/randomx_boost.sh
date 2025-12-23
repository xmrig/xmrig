#!/bin/sh -e

MSR_FILE=/sys/module/msr/parameters/allow_writes

if test -e "$MSR_FILE"; then
	echo on > $MSR_FILE
else
	modprobe msr allow_writes=on
fi

if grep -E 'AMD Ryzen|AMD EPYC|AuthenticAMD' /proc/cpuinfo > /dev/null;
	then
	if grep "cpu family[[:space:]]\{1,\}:[[:space:]]25" /proc/cpuinfo > /dev/null;
		then
			if grep "model[[:space:]]\{1,\}:[[:space:]]\(97\|117\)" /proc/cpuinfo > /dev/null;
				then
					echo "Detected Zen4 CPU"
					wrmsr -a 0xc0011020 0x4400000000000
					wrmsr -a 0xc0011021 0x4000000000040
					wrmsr -a 0xc0011022 0x8680000401570000
					wrmsr -a 0xc001102b 0x2040cc10
					echo "MSR register values for Zen4 applied"
				else
					echo "Detected Zen3 CPU"
					wrmsr -a 0xc0011020 0x4480000000000
					wrmsr -a 0xc0011021 0x1c000200000040
					wrmsr -a 0xc0011022 0xc000000401570000
					wrmsr -a 0xc001102b 0x2000cc10
					echo "MSR register values for Zen3 applied"
				fi
		elif grep "cpu family[[:space:]]\{1,\}:[[:space:]]26" /proc/cpuinfo > /dev/null;
			then
				echo "Detected Zen5 CPU"
				wrmsr -a 0xc0011020 0x4400000000000
				wrmsr -a 0xc0011021 0x4000000000040
				wrmsr -a 0xc0011022 0x8680000401570000
				wrmsr -a 0xc001102b 0x2040cc10
				echo "MSR register values for Zen5 applied"
		else
			echo "Detected Zen1/Zen2 CPU"
			wrmsr -a 0xc0011020 0
			wrmsr -a 0xc0011021 0x40
			wrmsr -a 0xc0011022 0x1510000
			wrmsr -a 0xc001102b 0x2000cc16
			echo "MSR register values for Zen1/Zen2 applied"
		fi
elif grep "Intel" /proc/cpuinfo > /dev/null;
	then
		echo "Detected Intel CPU"
		wrmsr -a 0x1a4 0xf
		echo "MSR register values for Intel applied"
else
	echo "No supported CPU detected"
fi
