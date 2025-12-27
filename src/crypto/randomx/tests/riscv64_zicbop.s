/* RISC-V - test if the prefetch instruction is present */

.text
.option arch, rv64gc_zicbop
.global main

main:
	lla x5, main
	prefetch.r (x5)
	mv x10, x0
	ret
