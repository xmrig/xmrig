/* RISC-V - test if the vector extension and prefetch instruction are present */

.text
.option arch, rv64gcv_zicbop
.global main

main:
	lla x5, main
	prefetch.r (x5)
	li x5, 4
	vsetvli x6, x5, e64, m1, ta, ma
	vxor.vv v0, v0, v0
	sub x10, x5, x6
	ret
