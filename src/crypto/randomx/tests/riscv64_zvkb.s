/* RISC-V - test if the vector bit manipulation extension is present */

.text
.option arch, rv64gcv_zvkb
.global main

main:
	vsetivli zero, 8, e32, m1, ta, ma
	vror.vv v0, v0, v0
	vror.vx v0, v0, x5
	vror.vi v0, v0, 1
	li x10, 0
	ret
