/* RISC-V - test if the vector bit manipulation extension is present */

.text
.option arch, rv64gcv_zvkned
.global main

main:
	vsetivli zero, 8, e32, m1, ta, ma
	vaesem.vv v0, v0
	vaesdm.vv v0, v0
	li x10, 0
	ret
