	xor eax, eax
	pop rcx
	mov qword ptr [rcx+0], r8
	mov qword ptr [rcx+8], r9
	mov qword ptr [rcx+16], r10
	mov qword ptr [rcx+24], r11
	mov qword ptr [rcx+32], r12
	mov qword ptr [rcx+40], r13
	mov qword ptr [rcx+48], r14
	mov qword ptr [rcx+56], r15
	pop rcx
	xorpd xmm0, xmm4
	xorpd xmm1, xmm5
	xorpd xmm2, xmm6
	xorpd xmm3, xmm7
	movapd xmmword ptr [rcx+0], xmm0
	movapd xmmword ptr [rcx+16], xmm1
	movapd xmmword ptr [rcx+32], xmm2
	movapd xmmword ptr [rcx+48], xmm3
