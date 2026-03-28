	mov rcx, [rsp+24]
	mov qword ptr [rcx+0], r8
	mov qword ptr [rcx+8], r9
	mov qword ptr [rcx+16], r10
	mov qword ptr [rcx+24], r11
	mov qword ptr [rcx+32], r12
	mov qword ptr [rcx+40], r13
	mov qword ptr [rcx+48], r14
	mov qword ptr [rcx+56], r15
	mov rcx, [rsp+16]
	aesenc xmm0, xmm4
	aesdec xmm1, xmm4
	aesenc xmm2, xmm4
	aesdec xmm3, xmm4
	aesenc xmm0, xmm5
	aesdec xmm1, xmm5
	aesenc xmm2, xmm5
	aesdec xmm3, xmm5
	aesenc xmm0, xmm6
	aesdec xmm1, xmm6
	aesenc xmm2, xmm6
	aesdec xmm3, xmm6
	aesenc xmm0, xmm7
	aesdec xmm1, xmm7
	aesenc xmm2, xmm7
	aesdec xmm3, xmm7
	movapd xmmword ptr [rcx+0], xmm0
	movapd xmmword ptr [rcx+16], xmm1
	movapd xmmword ptr [rcx+32], xmm2
	movapd xmmword ptr [rcx+48], xmm3
