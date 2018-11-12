	push	rbx
	push	rbp
	push	rsi
	push	rdi
	push	r12
	push	r13
	push	r14
	push	r15
	sub	rsp, 72

	movaps XMMWORD PTR [rsp], xmm6
	movaps XMMWORD PTR [rsp+16], xmm7
	movaps XMMWORD PTR [rsp+32], xmm8
	movaps XMMWORD PTR [rsp+48], xmm9

	mov	rax, QWORD PTR [rcx+48]
	xor	rax, QWORD PTR [rcx+16]
	mov	rdx, QWORD PTR [rcx+56]
	xor	rdx, QWORD PTR [rcx+24]
	mov	r8, QWORD PTR [rcx+32]
	xor	r8, QWORD PTR [rcx]
	movq	xmm4, rax
	mov	rax, QWORD PTR [rcx+256]
	mov	r13, QWORD PTR [rcx+40]
	movq	xmm0, rdx
	xor	r13, QWORD PTR [rcx+8]
	mov	rdx, r8
	mov	rdi, QWORD PTR [rcx+224]
	and	edx, 2097136
	mov	rax, QWORD PTR [rax+35]
	xor	rax, QWORD PTR [rcx+192]
	movq	xmm5, rax
	movq	xmm8, rdi
	punpcklqdq xmm4, xmm0
	mov	QWORD PTR [rsp+64], rdx

	movq xmm6, rcx
	mov rax, QWORD PTR [rcx+264]
	movq xmm7, rax

	mov eax, 262144

	#ifdef __APPLE__
	ALIGN 16
	#else
	ALIGN 64
	#endif
cn_fast_mainloop_soft_aes_sandybridge:
	movq xmm9, rax
	mov	r12, QWORD PTR [rcx+272]
	mov	esi, DWORD PTR [rdx+rdi]
	mov	r10d, DWORD PTR [rdx+rdi+4]
	mov	ebp, DWORD PTR [rdx+rdi+12]
	mov	r14d, DWORD PTR [rdx+rdi+8]
	mov	rdx, QWORD PTR [rsp+64]
	movzx	ecx, sil
	shr	esi, 8
	mov	r15d, DWORD PTR [r12+rcx*4]
	movzx	ecx, r10b
	shr	r10d, 8
	mov	edi, DWORD PTR [r12+rcx*4]
	movzx	ecx, r14b
	shr	r14d, 8
	mov	ebx, DWORD PTR [r12+rcx*4]
	movzx	ecx, bpl
	shr	ebp, 8
	mov	r9d, DWORD PTR [r12+rcx*4]
	movzx	ecx, r10b
	shr	r10d, 8
	xor	r15d, DWORD PTR [r12+rcx*4+1024]
	movzx	ecx, r14b
	shr	r14d, 8
	mov	eax, r14d
	shr	eax, 8
	xor	edi, DWORD PTR [r12+rcx*4+1024]
	add	eax, 256
	movzx	ecx, bpl
	shr	ebp, 8
	xor	ebx, DWORD PTR [r12+rcx*4+1024]
	movzx	ecx, sil
	shr	esi, 8
	xor	r9d, DWORD PTR [r12+rcx*4+1024]
	add	r12, 2048
	movzx	ecx, r10b
	shr	r10d, 8
	add	r10d, 256
	mov	r11d, DWORD PTR [r12+rax*4]
	xor	r11d, DWORD PTR [r12+rcx*4]
	xor	r11d, r9d
	movzx	ecx, sil
	mov	r10d, DWORD PTR [r12+r10*4]
	shr	esi, 8
	add	esi, 256
	xor	r10d, DWORD PTR [r12+rcx*4]
	movzx	ecx, bpl
	xor	r10d, ebx
	shr	ebp, 8
	add	ebp, 256
	movd	xmm1, r11d
	mov	r9d, DWORD PTR [r12+rcx*4]
	xor	r9d, DWORD PTR [r12+rsi*4]
	mov	eax, DWORD PTR [r12+rbp*4]
	xor	r9d, edi
	movq rdi, xmm8
	movzx	ecx, r14b
	movd	xmm0, r10d
	movd	xmm2, r9d
	punpckldq xmm2, xmm1
	movq	xmm1, r8
	xor	eax, DWORD PTR [r12+rcx*4]
	xor	eax, r15d
	movd	xmm3, eax
	movq	rax, xmm7
	punpckldq xmm3, xmm0
	movq	xmm0, r13
	punpcklqdq xmm1, xmm0
	punpckldq xmm3, xmm2
	pxor	xmm3, xmm1
	movq	r9, xmm3
	mov	r10d, r9d
	and	r10d, 2097136
	movdqa	xmm0, xmm3
	pxor	xmm0, xmm4
	movdqu	XMMWORD PTR [rdx+rdi], xmm0
	psrldq	xmm0, 11
	movq	rcx, xmm0
	movzx	ecx, cl
	mov	cl, BYTE PTR [rcx+rax]
	mov	BYTE PTR [rdi+rdx+11], cl
	mov	rbx, QWORD PTR [r10+rdi]
	mov rcx, r9
	lea	r9, QWORD PTR [r10+rdi]
	mov	r11, QWORD PTR [r9+8]
	mov	rax, rbx
	movdqa	xmm4, xmm3
	mul	rcx
	movq	rcx, xmm6
	add	r8, rdx
	add	r13, rax
	movq	rax, xmm5
	xor	rax, r13
	mov	QWORD PTR [r9], r8
	xor	r8, rbx
	mov	QWORD PTR [r9+8], rax
	movq rax, xmm9
	mov	rdx, r8
	xor	r13, r11
	and	edx, 2097136
	mov	QWORD PTR [rsp+64], rdx
	sub eax, 1
	jne	cnv1_mainloop_soft_aes_sandybridge

	movaps xmm6, XMMWORD PTR [rsp]
	movaps xmm7, XMMWORD PTR [rsp+16]
	movaps xmm8, XMMWORD PTR [rsp+32]
	movaps xmm9, XMMWORD PTR [rsp+48]

	add	rsp, 72
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rdi
	pop	rsi
	pop	rbp
	pop	rbx
