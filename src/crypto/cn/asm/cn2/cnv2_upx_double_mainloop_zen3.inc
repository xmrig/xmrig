	mov	rax, rsp
	push	rbx
	push	rbp
	push	rsi
	push	rdi
	push	r12
	push	r13
	push	r14
	push	r15
	sub	rsp, 232

	mov	rdi, QWORD PTR [rcx+8]

	mov	edx, 768
	mov	rbx, QWORD PTR [rcx]
	mov	ecx, 256
	movaps	XMMWORD PTR [rax-88], xmm6
	movaps	XMMWORD PTR [rax-104], xmm7
	mov	r13, QWORD PTR [rdi+224]
	movq	xmm0, QWORD PTR [rdi+104]
	mov	r12, QWORD PTR [rbx+224]
	movaps	XMMWORD PTR [rax-120], xmm8
	movaps	XMMWORD PTR [rax-136], xmm9
	movaps	XMMWORD PTR [rax-152], xmm10
	movaps	XMMWORD PTR [rsp+112], xmm11
	movaps	XMMWORD PTR [rsp+96], xmm12
	movaps	XMMWORD PTR [rsp+80], xmm13
	movq	xmm13, QWORD PTR [rbx+96]
	movaps	XMMWORD PTR [rsp+64], xmm14
	movq	xmm14, QWORD PTR [rbx+104]
	movaps	XMMWORD PTR [rsp+48], xmm15
	movq	xmm15, QWORD PTR [rdi+96]
	mov	QWORD PTR [rsp], r13
	movdqa	XMMWORD PTR [rsp+32], xmm0

	stmxcsr DWORD PTR [rsp+24]
	mov DWORD PTR [rsp+28], 16256
	ldmxcsr DWORD PTR [rsp+28]

	mov	rcx, QWORD PTR [rbx+56]
	xorps	xmm12, xmm12
	xor	rcx, QWORD PTR [rbx+24]
	mov	rax, QWORD PTR [rbx+48]
	xor	rax, QWORD PTR [rbx+16]
	mov	rsi, QWORD PTR [rbx+32]
	mov	rbp, QWORD PTR [rdi+32]
	movq	xmm0, rcx

	mov	rcx, QWORD PTR [rbx+88]
	xor	rcx, QWORD PTR [rbx+72]
	movq	xmm7, rax
	mov	rax, QWORD PTR [rbx+80]
	xor	rax, QWORD PTR [rbx+64]
	mov	r14, QWORD PTR [rbx+40]
	mov	r15, QWORD PTR [rdi+40]
	xor	rsi, QWORD PTR [rbx]
	xor	rbp, QWORD PTR [rdi]
	movq	xmm9, rax

	mov	rax, QWORD PTR [rdi+48]
	xor	rax, QWORD PTR [rdi+16]
	xor	r14, QWORD PTR [rbx+8]
	xor	r15, QWORD PTR [rdi+8]
	movq	xmm8, rax
	punpcklqdq xmm7, xmm0

	mov eax, 1023
	shl rax, 52
	movq xmm11, rax
	punpcklqdq xmm11, xmm11

	mov	rax, QWORD PTR [rdi+80]
	movq	xmm0, rcx
	mov	rcx, QWORD PTR [rdi+56]
	xor	rcx, QWORD PTR [rdi+24]
	punpcklqdq xmm9, xmm0
	mov	QWORD PTR [rsp+8], 16384
	movq	xmm0, rcx
	mov	rcx, QWORD PTR [rdi+88]
	xor	rcx, QWORD PTR [rdi+72]
	xor	rax, QWORD PTR [rdi+64]
	punpcklqdq xmm8, xmm0
	movq	xmm0, rcx
	movq	xmm10, rax
	mov	rax, 4389456576511
	mov	QWORD PTR [rsp+16], rax
	mov	rax, -4389456576512
	mov	QWORD PTR [rsp+216], rax
	punpcklqdq xmm10, xmm0

	ALIGN(64)
upx2_main_loop:
	mov	rdx, rsi
	mov	r9, rbp
	and	edx, 131056
	and	r9d, 131056
	movdqu	xmm6, XMMWORD PTR [rdx+r12]
	lea	r8, QWORD PTR [rdx+r12]
	movdqu	xmm4, XMMWORD PTR [r9+r13]
	lea	r10, QWORD PTR [r9+r13]
	mov	ecx, edx
	mov	eax, edx
	xor	rax, 32
	xor	rcx, 48
	xor	rdx, 16
	movq	xmm0, r14
	movq	xmm3, rsi
	movq	xmm5, rbp
	punpcklqdq xmm3, xmm0
	movq	xmm0, r15
	movdqu	xmm2, XMMWORD PTR [rax+r12]
	movdqu	xmm1, XMMWORD PTR [rcx+r12]
	paddq	xmm2, xmm3
	punpcklqdq xmm5, xmm0
	paddq	xmm1, xmm7
	aesenc	xmm6, xmm3
	aesenc	xmm4, xmm5
	movdqa	xmm0, xmm9
	movq	rdi, xmm4
	paddq	xmm0, XMMWORD PTR [rdx+r12]
	movdqu	XMMWORD PTR [rdx+r12], xmm0
	xor	edx, edx
	movdqu	XMMWORD PTR [rax+r12], xmm1
	movdqa	xmm0, xmm6
	movdqu	XMMWORD PTR [rcx+r12], xmm2
	pxor	xmm0, xmm7
	movdqu	XMMWORD PTR [r8], xmm0
	mov	ecx, r9d
	xor	rcx, 48
	mov	eax, r9d
	xor	rax, 32
	xor	r9, 16
	movdqa	xmm0, xmm10
	movdqu	xmm1, XMMWORD PTR [rcx+r13]
	movdqu	xmm2, XMMWORD PTR [rax+r13]
	paddq	xmm1, xmm8
	paddq	xmm0, XMMWORD PTR [r9+r13]
	paddq	xmm2, xmm5
	movdqu	XMMWORD PTR [r9+r13], xmm0
	movq	r9, xmm6
	movdqu	XMMWORD PTR [rax+r13], xmm1
	movdqa	xmm0, xmm4
	movdqu	XMMWORD PTR [rcx+r13], xmm2
	pxor	xmm0, xmm8
	movdqu	XMMWORD PTR [r10], xmm0
	movq	rcx, xmm14
	mov	rax, rcx
	movq	r10, xmm13
	shl	rax, 32
	movdqa	xmm0, xmm6
	xor	r10, rax
	psrldq	xmm0, 8
	lea	r8, QWORD PTR [rcx+rcx]
	movq	rax, xmm0
	add	r8d, r9d
	mov	ecx, -2147483647
	or	r8, rcx
	mov	r11, r9
	div	r8
	and	r11d, 131056
	movaps	xmm1, xmm12
	mov	eax, eax
	add	r11, r12
	shl	rdx, 32
	add	rdx, rax
	xor	r10, QWORD PTR [r11]
	mov	rbx, QWORD PTR [r11+8]
	lea	r8, QWORD PTR [rdx+r9]
	movq	xmm13, rdx
	mov	rax, r8
	shr	rax, 12
	movq	xmm0, rax
	paddq	xmm0, xmm11
	sqrtsd	xmm1, xmm0
	movq	rdx, xmm1
	mov	rax, rdx
	shr	rdx, 19
	shr	rax, 20
	mov	rcx, rdx
	sub	rcx, rax
	add	rax, QWORD PTR [rsp+216]
	sub	rcx, QWORD PTR [rsp+16]
	mov	r13, QWORD PTR [rsp]
	imul	rcx, rax
	mov	rax, r10
	sub	rcx, r8
	mov	rcx, r9
	adc	rdx, 0
	xor	rcx, 32
	and	ecx, 131056
	movq	xmm14, rdx
	movdqu	xmm1, XMMWORD PTR [rcx+r12]
	mul	r9
	paddq	xmm1, xmm3
	mov	r8, rax
	xor	r8, QWORD PTR [rcx+r12+8]
	add	r14, r8
	movq	xmm0, rax
	movq	xmm2, rdx
	xor	rdx, QWORD PTR [rcx+r12]
	mov	rax, r9
	xor	rax, 48
	punpcklqdq xmm2, xmm0
	and	eax, 131056
	add	rsi, rdx
	xor	r9, 16
	xor	edx, edx
	and	r9d, 131056
	movdqu	xmm0, XMMWORD PTR [rax+r12]
	paddq	xmm0, xmm7
	pxor	xmm2, XMMWORD PTR [r9+r12]
	paddq	xmm2, xmm9
	movdqu	XMMWORD PTR [r9+r12], xmm2
	movq	r9, xmm15
	movdqu	XMMWORD PTR [rcx+r12], xmm0
	movdqa	xmm0, xmm4
	mov	rcx, QWORD PTR [rsp+32]
	movdqu	XMMWORD PTR [rax+r12], xmm1
	mov	rax, rcx
	shl	rax, 32
	movaps	xmm1, xmm12
	xor	r9, rax
	psrldq	xmm0, 8
	lea	r8, QWORD PTR [rcx+rcx]
	mov	QWORD PTR [r11], rsi
	add	r8d, edi
	mov	QWORD PTR [r11+8], r14
	movq	rax, xmm0
	mov	ecx, -2147483647
	or	r8, rcx
	xor	rsi, r10
	div	r8
	mov	r10, rdi
	xor	r14, rbx
	mov	eax, eax
	and	r10d, 131056
	shl	rdx, 32
	add	r10, r13
	add	rdx, rax
	xor	r9, QWORD PTR [r10]
	mov	r11, QWORD PTR [r10+8]
	lea	r8, QWORD PTR [rdx+rdi]
	mov	rax, r8
	movq	xmm15, rdx
	shr	rax, 12
	movq	xmm0, rax
	paddq	xmm0, xmm11
	sqrtsd	xmm1, xmm0
	movq	rdx, xmm1
	mov	rax, rdx
	shr	rax, 20
	shr	rdx, 19
	mov	rcx, rdx
	sub	rcx, rax
	sub	rcx, QWORD PTR [rsp+16]
	movdqa	xmm9, xmm7
	movdqa	xmm7, xmm6
	add	rax, QWORD PTR [rsp+216]
	imul	rcx, rax
	mov	rax, r9
	sub	rcx, r8
	mov	rcx, rdi
	adc	rdx, 0
	xor	rcx, 32
	and	ecx, 131056
	mov	QWORD PTR [rsp+32], rdx
	movdqu	xmm1, XMMWORD PTR [rcx+r13]
	mul	rdi
	paddq	xmm1, xmm5
	mov	r8, rax
	xor	r8, QWORD PTR [rcx+r13+8]
	add	r15, r8
	movq	xmm0, rax
	movq	xmm2, rdx
	xor	rdx, QWORD PTR [rcx+r13]
	mov	rax, rdi
	xor	rdi, 16
	punpcklqdq xmm2, xmm0
	xor	rax, 48
	and	edi, 131056
	and	eax, 131056
	add	rbp, rdx
	pxor	xmm2, XMMWORD PTR [rdi+r13]
	movdqu	xmm0, XMMWORD PTR [rax+r13]
	paddq	xmm2, xmm10
	movdqu	XMMWORD PTR [rdi+r13], xmm2
	paddq	xmm0, xmm8
	movdqu	XMMWORD PTR [rcx+r13], xmm0
	movdqa	xmm10, xmm8
	movdqu	XMMWORD PTR [rax+r13], xmm1
	movdqa	xmm8, xmm4
	mov	QWORD PTR [r10], rbp
	xor	rbp, r9
	mov	QWORD PTR [r10+8], r15
	xor	r15, r11
	sub	QWORD PTR [rsp+8], 1
	jne	upx2_main_loop

	ldmxcsr DWORD PTR [rsp+24]

	movaps	xmm13, XMMWORD PTR [rsp+80]
	lea	r11, QWORD PTR [rsp+232]
	movaps	xmm6, XMMWORD PTR [r11-24]
	movaps	xmm7, XMMWORD PTR [r11-40]
	movaps	xmm8, XMMWORD PTR [r11-56]
	movaps	xmm9, XMMWORD PTR [r11-72]
	movaps	xmm10, XMMWORD PTR [r11-88]
	movaps	xmm11, XMMWORD PTR [r11-104]
	movaps	xmm12, XMMWORD PTR [r11-120]
	movaps	xmm14, XMMWORD PTR [rsp+64]
	movaps	xmm15, XMMWORD PTR [rsp+48]
	mov	rsp, r11
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rdi
	pop	rsi
	pop	rbp
	pop	rbx
