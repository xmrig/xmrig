	mov	rcx, [rcx]

	mov	QWORD PTR [rsp+16], rbx
	mov	QWORD PTR [rsp+24], rbp
	mov	QWORD PTR [rsp+32], rsi
	push	rdi
	push	r12
	push	r13
	push	r14
	push	r15
	sub	rsp, 64

	stmxcsr DWORD PTR [rsp]
	mov DWORD PTR [rsp+4], 24448
	ldmxcsr DWORD PTR [rsp+4]

	mov	rax, QWORD PTR [rcx+48]
	mov	r9, rcx
	xor	rax, QWORD PTR [rcx+16]
	mov	ebp, 524288
	mov	r8, QWORD PTR [rcx+32]
	xor	r8, QWORD PTR [rcx]
	mov	r11, QWORD PTR [rcx+40]
	mov	r10, r8
	mov	rdx, QWORD PTR [rcx+56]
	movq	xmm3, rax
	xor	rdx, QWORD PTR [rcx+24]
	xor	r11, QWORD PTR [rcx+8]
	mov	rbx, QWORD PTR [rcx+224]
	mov	rax, QWORD PTR [r9+80]
	xor	rax, QWORD PTR [r9+64]
	movq	xmm0, rdx
	mov	rcx, QWORD PTR [rcx+88]
	xor	rcx, QWORD PTR [r9+72]
	mov	rdi, QWORD PTR [r9+104]
	and	r10d, 2097136
	movaps	XMMWORD PTR [rsp+48], xmm6
	movq	xmm4, rax
	movaps	XMMWORD PTR [rsp+32], xmm7
	movaps	XMMWORD PTR [rsp+16], xmm8
	xorps	xmm8, xmm8
	mov ax, 1023
	shl rax, 52
	movq xmm7, rax
	mov	r15, QWORD PTR [r9+96]
	punpcklqdq xmm3, xmm0
	movq	xmm0, rcx
	punpcklqdq xmm4, xmm0

	ALIGN(64)
cnv2_main_loop_bulldozer:
	movdqa	xmm5, XMMWORD PTR [r10+rbx]
	movq xmm6, r8
	pinsrq xmm6, r11, 1
	lea	rdx, QWORD PTR [r10+rbx]
	lea	r9, QWORD PTR [rdi+rdi]
	shl	rdi, 32

	mov	ecx, r10d
	mov	eax, r10d
	xor	ecx, 16
	xor	eax, 32
	xor	r10d, 48
	aesenc	xmm5, xmm6
	movdqa	xmm2, XMMWORD PTR [rcx+rbx]
	movdqa	xmm1, XMMWORD PTR [rax+rbx]
	movdqa	xmm0, XMMWORD PTR [r10+rbx]
	paddq	xmm2, xmm3
	paddq	xmm1, xmm6
	paddq	xmm0, xmm4
	movdqa	XMMWORD PTR [rcx+rbx], xmm0
	movdqa	XMMWORD PTR [rax+rbx], xmm2
	movdqa	XMMWORD PTR [r10+rbx], xmm1

	movaps	xmm1, xmm8
	mov	rsi, r15
	xor	rsi, rdi

	mov edi, 1023
	shl rdi, 52

	movq	r14, xmm5
	pextrq rax, xmm5, 1

	movdqa	xmm0, xmm5
	pxor	xmm0, xmm3
	mov	r10, r14
	and	r10d, 2097136
	movdqa	XMMWORD PTR [rdx], xmm0
	xor	rsi, QWORD PTR [r10+rbx]
	lea	r12, QWORD PTR [r10+rbx]
	mov	r13, QWORD PTR [r10+rbx+8]

	add	r9d, r14d
	or	r9d, -2147483647
	xor	edx, edx
	div	r9
	mov	eax, eax
	shl	rdx, 32
	lea	r15, [rax+rdx]
	lea	rax, [r14+r15]
	shr	rax, 12
	add	rax, rdi
	movq	xmm0, rax
	sqrtsd	xmm1, xmm0
	movq	rdi, xmm1
	test	rdi, 524287
	je	sqrt_fixup_bulldozer
	shr	rdi, 19

sqrt_fixup_bulldozer_ret:
	mov	rax, rsi
	mul	r14
	movq xmm1, rax
	movq xmm0, rdx
	punpcklqdq xmm0, xmm1

	mov	r9d, r10d
	mov	ecx, r10d
	xor	r9d, 16
	xor	ecx, 32
	xor	r10d, 48
	movdqa	xmm1, XMMWORD PTR [rcx+rbx]
	xor rdx, [rcx+rbx]
	xor rax, [rcx+rbx+8]
	movdqa	xmm2, XMMWORD PTR [r9+rbx]
	pxor xmm2, xmm0
	paddq xmm4, XMMWORD PTR [r10+rbx]
	paddq	xmm2, xmm3
	paddq	xmm1, xmm6
	movdqa	XMMWORD PTR [r9+rbx], xmm4
	movdqa	XMMWORD PTR [rcx+rbx], xmm2
	movdqa	XMMWORD PTR [r10+rbx], xmm1

	movdqa	xmm4, xmm3
	add	r8, rdx
	add	r11, rax
	mov	QWORD PTR [r12], r8
	xor	r8, rsi
	mov	QWORD PTR [r12+8], r11
	mov	r10, r8
	xor	r11, r13
	and	r10d, 2097136
	movdqa	xmm3, xmm5
	dec	ebp
	jne	cnv2_main_loop_bulldozer

	ldmxcsr DWORD PTR [rsp]
	movaps	xmm6, XMMWORD PTR [rsp+48]
	lea	r11, QWORD PTR [rsp+64]
	mov	rbx, QWORD PTR [r11+56]
	mov	rbp, QWORD PTR [r11+64]
	mov	rsi, QWORD PTR [r11+72]
	movaps	xmm8, XMMWORD PTR [r11-48]
	movaps	xmm7, XMMWORD PTR [rsp+32]
	mov	rsp, r11
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rdi
	jmp cnv2_main_loop_bulldozer_endp

sqrt_fixup_bulldozer:
	movq r9, xmm5
	add r9, r15
	dec	rdi
	mov edx, -1022
	shl rdx, 32
	mov	rax, rdi
	shr	rdi, 19
	shr	rax, 20
	mov	rcx, rdi
	sub	rcx, rax
	lea	rcx, [rcx+rdx+1]
	add	rax, rdx
	imul	rcx, rax
	sub	rcx, r9
	adc	rdi, 0
	jmp	sqrt_fixup_bulldozer_ret

cnv2_main_loop_bulldozer_endp:
