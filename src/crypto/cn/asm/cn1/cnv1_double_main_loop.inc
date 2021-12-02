	mov	QWORD PTR [rsp+8], rbx
	mov	QWORD PTR [rsp+16], rbp
	mov	QWORD PTR [rsp+24], rsi
	push	rdi
	push	r12
	push	r13
	push	r14
	push	r15
	sub	rsp, 32
	mov	rdx, QWORD PTR [rcx]
	mov	r8, QWORD PTR [rcx+8]
	mov	r12d, 524288
	movaps	XMMWORD PTR [rsp+16], xmm6
	mov	rbx, QWORD PTR [rdx+32]
	xor	rbx, QWORD PTR [rdx]
	mov	rsi, QWORD PTR [rdx+40]
	mov	r10, rbx
	xor	rsi, QWORD PTR [rdx+8]
	and	r10d, 2097136
	mov	rdi, QWORD PTR [r8+32]
	xor	rdi, QWORD PTR [r8]
	movq	xmm3, rbx
	mov	rbp, QWORD PTR [r8+40]
	mov	r9, rdi
	xor	rbp, QWORD PTR [r8+8]
	movq	xmm0, rsi
	mov	rcx, QWORD PTR [rdx+56]
	and	r9d, 2097136
	xor	rcx, QWORD PTR [rdx+24]
	movq	xmm4, rdi
	mov	rax, QWORD PTR [rdx+48]
	xor	rax, QWORD PTR [rdx+16]
	mov	r14, QWORD PTR [rdx+224]
	mov	r13, QWORD PTR [rdx+232]
	mov	r15, QWORD PTR [r8+224]
	punpcklqdq xmm3, xmm0
	movq	xmm0, rbp
	movq	xmm5, rax
	punpcklqdq xmm4, xmm0
	mov	rax, QWORD PTR [r8+48]
	movq	xmm0, rcx
	xor	rax, QWORD PTR [r8+16]
	mov	rcx, QWORD PTR [r8+56]
	xor	rcx, QWORD PTR [r8+24]
	movdqu	xmm1, XMMWORD PTR [r14+r10]
	movq	xmm6, rax
	punpcklqdq xmm5, xmm0
	mov	rax, QWORD PTR [rdx+240]
	movq	xmm0, rcx
	movdqu	xmm2, XMMWORD PTR [r15+r9]
	mov	QWORD PTR [rsp], rax
	mov	rax, QWORD PTR [r8+240]
	mov	QWORD PTR [rsp+8], rax
	punpcklqdq xmm6, xmm0

	ALIGN(64)
main_loop_cnv1_double:
	aesenc	xmm1, xmm3
	aesenc	xmm2, xmm4
	movdqa	xmm0, xmm1
	movq	r11, xmm2
	pxor	xmm0, xmm5
	movdqa	xmm5, xmm1
	movq	QWORD PTR [r14+r10], xmm0
	pextrq	rcx, xmm0, 1
	mov	eax, ecx
	movdqa	xmm0, xmm2
	shr	rax, 24
	pxor	xmm0, xmm6
	movdqa	xmm6, xmm2
	mov	eax, DWORD PTR [r13+rax*4]
	xor	rax, rcx
	mov	QWORD PTR [r14+r10+8], rax
	movq	QWORD PTR [r15+r9], xmm0
	pextrq	rcx, xmm0, 1
	mov	eax, ecx
	shr	rax, 24
	mov	eax, DWORD PTR [r13+rax*4]
	xor	rax, rcx
	movq	rcx, xmm1
	mov	QWORD PTR [r15+r9+8], rax
	mov	r9, rcx
	and	r9d, 2097136
	mov	r10, QWORD PTR [r14+r9]
	mov	r8, QWORD PTR [r14+r9+8]
	mov	rax, r10
	mul	rcx
	add	rsi, rax
	add	rbx, rdx
	mov	rax, QWORD PTR [rsp]
	mov	QWORD PTR [r14+r9], rbx
	xor	rax, rsi
	mov	QWORD PTR [r14+r9+8], rax
	xor	rsi, r8
	xor	rbx, r10
	mov	r8, r11
	and	r8d, 2097136
	mov	r10, rbx
	and	r10d, 2097136
	movq	xmm3, rbx
	pinsrq	xmm3, rsi, 1
	mov	r9, QWORD PTR [r15+r8]
	mov	rcx, QWORD PTR [r15+r8+8]
	mov	rax, r9
	movdqu	xmm1, XMMWORD PTR [r14+r10]
	mul	r11
	add	rbp, rax
	add	rdi, rdx
	mov	rax, QWORD PTR [rsp+8]
	mov	QWORD PTR [r15+r8], rdi
	xor	rax, rbp
	xor	rdi, r9
	mov	QWORD PTR [r15+r8+8], rax
	mov	r9, rdi
	xor	rbp, rcx
	and	r9d, 2097136
	movq	xmm4, rdi
	pinsrq	xmm4, rbp, 1
	movdqu	xmm2, XMMWORD PTR [r15+r9]
	sub	r12, 1
	jne	main_loop_cnv1_double

	mov	rbx, QWORD PTR [rsp+80]
	mov	rbp, QWORD PTR [rsp+88]
	mov	rsi, QWORD PTR [rsp+96]
	movaps	xmm6, XMMWORD PTR [rsp+16]
	add	rsp, 32
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rdi
