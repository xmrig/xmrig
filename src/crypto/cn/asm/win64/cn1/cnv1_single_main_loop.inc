	mov	QWORD PTR [rsp+8], rbx
	mov	QWORD PTR [rsp+16], rbp
	mov	QWORD PTR [rsp+24], rsi
	mov	QWORD PTR [rsp+32], rdi
	push	r13
	push	r14
	push	r15
	mov	rdx, QWORD PTR [rcx]
	mov	esi, 524288
	mov	r11, QWORD PTR [rdx+32]
	xor	r11, QWORD PTR [rdx]
	mov	rdi, QWORD PTR [rdx+224]
	mov	rbx, QWORD PTR [rdx+40]
	xor	rbx, QWORD PTR [rdx+8]
	mov	rcx, QWORD PTR [rdx+56]
	xor	rcx, QWORD PTR [rdx+24]
	mov	rax, QWORD PTR [rdx+48]
	xor	rax, QWORD PTR [rdx+16]
	mov	rbp, QWORD PTR [rdx+240]
	mov	r14, QWORD PTR [rdx+232]
	movd	xmm2, rax
	pinsrq	xmm2, rcx, 1

	ALIGN(64)
main_loop_cnv1_single:
	mov	r8, r11
	and	r8d, 2097136
	movdqu	xmm1, XMMWORD PTR [rdi+r8]
	movd	xmm0, r11
	pinsrq	xmm0, rbx, 1
	aesenc	xmm1, xmm0
	movd	r15, xmm1
	mov	r9, r15
	and	r9d, 2097136
	movdqa	xmm0, xmm1
	pxor	xmm0, xmm2
	movdqa	xmm2, xmm1
	movd	QWORD PTR [rdi+r8], xmm0
	pextrq	rdx, xmm0, 1
	mov	eax, edx
	shr	rax, 24
	mov	ecx, DWORD PTR [r14+rax*4]
	xor	rcx, rdx
	mov	QWORD PTR [rdi+r8+8], rcx
	mov	r10, QWORD PTR [rdi+r9]
	mov	r8, QWORD PTR [rdi+r9+8]
	mov	rax, r10
	mul	r15
	add	rbx, rax
	add	r11, rdx
	mov	QWORD PTR [rdi+r9], r11
	mov	rax, rbx
	xor	rbx, r8
	xor	r11, r10
	xor	rax, rbp
	mov	QWORD PTR [rdi+r9+8], rax
	sub	rsi, 1
	jne	main_loop_cnv1_single

	pop	r15
	pop	r14
	pop	r13
	mov	rbx, QWORD PTR [rsp+8]
	mov	rbp, QWORD PTR [rsp+16]
	mov	rsi, QWORD PTR [rsp+24]
	mov	rdi, QWORD PTR [rsp+32]
