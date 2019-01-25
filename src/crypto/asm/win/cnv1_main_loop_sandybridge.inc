	mov	QWORD PTR [rsp+8], rbx
	mov	QWORD PTR [rsp+16], rbp
	mov	QWORD PTR [rsp+24], rsi
	mov	QWORD PTR [rsp+32], rdi
	push	r14
	push	r15
	mov	rax, QWORD PTR [rcx+48]
	mov	ebp, 524288
	xor	rax, QWORD PTR [rcx+16]
	mov	rdx, QWORD PTR [rcx+56]
	xor	rdx, QWORD PTR [rcx+24]
	mov	r8, QWORD PTR [rcx+32]
	xor	r8, QWORD PTR [rcx]
	movq	xmm3, rax
	mov	rax, QWORD PTR [rcx+256]
	mov	rdi, QWORD PTR [rcx+40]
	movq	xmm0, rdx
	xor	rdi, QWORD PTR [rcx+8]
	mov	rdx, r8
	mov	r15, QWORD PTR [rcx+264]
	and	edx, 2097136
	mov	r14, QWORD PTR [rax+35]
	xor	r14, QWORD PTR [rcx+192]
	mov	rsi, QWORD PTR [rcx+224]
	punpcklqdq xmm3, xmm0
	movdqu	xmm2, XMMWORD PTR [rdx+rsi]

	ALIGN 64
cnv1_main_loop_original_sandybridge:
	movq	xmm0, rdi
	movq	xmm1, r8
	punpcklqdq xmm1, xmm0
	aesenc	xmm2, xmm1
	movq r10, xmm2
	mov r9d, r10d
	and r9d, 2097136
	add r9, rsi
	movdqa	xmm0, xmm2
	pxor	xmm0, xmm3
	movdqa	xmm3, xmm2
	movdqu	XMMWORD PTR [rdx+rsi], xmm0
	psrldq	xmm0, 11
	movq	rax, xmm0
	movzx	eax, al
	movzx	eax, BYTE PTR [rax+r15]
	mov	BYTE PTR [rsi+rdx+11], al
	mov	rbx, QWORD PTR [r9]
	mov	r11, QWORD PTR [r9+8]
	mov	rax, rbx
	mul	r10
	add	r8, rdx
	mov	QWORD PTR [r9], r8
	add	rdi, rax
	mov	rax, r14
	xor	rax, rdi
	mov	QWORD PTR [r9+8], rax
	xor	r8, rbx
	mov	rdx, r8
	and	edx, 2097136
	movdqu	xmm2, XMMWORD PTR [rdx+rsi]
	xor	rdi, r11
	dec	ebp
	jne	cnv1_main_loop_original_sandybridge

	mov	rbx, QWORD PTR [rsp+24]
	mov	rbp, QWORD PTR [rsp+32]
	mov	rsi, QWORD PTR [rsp+40]
	mov	rdi, QWORD PTR [rsp+48]
	pop	r15
	pop	r14
