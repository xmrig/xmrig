	mov	rax, rsp
	mov	QWORD PTR [rax+8], rbx
	mov	QWORD PTR [rax+16], rbp
	mov	QWORD PTR [rax+24], rsi
	push	rdi
	push	r12
	push	r13
	push	r14
	push	r15
	sub	rsp, 144
	mov	r8, QWORD PTR [rcx]
	mov	r9, QWORD PTR [rcx+8]
	mov	r10, QWORD PTR [rcx+16]
	mov	r11, QWORD PTR [rcx+24]
	mov	rbp, QWORD PTR [r8+224]
	mov	r13, QWORD PTR [r8+232]
	mov	r14, QWORD PTR [r9+224]
	mov	r15, QWORD PTR [r10+224]
	mov	r12, QWORD PTR [r11+224]
	mov	rcx, QWORD PTR [r8+40]
	xor	rcx, QWORD PTR [r8+8]
	mov	rbx, QWORD PTR [r8+32]
	xor	rbx, QWORD PTR [r8]
	mov	rdi, QWORD PTR [r9+32]
	xor	rdi, QWORD PTR [r9]
	movd	xmm0, rcx
	mov	rcx, QWORD PTR [r9+40]
	xor	rcx, QWORD PTR [r9+8]
	movd	xmm1, rbx
	movaps	XMMWORD PTR [rax-56], xmm6
	movaps	XMMWORD PTR [rax-72], xmm7
	movaps	XMMWORD PTR [rax-88], xmm8
	movaps	XMMWORD PTR [rax-104], xmm9
	movaps	XMMWORD PTR [rax-120], xmm10
	movaps	XMMWORD PTR [rsp+48], xmm11
	movaps	XMMWORD PTR [rsp+32], xmm12
	and	ebx, 2097136
	mov	rsi, QWORD PTR [r10+32]
	movd	xmm2, rdi
	mov	rax, QWORD PTR [r8+240]
	and	edi, 2097136
	xor	rsi, QWORD PTR [r10]
	mov	rdx, QWORD PTR [r8+56]
	xor	rdx, QWORD PTR [r8+24]
	mov	QWORD PTR [rsp], rax
	mov	rax, QWORD PTR [r9+240]
	movd	xmm3, rsi
	mov	QWORD PTR [rsp+8], rax
	and	esi, 2097136
	mov	rax, QWORD PTR [r10+240]
	punpcklqdq xmm1, xmm0
	movd	xmm0, rcx
	mov	rcx, QWORD PTR [r10+40]
	xor	rcx, QWORD PTR [r10+8]
	mov	QWORD PTR [rsp+16], rax
	mov	rax, QWORD PTR [r11+240]
	punpcklqdq xmm2, xmm0
	movd	xmm0, rcx
	mov	QWORD PTR [rsp+24], rax
	mov	rcx, QWORD PTR [r11+40]
	xor	rcx, QWORD PTR [r11+8]
	mov	rax, QWORD PTR [r11+32]
	xor	rax, QWORD PTR [r11]
	punpcklqdq xmm3, xmm0
	movd	xmm0, rcx
	mov	rcx, QWORD PTR [r8+48]
	xor	rcx, QWORD PTR [r8+16]
	movd	xmm4, rax
	and	eax, 2097136
	punpcklqdq xmm4, xmm0
	movd	xmm0, rdx
	mov	rdx, QWORD PTR [r9+56]
	xor	rdx, QWORD PTR [r9+24]
	movd	xmm5, rcx
	mov	rcx, QWORD PTR [r9+48]
	xor	rcx, QWORD PTR [r9+16]
	punpcklqdq xmm5, xmm0
	movd	xmm0, rdx
	mov	rdx, QWORD PTR [r10+56]
	xor	rdx, QWORD PTR [r10+24]
	movd	xmm6, rcx
	mov	rcx, QWORD PTR [r10+48]
	xor	rcx, QWORD PTR [r10+16]
	punpcklqdq xmm6, xmm0
	movd	xmm0, rdx
	mov	rdx, QWORD PTR [r11+56]
	movd	xmm7, rcx
	punpcklqdq xmm7, xmm0
	xor	rdx, QWORD PTR [r11+24]
	mov	rcx, QWORD PTR [r11+48]
	xor	rcx, QWORD PTR [r11+16]
	mov	r11d, 524288
	movdqu	xmm9, XMMWORD PTR [rbp+rbx]
	movdqu	xmm10, XMMWORD PTR [r14+rdi]
	movd	xmm0, rdx
	movdqu	xmm11, XMMWORD PTR [r15+rsi]
	movdqu	xmm12, XMMWORD PTR [r12+rax]
	movd	xmm8, rcx
	punpcklqdq xmm8, xmm0

	ALIGN(64)
main_loop_cnv1_quad:
	aesenc	xmm9, xmm1
	aesenc	xmm10, xmm2
	aesenc	xmm11, xmm3
	aesenc	xmm12, xmm4
	movd	ecx, xmm9
	and	ecx, 2097136
	prefetcht0 BYTE PTR [rcx+rbp]
	movd	ecx, xmm10
	and	ecx, 2097136
	prefetcht0 BYTE PTR [rcx+r14]
	movd	ecx, xmm11
	and	ecx, 2097136
	prefetcht0 BYTE PTR [rcx+r15]
	movd	ecx, xmm12
	and	ecx, 2097136
	prefetcht0 BYTE PTR [rcx+r12]
	movdqa	xmm0, xmm9
	pxor	xmm0, xmm5
	movdqa	xmm5, xmm9
	movd	QWORD PTR [rbp+rbx], xmm0
	pextrq	rdx, xmm0, 1
	mov	ecx, edx
	movdqa	xmm0, xmm10
	shr	rcx, 24
	pxor	xmm0, xmm6
	mov	ecx, DWORD PTR [r13+rcx*4]
	xor	rcx, rdx
	mov	QWORD PTR [rbp+rbx+8], rcx
	movd	rbx, xmm1
	movd	QWORD PTR [r14+rdi], xmm0
	pextrq	rdx, xmm0, 1
	mov	ecx, edx
	movdqa	xmm0, xmm11
	shr	rcx, 24
	pxor	xmm0, xmm7
	mov	ecx, DWORD PTR [r13+rcx*4]
	xor	rcx, rdx
	mov	QWORD PTR [r14+rdi+8], rcx
	movd	rdi, xmm2
	movd	QWORD PTR [r15+rsi], xmm0
	pextrq	rdx, xmm0, 1
	mov	ecx, edx
	movdqa	xmm0, xmm12
	shr	rcx, 24
	pxor	xmm0, xmm8
	mov	ecx, DWORD PTR [r13+rcx*4]
	xor	rcx, rdx
	mov	QWORD PTR [r15+rsi+8], rcx
	movd	QWORD PTR [r12+rax], xmm0
	pextrq	rdx, xmm0, 1
	mov	ecx, edx
	shr	rcx, 24
	mov	ecx, DWORD PTR [r13+rcx*4]
	xor	rcx, rdx
	mov	QWORD PTR [r12+rax+8], rcx
	movd	rcx, xmm9
	mov	r8, rcx
	and	r8d, 2097136
	mov	r9, QWORD PTR [rbp+r8]
	mov	r10, QWORD PTR [rbp+r8+8]
	mov	rax, r9
	mul	rcx
	pextrq	rcx, xmm1, 1
	add	rcx, rax
	add	rbx, rdx
	mov	rax, QWORD PTR [rsp]
	mov	QWORD PTR [rbp+r8], rbx
	xor	rax, rcx
	mov	QWORD PTR [rbp+r8+8], rax
	xor	rcx, r10
	xor	rbx, r9
	movd	xmm1, rbx
	and	ebx, 2097136
	pinsrq	xmm1, rcx, 1
	movd	rcx, xmm10
	mov	r8, rcx
	and	r8d, 2097136
	movdqu	xmm9, XMMWORD PTR [rbp+rbx]
	mov	r9, QWORD PTR [r14+r8]
	mov	r10, QWORD PTR [r14+r8+8]
	mov	rax, r9
	mul	rcx
	pextrq	rcx, xmm2, 1
	add	rcx, rax
	add	rdi, rdx
	mov	rax, QWORD PTR [rsp+8]
	mov	QWORD PTR [r14+r8], rdi
	xor	rax, rcx
	xor	rdi, r9
	mov	QWORD PTR [r14+r8+8], rax
	xor	rcx, r10
	movd	xmm2, rdi
	and	edi, 2097136
	pinsrq	xmm2, rcx, 1
	movd	rcx, xmm11
	movd	rsi, xmm3
	mov	r8, rcx
	and	r8d, 2097136
	movdqa	xmm6, xmm10
	movdqa	xmm7, xmm11
	movdqa	xmm8, xmm12
	movdqu	xmm10, XMMWORD PTR [r14+rdi]
	mov	r9, QWORD PTR [r15+r8]
	mov	r10, QWORD PTR [r15+r8+8]
	mov	rax, r9
	mul	rcx
	pextrq	rcx, xmm3, 1
	add	rcx, rax
	add	rsi, rdx
	mov	rax, QWORD PTR [rsp+16]
	xor	rax, rcx
	mov	QWORD PTR [r15+r8], rsi
	mov	QWORD PTR [r15+r8+8], rax
	xor	rcx, r10
	xor	rsi, r9
	movd	xmm3, rsi
	and	esi, 2097136
	pinsrq	xmm3, rcx, 1
	movd	rcx, xmm12
	mov	r8, rcx
	and	r8d, 2097136
	movdqu	xmm11, XMMWORD PTR [r15+rsi]
	mov	r9, QWORD PTR [r12+r8]
	mov	r10, QWORD PTR [r12+r8+8]
	mov	rax, r9
	mul	rcx
	mov	rcx, rax
	movd	rax, xmm4
	add	rax, rdx
	mov	QWORD PTR [r12+r8], rax
	xor	rax, r9
	pextrq	rdx, xmm4, 1
	add	rdx, rcx
	mov	rcx, QWORD PTR [rsp+24]
	xor	rcx, rdx
	xor	rdx, r10
	movd	xmm4, rax
	mov	QWORD PTR [r12+r8+8], rcx
	and	eax, 2097136
	pinsrq	xmm4, rdx, 1
	movdqu	xmm12, XMMWORD PTR [r12+rax]
	sub	r11, 1
	jne	main_loop_cnv1_quad

	movaps	xmm7, XMMWORD PTR [rsp+112]
	lea	r11, QWORD PTR [rsp+144]
	mov	rbx, QWORD PTR [r11+48]
	mov	rbp, QWORD PTR [r11+56]
	mov	rsi, QWORD PTR [r11+64]
	movaps	xmm6, XMMWORD PTR [r11-16]
	movaps	xmm8, XMMWORD PTR [r11-48]
	movaps	xmm9, XMMWORD PTR [r11-64]
	movaps	xmm10, XMMWORD PTR [r11-80]
	movaps	xmm11, XMMWORD PTR [r11-96]
	movaps	xmm12, XMMWORD PTR [r11-112]
	mov	rsp, r11
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rdi
