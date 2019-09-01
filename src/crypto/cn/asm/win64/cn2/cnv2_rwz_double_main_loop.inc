	mov	rdx, [rcx+8]
	mov	rcx, [rcx]

	mov	rax, rsp
	push	rbx
	push	rbp
	push	rsi
	push	rdi
	push	r12
	push	r13
	push	r14
	push	r15
	sub	rsp, 184

	stmxcsr DWORD PTR [rsp+272]
	mov DWORD PTR [rsp+276], 24448
	ldmxcsr DWORD PTR [rsp+276]

	mov	r13, QWORD PTR [rcx+224]
	mov	r9, rdx
	mov	r10, QWORD PTR [rcx+32]
	mov	r8, rcx
	xor	r10, QWORD PTR [rcx]
	mov	r14d, 393216
	mov	r11, QWORD PTR [rcx+40]
	xor	r11, QWORD PTR [rcx+8]
	mov	rsi, QWORD PTR [rdx+224]
	mov	rdx, QWORD PTR [rcx+56]
	xor	rdx, QWORD PTR [rcx+24]
	mov	rdi, QWORD PTR [r9+32]
	xor	rdi, QWORD PTR [r9]
	mov	rbp, QWORD PTR [r9+40]
	xor	rbp, QWORD PTR [r9+8]
	movd	xmm0, rdx
	movaps	XMMWORD PTR [rax-88], xmm6
	movaps	XMMWORD PTR [rax-104], xmm7
	movaps	XMMWORD PTR [rax-120], xmm8
	movaps	XMMWORD PTR [rsp+112], xmm9
	movaps	XMMWORD PTR [rsp+96], xmm10
	movaps	XMMWORD PTR [rsp+80], xmm11
	movaps	XMMWORD PTR [rsp+64], xmm12
	movaps	XMMWORD PTR [rsp+48], xmm13
	movaps	XMMWORD PTR [rsp+32], xmm14
	movaps	XMMWORD PTR [rsp+16], xmm15
	mov	rdx, r10
	movd	xmm4, QWORD PTR [r8+96]
	and	edx, 2097136
	mov	rax, QWORD PTR [rcx+48]
	xorps	xmm13, xmm13
	xor	rax, QWORD PTR [rcx+16]
	mov	rcx, QWORD PTR [rcx+88]
	xor	rcx, QWORD PTR [r8+72]
	movd	xmm5, QWORD PTR [r8+104]
	movd	xmm7, rax

	mov eax, 1
	shl rax, 52
	movd xmm14, rax
	punpcklqdq xmm14, xmm14

	mov eax, 1023
	shl rax, 52
	movd xmm12, rax
	punpcklqdq xmm12, xmm12

	mov	rax, QWORD PTR [r8+80]
	xor	rax, QWORD PTR [r8+64]
	punpcklqdq xmm7, xmm0
	movd	xmm0, rcx
	mov	rcx, QWORD PTR [r9+56]
	xor	rcx, QWORD PTR [r9+24]
	movd	xmm3, rax
	mov	rax, QWORD PTR [r9+48]
	xor	rax, QWORD PTR [r9+16]
	punpcklqdq xmm3, xmm0
	movd	xmm0, rcx
	mov	QWORD PTR [rsp], r13
	mov	rcx, QWORD PTR [r9+88]
	xor	rcx, QWORD PTR [r9+72]
	movd	xmm6, rax
	mov	rax, QWORD PTR [r9+80]
	xor	rax, QWORD PTR [r9+64]
	punpcklqdq xmm6, xmm0
	movd	xmm0, rcx
	mov	QWORD PTR [rsp+256], r10
	mov	rcx, rdi
	mov	QWORD PTR [rsp+264], r11
	movd	xmm8, rax
	and	ecx, 2097136
	punpcklqdq xmm8, xmm0
	movd	xmm0, QWORD PTR [r9+96]
	punpcklqdq xmm4, xmm0
	movd	xmm0, QWORD PTR [r9+104]
	lea	r8, QWORD PTR [rcx+rsi]
	movdqu	xmm11, XMMWORD PTR [r8]
	punpcklqdq xmm5, xmm0
	lea	r9, QWORD PTR [rdx+r13]
	movdqu	xmm15, XMMWORD PTR [r9]

	ALIGN(64)
rwz_main_loop_double:
	movdqu	xmm9, xmm15
	mov eax, edx
	mov ebx, edx
	xor eax, 16
	xor ebx, 32
	xor edx, 48

	movd	xmm0, r11
	movd	xmm2, r10
	punpcklqdq xmm2, xmm0
	aesenc	xmm9, xmm2

	movdqu	xmm0, XMMWORD PTR [rdx+r13]
	movdqu	xmm1, XMMWORD PTR [rbx+r13]
	paddq	xmm0, xmm7
	paddq	xmm1, xmm2
	movdqu	XMMWORD PTR [rbx+r13], xmm0
	movdqu	xmm0, XMMWORD PTR [rax+r13]
	movdqu	XMMWORD PTR [rdx+r13], xmm1
	paddq	xmm0, xmm3
	movdqu	XMMWORD PTR [rax+r13], xmm0

	movd	r11, xmm9
	mov	edx, r11d
	and	edx, 2097136
	movdqa	xmm0, xmm9
	pxor	xmm0, xmm7
	movdqu	XMMWORD PTR [r9], xmm0

	lea	rbx, QWORD PTR [rdx+r13]
	mov	r10, QWORD PTR [rdx+r13]

	movdqu	xmm10, xmm11
	movd	xmm0, rbp
	movd	xmm11, rdi
	punpcklqdq xmm11, xmm0
	aesenc	xmm10, xmm11

	mov eax, ecx
	mov r12d, ecx
	xor eax, 16
	xor r12d, 32
	xor ecx, 48

	movdqu	xmm0, XMMWORD PTR [rcx+rsi]
	paddq	xmm0, xmm6
	movdqu	xmm1, XMMWORD PTR [r12+rsi]
	movdqu	XMMWORD PTR [r12+rsi], xmm0
	paddq	xmm1, xmm11
	movdqu	xmm0, XMMWORD PTR [rax+rsi]
	movdqu	XMMWORD PTR [rcx+rsi], xmm1
	paddq	xmm0, xmm8
	movdqu	XMMWORD PTR [rax+rsi], xmm0

	movd	rcx, xmm10
	and	ecx, 2097136

	movdqa	xmm0, xmm10
	pxor	xmm0, xmm6
	movdqu	XMMWORD PTR [r8], xmm0
	mov r12, QWORD PTR [rcx+rsi]

	mov	r9, QWORD PTR [rbx+8]

	xor edx, 16
	mov r8d, edx
	mov r15d, edx

	movd	rdx, xmm5
	shl	rdx, 32
	movd	rax, xmm4
	xor	rdx, rax
	xor	r10, rdx
	mov	rax, r10
	mul	r11
	mov r11d, r8d
	xor r11d, 48
	movd xmm0, rdx
	xor rdx, [r11+r13]
	movd xmm1, rax
	xor rax, [r11+r13+8]
	punpcklqdq xmm0, xmm1

	pxor xmm0, XMMWORD PTR [r8+r13]
	movdqu	xmm1, XMMWORD PTR [r11+r13]
	paddq	xmm0, xmm3
	paddq	xmm1, xmm2
	movdqu	XMMWORD PTR [r8+r13], xmm0
	xor	r8d, 32
	movdqu	xmm0, XMMWORD PTR [r8+r13]
	movdqu	XMMWORD PTR [r8+r13], xmm1
	paddq	xmm0, xmm7
	movdqu	XMMWORD PTR [r11+r13], xmm0

	mov	r11, QWORD PTR [rsp+256]
	add	r11, rdx
	mov	rdx, QWORD PTR [rsp+264]
	add	rdx, rax
	mov	QWORD PTR [rbx], r11
	xor	r11, r10
	mov	QWORD PTR [rbx+8], rdx
	xor	rdx, r9
	mov	QWORD PTR [rsp+256], r11
	and	r11d, 2097136
	mov	QWORD PTR [rsp+264], rdx
	mov	QWORD PTR [rsp+8], r11
	lea	r15, QWORD PTR [r11+r13]
	movdqu xmm15, XMMWORD PTR [r11+r13]
	lea	r13, QWORD PTR [rsi+rcx]
	movdqa	xmm0, xmm5
	psrldq	xmm0, 8
	movaps	xmm2, xmm13
	movd	r10, xmm0
	psllq	xmm5, 1
	shl	r10, 32
	movdqa	xmm0, xmm9
	psrldq	xmm0, 8
	movdqa	xmm1, xmm10
	movd	r11, xmm0
	psrldq	xmm1, 8
	movd	r8, xmm1
	psrldq	xmm4, 8
	movaps	xmm0, xmm13
	movd	rax, xmm4
	xor	r10, rax
	movaps	xmm1, xmm13
	xor	r10, r12
	lea	rax, QWORD PTR [r11+1]
	shr	rax, 1
	movdqa	xmm3, xmm9
	punpcklqdq xmm3, xmm10
	paddq	xmm5, xmm3
	movd	rdx, xmm5
	psrldq	xmm5, 8
	cvtsi2sd xmm2, rax
	or	edx, -2147483647
	lea	rax, QWORD PTR [r8+1]
	shr	rax, 1
	movd	r9, xmm5
	cvtsi2sd xmm0, rax
	or	r9d, -2147483647
	cvtsi2sd xmm1, rdx
	unpcklpd xmm2, xmm0
	movaps	xmm0, xmm13
	cvtsi2sd xmm0, r9
	unpcklpd xmm1, xmm0
	divpd	xmm2, xmm1
	paddq	xmm2, xmm14
	cvttsd2si rax, xmm2
	psrldq	xmm2, 8
	mov	rbx, rax
	imul	rax, rdx
	sub	r11, rax
	js	rwz_div_fix_1
rwz_div_fix_1_ret:

	cvttsd2si rdx, xmm2
	mov	rax, rdx
	imul	rax, r9
	movd	xmm2, r11d
	movd	xmm4, ebx
	sub	r8, rax
	js	rwz_div_fix_2
rwz_div_fix_2_ret:

	movd	xmm1, r8d
	movd	xmm0, edx
	punpckldq xmm2, xmm1
	punpckldq xmm4, xmm0
	punpckldq xmm4, xmm2
	paddq	xmm3, xmm4
	movdqa	xmm0, xmm3
	psrlq	xmm0, 12
	paddq	xmm0, xmm12
	sqrtpd	xmm1, xmm0
	movd	r9, xmm1
	movdqa xmm5, xmm1
	psrlq xmm5, 19
	test	r9, 524287
	je	rwz_sqrt_fix_1
rwz_sqrt_fix_1_ret:

	movd r9, xmm10
	psrldq	xmm1, 8
	movd	r8, xmm1
	test	r8, 524287
	je	rwz_sqrt_fix_2
rwz_sqrt_fix_2_ret:

	mov r12d, ecx
	mov r8d, ecx
	xor r12d, 16
	xor r8d, 32
	xor ecx, 48
	mov	rax, r10
	mul	r9
	movd xmm0, rax
	movd xmm3, rdx
	punpcklqdq xmm3, xmm0

	movdqu	xmm0, XMMWORD PTR [r12+rsi]
	pxor xmm0, xmm3
	movdqu	xmm1, XMMWORD PTR [r8+rsi]
	xor rdx, [r8+rsi]
	xor rax, [r8+rsi+8]
	movdqu	xmm3, XMMWORD PTR [rcx+rsi]
	paddq	xmm3, xmm6
	paddq	xmm1, xmm11
	paddq	xmm0, xmm8
	movdqu	XMMWORD PTR [r8+rsi], xmm3
	movdqu	XMMWORD PTR [rcx+rsi], xmm1
	movdqu	XMMWORD PTR [r12+rsi], xmm0

	add	rdi, rdx
	mov	QWORD PTR [r13], rdi
	xor	rdi, r10
	mov	ecx, edi
	and	ecx, 2097136
	lea	r8, QWORD PTR [rcx+rsi]

	mov rdx, QWORD PTR [r13+8]	
	add	rbp, rax
	mov	QWORD PTR [r13+8], rbp
	movdqu xmm11, XMMWORD PTR [rcx+rsi]
	xor	rbp, rdx
	mov	r13, QWORD PTR [rsp]
	movdqa	xmm3, xmm7
	mov	rdx, QWORD PTR [rsp+8]
	movdqa	xmm8, xmm6
	mov	r10, QWORD PTR [rsp+256]
	movdqa	xmm7, xmm9
	mov	r11, QWORD PTR [rsp+264]
	movdqa	xmm6, xmm10
	mov	r9, r15
	dec r14d
	jne	rwz_main_loop_double

	ldmxcsr DWORD PTR [rsp+272]
	movaps	xmm13, XMMWORD PTR [rsp+48]
	lea	r11, QWORD PTR [rsp+184]
	movaps	xmm6, XMMWORD PTR [r11-24]
	movaps	xmm7, XMMWORD PTR [r11-40]
	movaps	xmm8, XMMWORD PTR [r11-56]
	movaps	xmm9, XMMWORD PTR [r11-72]
	movaps	xmm10, XMMWORD PTR [r11-88]
	movaps	xmm11, XMMWORD PTR [r11-104]
	movaps	xmm12, XMMWORD PTR [r11-120]
	movaps	xmm14, XMMWORD PTR [rsp+32]
	movaps	xmm15, XMMWORD PTR [rsp+16]
	mov	rsp, r11
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rdi
	pop	rsi
	pop	rbp
	pop	rbx
	jmp rwz_cnv2_double_mainloop_asm_endp

rwz_div_fix_1:
	dec	rbx
	add	r11, rdx
	jmp	rwz_div_fix_1_ret

rwz_div_fix_2:
	dec	rdx
	add	r8, r9
	jmp	rwz_div_fix_2_ret

rwz_sqrt_fix_1:
	movd	r8, xmm3
	movdqa xmm0, xmm5
	psrldq xmm0, 8
	dec	r9
	mov r11d, -1022
	shl r11, 32
	mov	rax, r9
	shr	r9, 19
	shr	rax, 20
	mov	rdx, r9
	sub	rdx, rax
	lea	rdx, [rdx+r11+1]
	add	rax, r11
	imul	rdx, rax
	sub	rdx, r8
	adc	r9, 0
	movd xmm5, r9
	punpcklqdq xmm5, xmm0
	jmp	rwz_sqrt_fix_1_ret

rwz_sqrt_fix_2:
	psrldq	xmm3, 8
	movd	r11, xmm3
	dec	r8
	mov ebx, -1022
	shl rbx, 32
	mov	rax, r8
	shr	r8, 19
	shr	rax, 20
	mov	rdx, r8
	sub	rdx, rax
	lea	rdx, [rdx+rbx+1]
	add	rax, rbx
	imul	rdx, rax
	sub	rdx, r11
	adc	r8, 0
	movd xmm0, r8
	punpcklqdq xmm5, xmm0
	jmp	rwz_sqrt_fix_2_ret

rwz_cnv2_double_mainloop_asm_endp:
