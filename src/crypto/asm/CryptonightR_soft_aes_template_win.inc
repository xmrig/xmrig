PUBLIC CryptonightR_soft_aes_template_part1
PUBLIC CryptonightR_soft_aes_template_mainloop
PUBLIC CryptonightR_soft_aes_template_part2
PUBLIC CryptonightR_soft_aes_template_part3
PUBLIC CryptonightR_soft_aes_template_end

ALIGN(64)
CryptonightR_soft_aes_template_part1:
	mov	rcx, [rcx]

	mov	QWORD PTR [rsp+8], rcx
	push	rbx
	push	rbp
	push	rsi
	push	rdi
	push	r12
	push	r13
	push	r14
	push	r15
	sub	rsp, 232

	mov	eax, [rcx+96]
	mov	ebx, [rcx+100]
	mov	esi, [rcx+104]
	mov	edx, [rcx+108]
	mov [rsp+144], eax
	mov [rsp+148], ebx
	mov [rsp+152], esi
	mov [rsp+156], edx

	mov	rax, QWORD PTR [rcx+48]
	mov	r10, rcx
	xor	rax, QWORD PTR [rcx+16]
	mov	r8, QWORD PTR [rcx+32]
	xor	r8, QWORD PTR [rcx]
	mov	r9, QWORD PTR [rcx+40]
	xor	r9, QWORD PTR [rcx+8]
	movq	xmm4, rax
	mov	rdx, QWORD PTR [rcx+56]
	xor	rdx, QWORD PTR [rcx+24]
	mov	r11, QWORD PTR [rcx+224]
	mov	rcx, QWORD PTR [rcx+88]
	xor	rcx, QWORD PTR [r10+72]
	mov	rax, QWORD PTR [r10+80]
	movq	xmm0, rdx
	xor	rax, QWORD PTR [r10+64]

	movaps	XMMWORD PTR [rsp+16], xmm6
	movaps	XMMWORD PTR [rsp+32], xmm7
	movaps	XMMWORD PTR [rsp+48], xmm8
	movaps	XMMWORD PTR [rsp+64], xmm9
	movaps	XMMWORD PTR [rsp+80], xmm10
	movaps	XMMWORD PTR [rsp+96], xmm11
	movaps	XMMWORD PTR [rsp+112], xmm12
	movaps	XMMWORD PTR [rsp+128], xmm13

	movq	xmm5, rax

	mov	rax, r8
	punpcklqdq xmm4, xmm0
	and	eax, 2097136
	movq	xmm10, QWORD PTR [r10+96]
	movq	xmm0, rcx
	mov	rcx, QWORD PTR [r10+104]
	xorps	xmm9, xmm9
	mov	QWORD PTR [rsp+328], rax
	movq	xmm12, r11
	mov	QWORD PTR [rsp+320], r9
	punpcklqdq xmm5, xmm0
	movq xmm13, rcx
	mov r12d, 524288

	ALIGN(64)
CryptonightR_soft_aes_template_mainloop:
	movd xmm11, r12d
	mov	r12, QWORD PTR [r10+272]
	lea	r13, QWORD PTR [rax+r11]
	mov	esi, DWORD PTR [r13]
	movq	xmm0, r9
	mov	r10d, DWORD PTR [r13+4]
	movq	xmm7, r8
	mov	ebp, DWORD PTR [r13+12]
	mov	r14d, DWORD PTR [r13+8]
	mov	rdx, QWORD PTR [rsp+328]
	movzx	ecx, sil
	shr	esi, 8
	punpcklqdq xmm7, xmm0
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
	movd	xmm1, r11d
	add	ebp, 256
	movq	r11, xmm12
	mov	r9d, DWORD PTR [r12+rcx*4]
	xor	r9d, DWORD PTR [r12+rsi*4]
	mov	eax, DWORD PTR [r12+rbp*4]
	xor	r9d, edi
	movzx	ecx, r14b
	movd	xmm0, r10d
	movd	xmm2, r9d
	xor	eax, DWORD PTR [r12+rcx*4]
	mov	rcx, rdx
	xor	eax, r15d
	punpckldq xmm2, xmm1
	xor	rcx, 16
	movd	xmm6, eax
	mov	rax, rdx
	punpckldq xmm6, xmm0
	xor	rax, 32
	punpckldq xmm6, xmm2
	xor	rdx, 48
	movdqu	xmm2, XMMWORD PTR [rcx+r11]
	pxor xmm6, xmm2
	pxor	xmm6, xmm7
	paddq	xmm2, xmm4
	movdqu	xmm1, XMMWORD PTR [rax+r11]
	movdqu	xmm0, XMMWORD PTR [rdx+r11]
	pxor xmm6, xmm1
	pxor xmm6, xmm0
	paddq	xmm0, xmm5
	movdqu	XMMWORD PTR [rcx+r11], xmm0
	movdqu	XMMWORD PTR [rax+r11], xmm2
	movq rcx, xmm13
	paddq	xmm1, xmm7
	movdqu	XMMWORD PTR [rdx+r11], xmm1
	movq	rdi, xmm6
	mov	r10, rdi
	and	r10d, 2097136
	movdqa	xmm0, xmm6
	pxor	xmm0, xmm4
	movdqu	XMMWORD PTR [r13], xmm0

	mov ebx, [rsp+144]
	mov ebp, [rsp+152]
	add ebx, [rsp+148]
	add ebp, [rsp+156]
	shl rbp, 32
	or rbx, rbp

	xor rbx, QWORD PTR [r10+r11]
	lea	r14, QWORD PTR [r10+r11]
	mov	rbp, QWORD PTR [r14+8]

	mov [rsp+160], rbx
	mov [rsp+168], rdi
	mov [rsp+176], rbp
	mov [rsp+184], r10
	mov r10, rsp

	mov ebx, [rsp+144]
	mov esi, [rsp+148]
	mov edi, [rsp+152]
	mov ebp, [rsp+156]

	movd esp, xmm7
	movaps xmm0, xmm7
	psrldq xmm0, 8
	movd r15d, xmm0
	movd eax, xmm4
	movd edx, xmm5
	movaps xmm0, xmm5
	psrldq xmm0, 8
	movd r9d, xmm0

CryptonightR_soft_aes_template_part2:
	mov rsp, r10
	mov [rsp+144], ebx
	mov [rsp+148], esi
	mov [rsp+152], edi
	mov [rsp+156], ebp

	mov edi, edi
	shl rbp, 32
	or rbp, rdi
	xor r8, rbp

	mov ebx, ebx
	shl rsi, 32
	or rsi, rbx
	xor QWORD PTR [rsp+320], rsi

	mov rbx, [rsp+160]
	mov rdi, [rsp+168]
	mov rbp, [rsp+176]
	mov r10, [rsp+184]

	mov	r9, r10
	xor	r9, 16
	mov	rcx, r10
	xor	rcx, 32
	xor	r10, 48
	mov	rax, rbx
	mul	rdi
	movdqu	xmm2, XMMWORD PTR [r9+r11]
	movdqu	xmm1, XMMWORD PTR [rcx+r11]
	pxor xmm6, xmm2
	pxor xmm6, xmm1
	paddq	xmm1, xmm7
	add	r8, rdx
	movdqu	xmm0, XMMWORD PTR [r10+r11]
	pxor xmm6, xmm0
	paddq	xmm0, xmm5
	paddq	xmm2, xmm4
	movdqu	XMMWORD PTR [r9+r11], xmm0
	movdqa	xmm5, xmm4
	mov	r9, QWORD PTR [rsp+320]
	movdqa	xmm4, xmm6
	add	r9, rax
	movdqu	XMMWORD PTR [rcx+r11], xmm2
	movdqu	XMMWORD PTR [r10+r11], xmm1
	mov	r10, QWORD PTR [rsp+304]
	movd r12d, xmm11
	mov	QWORD PTR [r14], r8
	xor	r8, rbx
	mov	rax, r8
	mov	QWORD PTR [r14+8], r9
	and	eax, 2097136
	xor	r9, rbp
	mov	QWORD PTR [rsp+320], r9
	mov	QWORD PTR [rsp+328], rax
	sub	r12d, 1
	jne	CryptonightR_soft_aes_template_mainloop

CryptonightR_soft_aes_template_part3:
	movaps	xmm6, XMMWORD PTR [rsp+16]
	movaps	xmm7, XMMWORD PTR [rsp+32]
	movaps	xmm8, XMMWORD PTR [rsp+48]
	movaps	xmm9, XMMWORD PTR [rsp+64]
	movaps	xmm10, XMMWORD PTR [rsp+80]
	movaps	xmm11, XMMWORD PTR [rsp+96]
	movaps	xmm12, XMMWORD PTR [rsp+112]
	movaps	xmm13, XMMWORD PTR [rsp+128]

	add	rsp, 232
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rdi
	pop	rsi
	pop	rbp
	pop	rbx
	ret
CryptonightR_soft_aes_template_end:
