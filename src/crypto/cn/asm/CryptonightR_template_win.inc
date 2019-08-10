PUBLIC CryptonightR_template_part1
PUBLIC CryptonightR_template_mainloop
PUBLIC CryptonightR_template_part2
PUBLIC CryptonightR_template_part3
PUBLIC CryptonightR_template_end
PUBLIC CryptonightR_template_double_part1
PUBLIC CryptonightR_template_double_mainloop
PUBLIC CryptonightR_template_double_part2
PUBLIC CryptonightR_template_double_part3
PUBLIC CryptonightR_template_double_part4
PUBLIC CryptonightR_template_double_end

ALIGN(64)
CryptonightR_template_part1:
	mov	rcx, [rcx]

	mov	QWORD PTR [rsp+16], rbx
	mov	QWORD PTR [rsp+24], rbp
	mov	QWORD PTR [rsp+32], rsi
	push	r10
	push	r11
	push	r12
	push	r13
	push	r14
	push	r15
	push	rdi
	sub	rsp, 64
	mov	r12, rcx
	mov	r8, QWORD PTR [r12+32]
	mov	rdx, r12
	xor	r8, QWORD PTR [r12]
	mov	r15, QWORD PTR [r12+40]
	mov	r9, r8
	xor	r15, QWORD PTR [r12+8]
	mov	r11, QWORD PTR [r12+224]
	mov	r12, QWORD PTR [r12+56]
	xor	r12, QWORD PTR [rdx+24]
	mov	rax, QWORD PTR [rdx+48]
	xor	rax, QWORD PTR [rdx+16]
	movaps	XMMWORD PTR [rsp+48], xmm6
	movq	xmm0, r12
	movaps	XMMWORD PTR [rsp+32], xmm7
	movaps	XMMWORD PTR [rsp+16], xmm8
	movaps	XMMWORD PTR [rsp], xmm9
	mov	r12, QWORD PTR [rdx+88]
	xor	r12, QWORD PTR [rdx+72]
	movq	xmm6, rax
	mov	rax, QWORD PTR [rdx+80]
	xor	rax, QWORD PTR [rdx+64]
	punpcklqdq xmm6, xmm0
	and	r9d, 2097136
	movq	xmm0, r12
	movq	xmm7, rax
	punpcklqdq xmm7, xmm0
	mov r10d, r9d
	movq	xmm9, rsp
	mov rsp, r8
	mov	r8d, 524288

	mov	ebx, [rdx+96]
	mov	esi, [rdx+100]
	mov	edi, [rdx+104]
	mov	ebp, [rdx+108]

	ALIGN(64)
CryptonightR_template_mainloop:
	movdqa	xmm5, XMMWORD PTR [r9+r11]
	movq	xmm0, r15
	movq	xmm4, rsp
	punpcklqdq xmm4, xmm0
	lea	rdx, QWORD PTR [r9+r11]

	aesenc	xmm5, xmm4

	mov	r13d, r9d
	mov	eax, r9d
	xor	r9d, 48
	xor	r13d, 16
	xor	eax, 32
	movdqu	xmm0, XMMWORD PTR [r9+r11]
	movaps xmm3, xmm0
	movdqu	xmm2, XMMWORD PTR [r13+r11]
	movdqu	xmm1, XMMWORD PTR [rax+r11]
	pxor xmm0, xmm2
	pxor xmm5, xmm1
	pxor xmm5, xmm0

	movq	r12, xmm5
	movd	r10d, xmm5
	and	r10d, 2097136

	paddq	xmm3, xmm7
	paddq	xmm2, xmm6
	paddq	xmm1, xmm4
	movdqu	XMMWORD PTR [r13+r11], xmm3
	movdqu	XMMWORD PTR [rax+r11], xmm2
	movdqu	XMMWORD PTR [r9+r11], xmm1

	movdqa	xmm0, xmm5
	pxor	xmm0, xmm6
	movdqu	XMMWORD PTR [rdx], xmm0

	lea	r13d, [ebx+esi]
	lea	edx, [edi+ebp]
	shl rdx, 32
	or	r13, rdx

	movd eax, xmm6
	movd edx, xmm7
	pextrd r9d, xmm7, 2

	xor	r13, QWORD PTR [r10+r11]
	mov	r14, QWORD PTR [r10+r11+8]

CryptonightR_template_part2:
	lea	rcx, [r10+r11]

	mov eax, edi
	mov edx, ebp
	shl rdx, 32
	or rax, rdx
	xor rsp, rax

	mov eax, ebx
	mov edx, esi
	shl rdx, 32
	or rax, rdx
	xor r15, rax

	mov	rax, r13
	mul	r12
	add	r15, rax
	add	rsp, rdx

	mov	r9d, r10d
	mov	r12d, r10d
	xor	r9d, 16
	xor	r12d, 32
	xor	r10d, 48
	movdqa	xmm1, XMMWORD PTR [r12+r11]
	movaps xmm3, xmm1
	movdqa	xmm2, XMMWORD PTR [r9+r11]
	movdqa	xmm0, XMMWORD PTR [r10+r11]
	pxor xmm1, xmm2
	pxor xmm5, xmm0
	pxor xmm5, xmm1
	paddq	xmm3, xmm4
	paddq	xmm2, xmm6
	paddq	xmm0, xmm7
	movdqu	XMMWORD PTR [r9+r11], xmm0
	movdqu	XMMWORD PTR [r12+r11], xmm2
	movdqu	XMMWORD PTR [r10+r11], xmm3

	movdqa	xmm7, xmm6
	mov	QWORD PTR [rcx], rsp
	xor	rsp, r13
	mov	r9d, esp
	mov	QWORD PTR [rcx+8], r15
	and	r9d, 2097136
	xor	r15, r14
	movdqa	xmm6, xmm5
	dec	r8d
	jnz	CryptonightR_template_mainloop

CryptonightR_template_part3:
	movq	rsp, xmm9

	mov	rbx, QWORD PTR [rsp+136]
	mov	rbp, QWORD PTR [rsp+144]
	mov	rsi, QWORD PTR [rsp+152]
	movaps	xmm6, XMMWORD PTR [rsp+48]
	movaps	xmm7, XMMWORD PTR [rsp+32]
	movaps	xmm8, XMMWORD PTR [rsp+16]
	movaps	xmm9, XMMWORD PTR [rsp]
	add	rsp, 64
	pop	rdi
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	r11
	pop	r10
	ret	0
CryptonightR_template_end:

ALIGN(64)
CryptonightR_template_double_part1:
	mov	rdx, [rcx+8]
	mov	rcx, [rcx]

	mov	QWORD PTR [rsp+24], rbx
	push	rbp
	push	rsi
	push	rdi
	push	r12
	push	r13
	push	r14
	push	r15
	sub	rsp, 320
	mov	r14, QWORD PTR [rcx+32]
	mov	r8, rcx
	xor	r14, QWORD PTR [rcx]
	mov	r12, QWORD PTR [rcx+40]
	mov	ebx, r14d
	mov	rsi, QWORD PTR [rcx+224]
	and	ebx, 2097136
	xor	r12, QWORD PTR [rcx+8]
	mov	rcx, QWORD PTR [rcx+56]
	xor	rcx, QWORD PTR [r8+24]
	mov	rax, QWORD PTR [r8+48]
	xor	rax, QWORD PTR [r8+16]
	mov	r15, QWORD PTR [rdx+32]
	xor	r15, QWORD PTR [rdx]
	movq	xmm0, rcx
	mov	rcx, QWORD PTR [r8+88]
	xor	rcx, QWORD PTR [r8+72]
	mov	r13, QWORD PTR [rdx+40]
	mov	rdi, QWORD PTR [rdx+224]
	xor	r13, QWORD PTR [rdx+8]
	movaps	XMMWORD PTR [rsp+160], xmm6
	movaps	XMMWORD PTR [rsp+176], xmm7
	movaps	XMMWORD PTR [rsp+192], xmm8
	movaps	XMMWORD PTR [rsp+208], xmm9
	movaps	XMMWORD PTR [rsp+224], xmm10
	movaps	XMMWORD PTR [rsp+240], xmm11
	movaps	XMMWORD PTR [rsp+256], xmm12
	movaps	XMMWORD PTR [rsp+272], xmm13
	movaps	XMMWORD PTR [rsp+288], xmm14
	movaps	XMMWORD PTR [rsp+304], xmm15
	movq	xmm7, rax
	mov	rax, QWORD PTR [r8+80]
	xor	rax, QWORD PTR [r8+64]

	movaps xmm1, XMMWORD PTR [rdx+96]
	movaps xmm2, XMMWORD PTR [r8+96]
	movaps XMMWORD PTR [rsp], xmm1
	movaps XMMWORD PTR [rsp+16], xmm2

	mov	r8d, r15d
	punpcklqdq xmm7, xmm0
	movq	xmm0, rcx
	mov	rcx, QWORD PTR [rdx+56]
	xor	rcx, QWORD PTR [rdx+24]
	movq	xmm9, rax
	mov	QWORD PTR [rsp+128], rsi
	mov	rax, QWORD PTR [rdx+48]
	xor	rax, QWORD PTR [rdx+16]
	punpcklqdq xmm9, xmm0
	movq	xmm0, rcx
	mov	rcx, QWORD PTR [rdx+88]
	xor	rcx, QWORD PTR [rdx+72]
	movq	xmm8, rax
	mov	QWORD PTR [rsp+136], rdi
	mov	rax, QWORD PTR [rdx+80]
	xor	rax, QWORD PTR [rdx+64]
	punpcklqdq xmm8, xmm0
	and	r8d, 2097136
	movq	xmm0, rcx
	mov	r11d, 524288
	movq	xmm10, rax
	punpcklqdq xmm10, xmm0
	
	movq xmm14, QWORD PTR [rsp+128]
	movq xmm15, QWORD PTR [rsp+136]

	ALIGN(64)
CryptonightR_template_double_mainloop:
	movdqu	xmm6, XMMWORD PTR [rbx+rsi]
	movq	xmm0, r12
	mov	ecx, ebx
	movq	xmm3, r14
	punpcklqdq xmm3, xmm0
	xor	ebx, 16
	aesenc	xmm6, xmm3
	movq	xmm4, r15
	movdqu	xmm0, XMMWORD PTR [rbx+rsi]
	pxor	xmm6, xmm0
	xor	ebx, 48
	paddq	xmm0, xmm7
	movdqu	xmm1, XMMWORD PTR [rbx+rsi]
	pxor	xmm6, xmm1
	movdqu	XMMWORD PTR [rbx+rsi], xmm0
	paddq	xmm1, xmm3
	xor	ebx, 16
	mov	eax, ebx
	xor	rax, 32
	movdqu	xmm0, XMMWORD PTR [rbx+rsi]
	pxor	xmm6, xmm0
	movq	rdx, xmm6
	movdqu	XMMWORD PTR [rbx+rsi], xmm1
	paddq	xmm0, xmm9
	movdqu	XMMWORD PTR [rax+rsi], xmm0
	movdqa	xmm0, xmm6
	pxor	xmm0, xmm7
	movdqu	XMMWORD PTR [rcx+rsi], xmm0
	mov	esi, edx
	movdqu	xmm5, XMMWORD PTR [r8+rdi]
	and	esi, 2097136
	mov	ecx, r8d
	movq	xmm0, r13
	punpcklqdq xmm4, xmm0
	xor	r8d, 16
	aesenc	xmm5, xmm4
	movdqu	xmm0, XMMWORD PTR [r8+rdi]
	pxor	xmm5, xmm0
	xor	r8d, 48
	paddq	xmm0, xmm8
	movdqu	xmm1, XMMWORD PTR [r8+rdi]
	pxor	xmm5, xmm1
	movdqu	XMMWORD PTR [r8+rdi], xmm0
	paddq	xmm1, xmm4
	xor	r8d, 16
	mov	eax, r8d
	xor	rax, 32
	movdqu	xmm0, XMMWORD PTR [r8+rdi]
	pxor	xmm5, xmm0
	movdqu	XMMWORD PTR [r8+rdi], xmm1
	paddq	xmm0, xmm10
	movdqu	XMMWORD PTR [rax+rdi], xmm0
	movdqa	xmm0, xmm5
	pxor	xmm0, xmm8
	movdqu	XMMWORD PTR [rcx+rdi], xmm0
	movq	rdi, xmm5
	movq	rcx, xmm14
	mov	ebp, edi
	mov	r8, QWORD PTR [rcx+rsi]
	mov	r10, QWORD PTR [rcx+rsi+8]
	lea	r9, QWORD PTR [rcx+rsi]
	xor	esi, 16

	movq xmm0, rsp
	movq xmm1, rsi
	movq xmm2, rdi
	movq xmm11, rbp
	movq xmm12, r15
	movq xmm13, rdx
	mov [rsp+104], rcx
	mov [rsp+112], r9

	mov ebx, DWORD PTR [rsp+16]
	mov esi, DWORD PTR [rsp+20]
	mov edi, DWORD PTR [rsp+24]
	mov ebp, DWORD PTR [rsp+28]

	lea	eax, [ebx+esi]
	lea	edx, [edi+ebp]
	shl rdx, 32
	or	rax, rdx
	xor r8, rax

	movd esp, xmm3
	pextrd r15d, xmm3, 2
	movd eax, xmm7
	movd edx, xmm9
	pextrd r9d, xmm9, 2

CryptonightR_template_double_part2:

	mov eax, edi
	mov edx, ebp
	shl rdx, 32
	or rax, rdx
	xor r14, rax

	mov eax, ebx
	mov edx, esi
	shl rdx, 32
	or rax, rdx
	xor r12, rax

	movq rsp, xmm0
	mov DWORD PTR [rsp+16], ebx
	mov DWORD PTR [rsp+20], esi
	mov DWORD PTR [rsp+24], edi
	mov DWORD PTR [rsp+28], ebp

	movq rsi, xmm1
	movq rdi, xmm2
	movq rbp, xmm11
	movq r15, xmm12
	movq rdx, xmm13
	mov rcx, [rsp+104]
	mov r9, [rsp+112]

	mov rbx, r8
	mov	rax, r8
	mul	rdx
	and	ebp, 2097136
	mov	r8, rax
	movdqu	xmm1, XMMWORD PTR [rcx+rsi]
	pxor	xmm6, xmm1
	xor	esi, 48
	paddq	xmm1, xmm7
	movdqu	xmm2, XMMWORD PTR [rsi+rcx]
	pxor	xmm6, xmm2
	paddq	xmm2, xmm3
	movdqu	XMMWORD PTR [rsi+rcx], xmm1
	xor	esi, 16
	mov	eax, esi
	mov	rsi, rcx
	movdqu	xmm0, XMMWORD PTR [rax+rcx]
	pxor	xmm6, xmm0
	movdqu	XMMWORD PTR [rax+rcx], xmm2
	paddq	xmm0, xmm9
	add	r12, r8
	xor	rax, 32
	add	r14, rdx
	movdqa	xmm9, xmm7
	movdqa	xmm7, xmm6
	movdqu	XMMWORD PTR [rax+rcx], xmm0
	mov	QWORD PTR [r9+8], r12
	xor	r12, r10
	mov	QWORD PTR [r9], r14
	movq rcx, xmm15
	xor	r14, rbx
	mov	r10d, ebp
	mov	ebx, r14d
	xor	ebp, 16
	and	ebx, 2097136
	mov	r8, QWORD PTR [r10+rcx]
	mov	r9, QWORD PTR [r10+rcx+8]

	movq xmm0, rsp
	movq xmm1, rbx
	movq xmm2, rsi
	movq xmm11, rdi
	movq xmm12, rbp
	movq xmm13, r15
	mov [rsp+104], rcx
	mov [rsp+112], r9

	mov ebx, DWORD PTR [rsp]
	mov esi, DWORD PTR [rsp+4]
	mov edi, DWORD PTR [rsp+8]
	mov ebp, DWORD PTR [rsp+12]

	lea	eax, [ebx+esi]
	lea	edx, [edi+ebp]
	shl rdx, 32
	or	rax, rdx

	xor r8, rax
	movq xmm3, r8

	movd esp, xmm4
	pextrd r15d, xmm4, 2
	movd eax, xmm8
	movd edx, xmm10
	pextrd r9d, xmm10, 2

CryptonightR_template_double_part3:

	movq r15, xmm13

	mov eax, edi
	mov edx, ebp
	shl rdx, 32
	or rax, rdx
	xor r15, rax

	mov eax, ebx
	mov edx, esi
	shl rdx, 32
	or rax, rdx
	xor r13, rax

	movq rsp, xmm0
	mov DWORD PTR [rsp], ebx
	mov DWORD PTR [rsp+4], esi
	mov DWORD PTR [rsp+8], edi
	mov DWORD PTR [rsp+12], ebp

	movq rbx, xmm1
	movq rsi, xmm2
	movq rdi, xmm11
	movq rbp, xmm12
	mov rcx, [rsp+104]
	mov r9, [rsp+112]

	mov rax, r8
	mul	rdi
	mov	rdi, rcx
	mov	r8, rax
	movdqu	xmm1, XMMWORD PTR [rbp+rcx]
	pxor xmm5, xmm1
	xor	ebp, 48
	paddq	xmm1, xmm8
	add	r13, r8
	movdqu	xmm2, XMMWORD PTR [rbp+rcx]
	pxor xmm5, xmm2
	add	r15, rdx
	movdqu	XMMWORD PTR [rbp+rcx], xmm1
	paddq	xmm2, xmm4
	xor	ebp, 16
	mov	eax, ebp
	xor	rax, 32
	movdqu	xmm0, XMMWORD PTR [rbp+rcx]
	pxor xmm5, xmm0
	movdqu	XMMWORD PTR [rbp+rcx], xmm2
	paddq	xmm0, xmm10
	movdqu	XMMWORD PTR [rax+rcx], xmm0
	movq rax, xmm3
	movdqa	xmm10, xmm8
	mov	QWORD PTR [r10+rcx], r15
	movdqa	xmm8, xmm5
	xor	r15, rax
	mov	QWORD PTR [r10+rcx+8], r13
	mov	r8d, r15d
	xor	r13, r9
	and	r8d, 2097136
	dec r11d
	jnz	CryptonightR_template_double_mainloop

CryptonightR_template_double_part4:

	mov	rbx, QWORD PTR [rsp+400]
	movaps	xmm6, XMMWORD PTR [rsp+160]
	movaps	xmm7, XMMWORD PTR [rsp+176]
	movaps	xmm8, XMMWORD PTR [rsp+192]
	movaps	xmm9, XMMWORD PTR [rsp+208]
	movaps	xmm10, XMMWORD PTR [rsp+224]
	movaps	xmm11, XMMWORD PTR [rsp+240]
	movaps	xmm12, XMMWORD PTR [rsp+256]
	movaps	xmm13, XMMWORD PTR [rsp+272]
	movaps	xmm14, XMMWORD PTR [rsp+288]
	movaps	xmm15, XMMWORD PTR [rsp+304]
	add	rsp, 320
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rdi
	pop	rsi
	pop	rbp
	ret	0
CryptonightR_template_double_end:
