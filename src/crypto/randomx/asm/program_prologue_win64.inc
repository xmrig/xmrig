	;# callee-saved registers - Microsoft x64 calling convention
	push rbx
	push rbp
	push rdi
	push rsi
	push r12
	push r13
	push r14
	push r15
	sub rsp, 80
	movdqu xmmword ptr [rsp+64], xmm6
	movdqu xmmword ptr [rsp+48], xmm7
	movdqu xmmword ptr [rsp+32], xmm8
	movdqu xmmword ptr [rsp+16], xmm9
	movdqu xmmword ptr [rsp+0], xmm10
	sub rsp, 80
	movdqu xmmword ptr [rsp+64], xmm11
	movdqu xmmword ptr [rsp+48], xmm12
	movdqu xmmword ptr [rsp+32], xmm13
	movdqu xmmword ptr [rsp+16], xmm14
	movdqu xmmword ptr [rsp+0], xmm15

	;# function arguments
	push rcx                    ;# RegisterFile& registerFile
	mov rbp, qword ptr [rdx]    ;# "mx", "ma"
	mov rdi, qword ptr [rdx+8]  ;# uint8_t* dataset

	;# dataset prefetch for the first iteration of the main loop
	mov rax, rbp
	shr rax, 32
	and eax, RANDOMX_DATASET_BASE_MASK
	prefetchnta byte ptr [rdi+rax]

	mov rsi, r8                 ;# uint8_t* scratchpad
	mov rbx, r9                 ;# loop counter

	mov rax, rbp
	ror rbp, 32

	;# zero integer registers
	xor r8, r8
	xor r9, r9
	xor r10, r10
	xor r11, r11
	xor r12, r12
	xor r13, r13
	xor r14, r14
	xor r15, r15

	;# load constant registers
	lea rcx, [rcx+120]
	movapd xmm8, xmmword ptr [rcx+72]
	movapd xmm9, xmmword ptr [rcx+88]
	movapd xmm10, xmmword ptr [rcx+104]
	movapd xmm11, xmmword ptr [rcx+120]
