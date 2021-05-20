	;# callee-saved registers - System V AMD64 ABI
	push rbx
	push rbp
	push r12
	push r13
	push r14
	push r15

	;# function arguments
	mov rbx, rcx                ;# loop counter
	push rdi                    ;# RegisterFile& registerFile
	mov rcx, rdi
	mov rbp, qword ptr [rsi]    ;# "mx", "ma"
	mov rdi, qword ptr [rsi+8]  ;# uint8_t* dataset

	;# dataset prefetch for the first iteration of the main loop
	mov rax, rbp
	shr rax, 32
	and eax, RANDOMX_DATASET_BASE_MASK
	prefetchnta byte ptr [rdi+rax]

	mov rsi, rdx                ;# uint8_t* scratchpad

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
