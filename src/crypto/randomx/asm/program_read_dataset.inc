	xor rbp, rax                       ;# modify "mx"
	mov edx, ebp                       ;# edx = mx
	and edx, RANDOMX_DATASET_BASE_MASK
	prefetchnta byte ptr [rdi+rdx]
	ror rbp, 32                        ;# swap "ma" and "mx"
	mov edx, ebp                       ;# edx = ma
	and edx, RANDOMX_DATASET_BASE_MASK
	lea rcx, [rdi+rdx]                 ;# dataset cache line
	xor r8,  qword ptr [rcx+0]
	xor r9,  qword ptr [rcx+8]
	xor r10, qword ptr [rcx+16]
	xor r11, qword ptr [rcx+24]
	xor r12, qword ptr [rcx+32]
	xor r13, qword ptr [rcx+40]
	xor r14, qword ptr [rcx+48]
	xor r15, qword ptr [rcx+56]
	