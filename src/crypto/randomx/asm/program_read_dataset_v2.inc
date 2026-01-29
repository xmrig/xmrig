	mov ecx, ebp                       ;# ecx = ma
	and ecx, RANDOMX_DATASET_BASE_MASK
	xor r8, qword ptr [rdi+rcx]
	xor rbp, rax                       ;# modify "ma"
	mov edx, ebp                       ;# edx = "ma"
	ror rbp, 32                        ;# swap "ma" and "mx"
	and edx, RANDOMX_DATASET_BASE_MASK
	prefetchnta byte ptr [rdi+rdx]
	xor r9,  qword ptr [rdi+rcx+8]
	xor r10, qword ptr [rdi+rcx+16]
	xor r11, qword ptr [rdi+rcx+24]
	xor r12, qword ptr [rdi+rcx+32]
	xor r13, qword ptr [rdi+rcx+40]
	xor r14, qword ptr [rdi+rcx+48]
	xor r15, qword ptr [rdi+rcx+56]
	