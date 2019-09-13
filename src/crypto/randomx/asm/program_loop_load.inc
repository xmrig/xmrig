	lea rcx, [rsi+rax]
	push rcx
	xor r8,  qword ptr [rcx+0]
	xor r9,  qword ptr [rcx+8]
	xor r10, qword ptr [rcx+16]
	xor r11, qword ptr [rcx+24]
	xor r12, qword ptr [rcx+32]
	xor r13, qword ptr [rcx+40]
	xor r14, qword ptr [rcx+48]
	xor r15, qword ptr [rcx+56]
	lea rcx, [rsi+rdx]
	push rcx
	cvtdq2pd xmm0, qword ptr [rcx+0]
	cvtdq2pd xmm1, qword ptr [rcx+8]
	cvtdq2pd xmm2, qword ptr [rcx+16]
	cvtdq2pd xmm3, qword ptr [rcx+24]
	cvtdq2pd xmm4, qword ptr [rcx+32]
	cvtdq2pd xmm5, qword ptr [rcx+40]
	cvtdq2pd xmm6, qword ptr [rcx+48]
	cvtdq2pd xmm7, qword ptr [rcx+56]
	andps xmm4, xmm13
	andps xmm5, xmm13
	andps xmm6, xmm13
	andps xmm7, xmm13
	orps xmm4, xmm14
	orps xmm5, xmm14
	orps xmm6, xmm14
	orps xmm7, xmm14
