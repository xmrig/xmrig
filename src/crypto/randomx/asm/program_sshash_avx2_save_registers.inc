	push rbx
	push rbp
	push rdi
	push rsi
	push r12
	push r13
	push r14
	push r15

	;# save all XMM registers just to be safe for all calling conventions
	sub rsp, 256
	movdqu xmmword ptr [rsp], xmm0
	movdqu xmmword ptr [rsp + 16], xmm1
	movdqu xmmword ptr [rsp + 32], xmm2
	movdqu xmmword ptr [rsp + 48], xmm3
	movdqu xmmword ptr [rsp + 64], xmm4
	movdqu xmmword ptr [rsp + 80], xmm5
	movdqu xmmword ptr [rsp + 96], xmm6
	movdqu xmmword ptr [rsp + 112], xmm7
	movdqu xmmword ptr [rsp + 128], xmm8
	movdqu xmmword ptr [rsp + 144], xmm9
	movdqu xmmword ptr [rsp + 160], xmm10
	movdqu xmmword ptr [rsp + 176], xmm11
	movdqu xmmword ptr [rsp + 192], xmm12
	movdqu xmmword ptr [rsp + 208], xmm13
	movdqu xmmword ptr [rsp + 224], xmm14
	movdqu xmmword ptr [rsp + 240], xmm15
