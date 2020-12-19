	add rsp, 40
	pop r9

	movdqu xmm0,  xmmword ptr [rsp]
	movdqu xmm1,  xmmword ptr [rsp + 16]
	movdqu xmm2,  xmmword ptr [rsp + 32]
	movdqu xmm3,  xmmword ptr [rsp + 48]
	movdqu xmm4,  xmmword ptr [rsp + 64]
	movdqu xmm5,  xmmword ptr [rsp + 80]
	movdqu xmm6,  xmmword ptr [rsp + 96]
	movdqu xmm7,  xmmword ptr [rsp + 112]
	movdqu xmm8,  xmmword ptr [rsp + 128]
	movdqu xmm9,  xmmword ptr [rsp + 144]
	movdqu xmm10, xmmword ptr [rsp + 160]
	movdqu xmm11, xmmword ptr [rsp + 176]
	movdqu xmm12, xmmword ptr [rsp + 192]
	movdqu xmm13, xmmword ptr [rsp + 208]
	movdqu xmm14, xmmword ptr [rsp + 224]
	movdqu xmm15, xmmword ptr [rsp + 240]
	vzeroupper
	add rsp, 256

	pop r15
	pop r14
	pop r13
	pop r12
	pop rsi
	pop rdi
	pop rbp
	pop rbx
	ret
