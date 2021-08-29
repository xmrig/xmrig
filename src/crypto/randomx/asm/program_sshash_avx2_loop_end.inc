	mov qword ptr [rsi+0], r8
	vpunpcklqdq ymm8, ymm0, ymm1
	mov qword ptr [rsi+8], r9
	vpunpcklqdq ymm9, ymm2, ymm3
	mov qword ptr [rsi+16], r10
	vpunpcklqdq ymm10, ymm4, ymm5
	mov qword ptr [rsi+24], r11
	vpunpcklqdq ymm11, ymm6, ymm7
	mov qword ptr [rsi+32], r12
	vpunpckhqdq ymm12, ymm0, ymm1
	mov qword ptr [rsi+40], r13
	vpunpckhqdq ymm13, ymm2, ymm3
	mov qword ptr [rsi+48], r14
	vpunpckhqdq ymm14, ymm4, ymm5
	mov qword ptr [rsi+56], r15
	vpunpckhqdq ymm15, ymm6, ymm7

	vperm2i128 ymm0, ymm8, ymm9, 32
	vperm2i128 ymm1, ymm10, ymm11, 32
	vmovdqu ymmword ptr [rsi+64], ymm0
	vmovdqu ymmword ptr [rsi+96], ymm1
	vperm2i128 ymm2, ymm12, ymm13, 32
	vperm2i128 ymm3, ymm14, ymm15, 32
	vmovdqu ymmword ptr [rsi+128], ymm2
	vmovdqu ymmword ptr [rsi+160], ymm3
	vperm2i128 ymm4, ymm8, ymm9, 49
	vperm2i128 ymm5, ymm10, ymm11, 49
	vmovdqu ymmword ptr [rsi+192], ymm4
	vmovdqu ymmword ptr [rsi+224], ymm5
	vperm2i128 ymm6, ymm12, ymm13, 49
	vperm2i128 ymm7, ymm14, ymm15, 49
	vmovdqu ymmword ptr [rsi+256], ymm6
	vmovdqu ymmword ptr [rsi+288], ymm7

	add rbp, 5
	add rsi, 320
	cmp rbp, qword ptr [rsp+40]
	db 15, 130, 0, 0, 0, 0		;# jb rel32
