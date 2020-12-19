	sub rsp, 40
	mov [rsp], rbx
	vmovdqu ymmword ptr [rsp+8], ymm14

	mov rax, [rsp+40]
	mov rbx, [rsp+48]
	mov rcx, [rsp+56]
	mov rdx, [rsp+64]

	vmovdqu ymm8, ymmword ptr [rax]			;# ymm8  = r0[1], r1[1], r2[1], r3[1]
	vmovdqu ymm9, ymmword ptr [rbx]			;# ymm9  = r0[2], r1[2], r2[2], r3[2]
	vmovdqu ymm10, ymmword ptr [rcx]		;# ymm10 = r0[3], r1[3], r2[3], r3[3]
	vmovdqu ymm11, ymmword ptr [rdx]		;# ymm11 = r0[4], r1[4], r2[4], r3[4]

	vpunpcklqdq ymm12, ymm8, ymm9			;# ymm12 = r0[1], r0[2], r2[1], r2[2]
	vpunpcklqdq ymm13, ymm10, ymm11			;# ymm13 = r0[3], r0[4], r2[3], r2[4]
	vperm2i128 ymm14, ymm12, ymm13, 32		;# ymm14 = r0[1], r0[2], r0[3], r0[4]
	vpxor ymm0, ymm0, ymm14
	vperm2i128 ymm14, ymm12, ymm13, 49		;# ymm14 = r2[1], r2[2], r2[3], r2[4]
	vpxor ymm2, ymm2, ymm14

	vpunpckhqdq ymm12, ymm8, ymm9			;# ymm12 = r1[1], r1[2], r3[1], r3[2]
	vpunpckhqdq ymm13, ymm10, ymm11			;# ymm13 = r1[3], r1[4], r3[3], r3[4]
	vperm2i128 ymm14, ymm12, ymm13, 32		;# ymm14 = r1[1], r1[2], r1[3], r1[4]
	vpxor ymm1, ymm1, ymm14
	vperm2i128 ymm14, ymm12, ymm13, 49		;# ymm14 = r3[1], r3[2], r3[3], r3[4]
	vpxor ymm3, ymm3, ymm14

	vmovdqu ymm8, ymmword ptr [rax+32]		;# ymm8  = r4[1], r5[1], r6[1], r7[1]
	vmovdqu ymm9, ymmword ptr [rbx+32]		;# ymm9  = r4[2], r5[2], r6[2], r7[2]
	vmovdqu ymm10, ymmword ptr [rcx+32]		;# ymm10 = r4[3], r5[3], r6[3], r7[3]
	vmovdqu ymm11, ymmword ptr [rdx+32]		;# ymm11 = r4[4], r5[4], r6[4], r7[4]

	vpunpcklqdq ymm12, ymm8, ymm9			;# ymm12 = r4[1], r4[2], r6[1], r6[2]
	vpunpcklqdq ymm13, ymm10, ymm11			;# ymm13 = r4[3], r4[4], r6[3], r6[4]
	vperm2i128 ymm14, ymm12, ymm13, 32		;# ymm14 = r4[1], r4[2], r4[3], r4[4]
	vpxor ymm4, ymm4, ymm14
	vperm2i128 ymm14, ymm12, ymm13, 49		;# ymm14 = r6[1], r6[2], r6[3], r6[4]
	vpxor ymm6, ymm6, ymm14

	vpunpckhqdq ymm12, ymm8, ymm9			;# ymm12 = r5[1], r5[2], r7[1], r7[2]
	vpunpckhqdq ymm13, ymm10, ymm11			;# ymm13 = r5[3], r5[4], r7[3], r7[4]
	vperm2i128 ymm14, ymm12, ymm13, 32		;# ymm14 = r5[1], r5[2], r5[3], r5[4]
	vpxor ymm5, ymm5, ymm14
	vperm2i128 ymm14, ymm12, ymm13, 49		;# ymm14 = r7[1], r7[2], r7[3], r7[4]
	vpxor ymm7, ymm7, ymm14

	mov rbx, [rsp]
	vmovdqu ymm14, ymmword ptr [rsp+8]
	add rsp, 40
