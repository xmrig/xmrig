	sub rsp, 200
	mov qword ptr [rsp+64], rbx
	mov qword ptr [rsp+56], r8
	mov qword ptr [rsp+48], r9
	mov qword ptr [rsp+40], r10
	mov qword ptr [rsp+32], r11
	mov qword ptr [rsp+24], r12
	mov qword ptr [rsp+16], r13
	mov qword ptr [rsp+8], r14
	mov qword ptr [rsp+0], r15
	ror rbp, 32                        ;# swap "ma" and "mx"
	xor rbp, rax                       ;# modify "mx"
	mov rbx, rbp                       ;# ebx = ma
	shr rbx, 38
	and ebx, RANDOMX_DATASET_BASE_MASK / 64 ;# ebx = Dataset block number
	;# add ebx, datasetOffset / 64
	;# call 32768