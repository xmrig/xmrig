
.code
; procedure exec_cpuid
; Signature: void exec_cpiud(uint32_t *regs)
exec_cpuid Proc
	push	rbx
	push	rcx
	push	rdx
	push	rdi
	
	mov	rdi,	rcx
	
	mov	eax,	[rdi]
	mov	ebx,	[rdi+4]
	mov	ecx,	[rdi+8]
	mov	edx,	[rdi+12]
	
	cpuid
	
	mov	[rdi],	eax
	mov	[rdi+4],	ebx
	mov	[rdi+8],	ecx
	mov	[rdi+12],	edx
	pop	rdi
	pop	rdx
	pop	rcx
	pop	rbx
	ret
exec_cpuid endp

; procedure cpu_rdtsc
; Signature: void cpu_rdtsc(uint64_t *result)
cpu_rdtsc Proc
	push	rdx
	rdtsc
	mov	[rcx],	eax
	mov	[rcx+4],	edx
	pop	rdx
	ret
cpu_rdtsc endp

; procedure busy_sse_loop
; Signature: void busy_sse_loop(int cycles)
busy_sse_loop Proc
	; save xmm6 & xmm7 into the shadow area, as Visual C++ 2008
	; expects that we don't touch them:
	movups	[rsp + 8],	xmm6
	movups	[rsp + 24],	xmm7

	xorps	xmm0,	xmm0
	xorps	xmm1,	xmm1
	xorps	xmm2,	xmm2
	xorps	xmm3,	xmm3
	xorps	xmm4,	xmm4
	xorps	xmm5,	xmm5
	xorps	xmm6,	xmm6
	xorps	xmm7,	xmm7
	; --
	align 16
bsLoop:
	; 0:
	addps	xmm0,	xmm1
	addps	xmm1,	xmm2
	addps	xmm2,	xmm3
	addps	xmm3,	xmm4
	addps	xmm4,	xmm5
	addps	xmm5,	xmm6
	addps	xmm6,	xmm7
	addps	xmm7,	xmm0
	; 1:
	addps	xmm0,	xmm1
	addps	xmm1,	xmm2
	addps	xmm2,	xmm3
	addps	xmm3,	xmm4
	addps	xmm4,	xmm5
	addps	xmm5,	xmm6
	addps	xmm6,	xmm7
	addps	xmm7,	xmm0
	; 2:
	addps	xmm0,	xmm1
	addps	xmm1,	xmm2
	addps	xmm2,	xmm3
	addps	xmm3,	xmm4
	addps	xmm4,	xmm5
	addps	xmm5,	xmm6
	addps	xmm6,	xmm7
	addps	xmm7,	xmm0
	; 3:
	addps	xmm0,	xmm1
	addps	xmm1,	xmm2
	addps	xmm2,	xmm3
	addps	xmm3,	xmm4
	addps	xmm4,	xmm5
	addps	xmm5,	xmm6
	addps	xmm6,	xmm7
	addps	xmm7,	xmm0
	; 4:
	addps	xmm0,	xmm1
	addps	xmm1,	xmm2
	addps	xmm2,	xmm3
	addps	xmm3,	xmm4
	addps	xmm4,	xmm5
	addps	xmm5,	xmm6
	addps	xmm6,	xmm7
	addps	xmm7,	xmm0
	; 5:
	addps	xmm0,	xmm1
	addps	xmm1,	xmm2
	addps	xmm2,	xmm3
	addps	xmm3,	xmm4
	addps	xmm4,	xmm5
	addps	xmm5,	xmm6
	addps	xmm6,	xmm7
	addps	xmm7,	xmm0
	; 6:
	addps	xmm0,	xmm1
	addps	xmm1,	xmm2
	addps	xmm2,	xmm3
	addps	xmm3,	xmm4
	addps	xmm4,	xmm5
	addps	xmm5,	xmm6
	addps	xmm6,	xmm7
	addps	xmm7,	xmm0
	; 7:
	addps	xmm0,	xmm1
	addps	xmm1,	xmm2
	addps	xmm2,	xmm3
	addps	xmm3,	xmm4
	addps	xmm4,	xmm5
	addps	xmm5,	xmm6
	addps	xmm6,	xmm7
	addps	xmm7,	xmm0
	; 8:
	addps	xmm0,	xmm1
	addps	xmm1,	xmm2
	addps	xmm2,	xmm3
	addps	xmm3,	xmm4
	addps	xmm4,	xmm5
	addps	xmm5,	xmm6
	addps	xmm6,	xmm7
	addps	xmm7,	xmm0
	; 9:
	addps	xmm0,	xmm1
	addps	xmm1,	xmm2
	addps	xmm2,	xmm3
	addps	xmm3,	xmm4
	addps	xmm4,	xmm5
	addps	xmm5,	xmm6
	addps	xmm6,	xmm7
	addps	xmm7,	xmm0
	; 10:
	addps	xmm0,	xmm1
	addps	xmm1,	xmm2
	addps	xmm2,	xmm3
	addps	xmm3,	xmm4
	addps	xmm4,	xmm5
	addps	xmm5,	xmm6
	addps	xmm6,	xmm7
	addps	xmm7,	xmm0
	; 11:
	addps	xmm0,	xmm1
	addps	xmm1,	xmm2
	addps	xmm2,	xmm3
	addps	xmm3,	xmm4
	addps	xmm4,	xmm5
	addps	xmm5,	xmm6
	addps	xmm6,	xmm7
	addps	xmm7,	xmm0
	; 12:
	addps	xmm0,	xmm1
	addps	xmm1,	xmm2
	addps	xmm2,	xmm3
	addps	xmm3,	xmm4
	addps	xmm4,	xmm5
	addps	xmm5,	xmm6
	addps	xmm6,	xmm7
	addps	xmm7,	xmm0
	; 13:
	addps	xmm0,	xmm1
	addps	xmm1,	xmm2
	addps	xmm2,	xmm3
	addps	xmm3,	xmm4
	addps	xmm4,	xmm5
	addps	xmm5,	xmm6
	addps	xmm6,	xmm7
	addps	xmm7,	xmm0
	; 14:
	addps	xmm0,	xmm1
	addps	xmm1,	xmm2
	addps	xmm2,	xmm3
	addps	xmm3,	xmm4
	addps	xmm4,	xmm5
	addps	xmm5,	xmm6
	addps	xmm6,	xmm7
	addps	xmm7,	xmm0
	; 15:
	addps	xmm0,	xmm1
	addps	xmm1,	xmm2
	addps	xmm2,	xmm3
	addps	xmm3,	xmm4
	addps	xmm4,	xmm5
	addps	xmm5,	xmm6
	addps	xmm6,	xmm7
	addps	xmm7,	xmm0
	; 16:
	addps	xmm0,	xmm1
	addps	xmm1,	xmm2
	addps	xmm2,	xmm3
	addps	xmm3,	xmm4
	addps	xmm4,	xmm5
	addps	xmm5,	xmm6
	addps	xmm6,	xmm7
	addps	xmm7,	xmm0
	; 17:
	addps	xmm0,	xmm1
	addps	xmm1,	xmm2
	addps	xmm2,	xmm3
	addps	xmm3,	xmm4
	addps	xmm4,	xmm5
	addps	xmm5,	xmm6
	addps	xmm6,	xmm7
	addps	xmm7,	xmm0
	; 18:
	addps	xmm0,	xmm1
	addps	xmm1,	xmm2
	addps	xmm2,	xmm3
	addps	xmm3,	xmm4
	addps	xmm4,	xmm5
	addps	xmm5,	xmm6
	addps	xmm6,	xmm7
	addps	xmm7,	xmm0
	; 19:
	addps	xmm0,	xmm1
	addps	xmm1,	xmm2
	addps	xmm2,	xmm3
	addps	xmm3,	xmm4
	addps	xmm4,	xmm5
	addps	xmm5,	xmm6
	addps	xmm6,	xmm7
	addps	xmm7,	xmm0
	; 20:
	addps	xmm0,	xmm1
	addps	xmm1,	xmm2
	addps	xmm2,	xmm3
	addps	xmm3,	xmm4
	addps	xmm4,	xmm5
	addps	xmm5,	xmm6
	addps	xmm6,	xmm7
	addps	xmm7,	xmm0
	; 21:
	addps	xmm0,	xmm1
	addps	xmm1,	xmm2
	addps	xmm2,	xmm3
	addps	xmm3,	xmm4
	addps	xmm4,	xmm5
	addps	xmm5,	xmm6
	addps	xmm6,	xmm7
	addps	xmm7,	xmm0
	; 22:
	addps	xmm0,	xmm1
	addps	xmm1,	xmm2
	addps	xmm2,	xmm3
	addps	xmm3,	xmm4
	addps	xmm4,	xmm5
	addps	xmm5,	xmm6
	addps	xmm6,	xmm7
	addps	xmm7,	xmm0
	; 23:
	addps	xmm0,	xmm1
	addps	xmm1,	xmm2
	addps	xmm2,	xmm3
	addps	xmm3,	xmm4
	addps	xmm4,	xmm5
	addps	xmm5,	xmm6
	addps	xmm6,	xmm7
	addps	xmm7,	xmm0
	; 24:
	addps	xmm0,	xmm1
	addps	xmm1,	xmm2
	addps	xmm2,	xmm3
	addps	xmm3,	xmm4
	addps	xmm4,	xmm5
	addps	xmm5,	xmm6
	addps	xmm6,	xmm7
	addps	xmm7,	xmm0
	; 25:
	addps	xmm0,	xmm1
	addps	xmm1,	xmm2
	addps	xmm2,	xmm3
	addps	xmm3,	xmm4
	addps	xmm4,	xmm5
	addps	xmm5,	xmm6
	addps	xmm6,	xmm7
	addps	xmm7,	xmm0
	; 26:
	addps	xmm0,	xmm1
	addps	xmm1,	xmm2
	addps	xmm2,	xmm3
	addps	xmm3,	xmm4
	addps	xmm4,	xmm5
	addps	xmm5,	xmm6
	addps	xmm6,	xmm7
	addps	xmm7,	xmm0
	; 27:
	addps	xmm0,	xmm1
	addps	xmm1,	xmm2
	addps	xmm2,	xmm3
	addps	xmm3,	xmm4
	addps	xmm4,	xmm5
	addps	xmm5,	xmm6
	addps	xmm6,	xmm7
	addps	xmm7,	xmm0
	; 28:
	addps	xmm0,	xmm1
	addps	xmm1,	xmm2
	addps	xmm2,	xmm3
	addps	xmm3,	xmm4
	addps	xmm4,	xmm5
	addps	xmm5,	xmm6
	addps	xmm6,	xmm7
	addps	xmm7,	xmm0
	; 29:
	addps	xmm0,	xmm1
	addps	xmm1,	xmm2
	addps	xmm2,	xmm3
	addps	xmm3,	xmm4
	addps	xmm4,	xmm5
	addps	xmm5,	xmm6
	addps	xmm6,	xmm7
	addps	xmm7,	xmm0
	; 30:
	addps	xmm0,	xmm1
	addps	xmm1,	xmm2
	addps	xmm2,	xmm3
	addps	xmm3,	xmm4
	addps	xmm4,	xmm5
	addps	xmm5,	xmm6
	addps	xmm6,	xmm7
	addps	xmm7,	xmm0
	; 31:
	addps	xmm0,	xmm1
	addps	xmm1,	xmm2
	addps	xmm2,	xmm3
	addps	xmm3,	xmm4
	addps	xmm4,	xmm5
	addps	xmm5,	xmm6
	addps	xmm6,	xmm7
	addps	xmm7,	xmm0
	; ----------------------
	dec		ecx
	jnz		bsLoop

	; restore xmm6 & xmm7:
	movups	xmm6,	[rsp + 8]
	movups	xmm7,	[rsp + 24]
	ret
busy_sse_loop endp

END
