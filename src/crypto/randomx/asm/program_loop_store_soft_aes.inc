	mov rcx, [rsp+24]
	mov qword ptr [rcx+0], r8
	mov qword ptr [rcx+8], r9
	mov qword ptr [rcx+16], r10
	mov qword ptr [rcx+24], r11
	mov qword ptr [rcx+32], r12
	mov qword ptr [rcx+40], r13
	mov qword ptr [rcx+48], r14
	mov qword ptr [rcx+56], r15

	movapd xmmword ptr [rsp+40], xmm0
	movapd xmmword ptr [rsp+56], xmm1
	movapd xmmword ptr [rsp+72], xmm2
	movapd xmmword ptr [rsp+88], xmm3
	movapd xmmword ptr [rsp+104], xmm4
	movapd xmmword ptr [rsp+120], xmm5
	movapd xmmword ptr [rsp+136], xmm6
	movapd xmmword ptr [rsp+152], xmm7

	mov [rsp+168], rax
	mov [rsp+176], rbx
	mov [rsp+184], rdx
	mov [rsp+192], rsi
	mov [rsp+200], rdi
	mov [rsp+208], rbp
	mov [rsp+216], r8
	mov [rsp+224], r9

	mov r8, [rsp+232] ;# aes_lut_enc
	mov r9, [rsp+240] ;# aes_lut_dec

	movapd xmm12, xmmword ptr [rsp-8] ;# "call" will overwrite IMUL_RCP's data on stack, so save it

	lea rsi, [rsp+104]
	lea rdi, [rsp+40]
	call soft_aes_enc
	lea rdi, [rsp+56]
	call soft_aes_dec
	lea rdi, [rsp+72]
	call soft_aes_enc
	lea rdi, [rsp+88]
	call soft_aes_dec

	lea rsi, [rsp+120]
	lea rdi, [rsp+40]
	call soft_aes_enc
	lea rdi, [rsp+56]
	call soft_aes_dec
	lea rdi, [rsp+72]
	call soft_aes_enc
	lea rdi, [rsp+88]
	call soft_aes_dec

	lea rsi, [rsp+136]
	lea rdi, [rsp+40]
	call soft_aes_enc
	lea rdi, [rsp+56]
	call soft_aes_dec
	lea rdi, [rsp+72]
	call soft_aes_enc
	lea rdi, [rsp+88]
	call soft_aes_dec

	lea rsi, [rsp+152]
	lea rdi, [rsp+40]
	call soft_aes_enc
	lea rdi, [rsp+56]
	call soft_aes_dec
	lea rdi, [rsp+72]
	call soft_aes_enc
	lea rdi, [rsp+88]
	call soft_aes_dec

	movapd xmmword ptr [rsp-8], xmm12

	jmp soft_aes_end

soft_aes_enc:
	mov eax, dword ptr [rsi+0]
	mov ebx, dword ptr [rsi+4]
	mov ecx, dword ptr [rsi+8]
	mov edx, dword ptr [rsi+12]

	movzx ebp, byte ptr [rdi+0]
	xor eax, dword ptr [r8+rbp*4]
	movzx ebp, byte ptr [rdi+1]
	xor edx, dword ptr [r8+rbp*4+1024]
	movzx ebp, byte ptr [rdi+2]
	xor ecx, dword ptr [r8+rbp*4+2048]
	movzx ebp, byte ptr [rdi+3]
	xor ebx, dword ptr [r8+rbp*4+3072]

	movzx ebp, byte ptr [rdi+4]
	xor ebx, dword ptr [r8+rbp*4]
	movzx ebp, byte ptr [rdi+5]
	xor eax, dword ptr [r8+rbp*4+1024]
	movzx ebp, byte ptr [rdi+6]
	xor edx, dword ptr [r8+rbp*4+2048]
	movzx ebp, byte ptr [rdi+7]
	xor ecx, dword ptr [r8+rbp*4+3072]

	movzx ebp, byte ptr [rdi+8]
	xor ecx, dword ptr [r8+rbp*4]
	movzx ebp, byte ptr [rdi+9]
	xor ebx, dword ptr [r8+rbp*4+1024]
	movzx ebp, byte ptr [rdi+10]
	xor eax, dword ptr [r8+rbp*4+2048]
	movzx ebp, byte ptr [rdi+11]
	xor edx, dword ptr [r8+rbp*4+3072]

	movzx ebp, byte ptr [rdi+12]
	xor edx, dword ptr [r8+rbp*4]
	movzx ebp, byte ptr [rdi+13]
	xor ecx, dword ptr [r8+rbp*4+1024]
	movzx ebp, byte ptr [rdi+14]
	xor ebx, dword ptr [r8+rbp*4+2048]
	movzx ebp, byte ptr [rdi+15]
	xor eax, dword ptr [r8+rbp*4+3072]

	mov dword ptr [rdi+0], eax
	mov dword ptr [rdi+4], ebx
	mov dword ptr [rdi+8], ecx
	mov dword ptr [rdi+12], edx

	ret

soft_aes_dec:
	mov eax, dword ptr [rsi+0]
	mov ebx, dword ptr [rsi+4]
	mov ecx, dword ptr [rsi+8]
	mov edx, dword ptr [rsi+12]

	movzx ebp, byte ptr [rdi+0]
	xor eax, dword ptr [r9+rbp*4]
	movzx ebp, byte ptr [rdi+1]
	xor ebx, dword ptr [r9+rbp*4+1024]
	movzx ebp, byte ptr [rdi+2]
	xor ecx, dword ptr [r9+rbp*4+2048]
	movzx ebp, byte ptr [rdi+3]
	xor edx, dword ptr [r9+rbp*4+3072]

	movzx ebp, byte ptr [rdi+4]
	xor ebx, dword ptr [r9+rbp*4]
	movzx ebp, byte ptr [rdi+5]
	xor ecx, dword ptr [r9+rbp*4+1024]
	movzx ebp, byte ptr [rdi+6]
	xor edx, dword ptr [r9+rbp*4+2048]
	movzx ebp, byte ptr [rdi+7]
	xor eax, dword ptr [r9+rbp*4+3072]

	movzx ebp, byte ptr [rdi+8]
	xor ecx, dword ptr [r9+rbp*4]
	movzx ebp, byte ptr [rdi+9]
	xor edx, dword ptr [r9+rbp*4+1024]
	movzx ebp, byte ptr [rdi+10]
	xor eax, dword ptr [r9+rbp*4+2048]
	movzx ebp, byte ptr [rdi+11]
	xor ebx, dword ptr [r9+rbp*4+3072]

	movzx ebp, byte ptr [rdi+12]
	xor edx, dword ptr [r9+rbp*4]
	movzx ebp, byte ptr [rdi+13]
	xor eax, dword ptr [r9+rbp*4+1024]
	movzx ebp, byte ptr [rdi+14]
	xor ebx, dword ptr [r9+rbp*4+2048]
	movzx ebp, byte ptr [rdi+15]
	xor ecx, dword ptr [r9+rbp*4+3072]

	mov dword ptr [rdi+0], eax
	mov dword ptr [rdi+4], ebx
	mov dword ptr [rdi+8], ecx
	mov dword ptr [rdi+12], edx

	ret

soft_aes_end:

	mov rax, [rsp+168]
	mov rbx, [rsp+176]
	mov rcx, [rsp+16]
	mov rdx, [rsp+184]
	mov rsi, [rsp+192]
	mov rdi, [rsp+200]
	mov rbp, [rsp+208]
	mov r8, [rsp+216]
	mov r9, [rsp+224]

	movapd xmm0, xmmword ptr [rsp+40]
	movapd xmm1, xmmword ptr [rsp+56]
	movapd xmm2, xmmword ptr [rsp+72]
	movapd xmm3, xmmword ptr [rsp+88]

	movapd xmmword ptr [rcx+0], xmm0
	movapd xmmword ptr [rcx+16], xmm1
	movapd xmmword ptr [rcx+32], xmm2
	movapd xmmword ptr [rcx+48], xmm3
