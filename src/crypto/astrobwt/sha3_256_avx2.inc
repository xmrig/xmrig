;# XMRig
;# Copyright 2010      Jeff Garzik              <jgarzik@pobox.com>
;# Copyright 2012-2014 pooler                   <pooler@litecoinpool.org>
;# Copyright 2014      Lucas Jones              <https://github.com/lucasjones>
;# Copyright 2014-2016 Wolf9466                 <https://github.com/OhGodAPet>
;# Copyright 2016      Jay D Dee                <jayddee246@gmail.com>
;# Copyright 2017-2019 XMR-Stak                 <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
;# Copyright 2018      Lee Clagett              <https://github.com/vtnerd>
;# Copyright 2018-2019 tevador                  <tevador@gmail.com>
;# Copyright 2000      Transmeta Corporation    <https://github.com/intel/msr-tools>
;# Copyright 2004-2008 H. Peter Anvin           <https://github.com/intel/msr-tools>
;# Copyright 2018-2020 SChernykh                <https://github.com/SChernykh>
;# Copyright 2016-2020 XMRig                    <https://github.com/xmrig>, <support@xmrig.com>
;#
;#   This program is free software: you can redistribute it and/or modify
;#   it under the terms of the GNU General Public License as published by
;#   the Free Software Foundation, either version 3 of the License, or
;#   (at your option) any later version.
;#
;#   This program is distributed in the hope that it will be useful,
;#   but WITHOUT ANY WARRANTY; without even the implied warranty of
;#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
;#   GNU General Public License for more details.
;#
;#   You should have received a copy of the GNU General Public License
;#   along with this program. If not, see <http://www.gnu.org/licenses/>.
;#

	vzeroupper

	mov qword ptr [rsp+8],rbx
	mov qword ptr [rsp+16],rsi
	mov qword ptr [rsp+24],rdi
	push rbp
	push r12
	push r13
	push r14
	push r15

	sub rsp, 80
	movdqu xmmword ptr [rsp+64], xmm6
	movdqu xmmword ptr [rsp+48], xmm7
	movdqu xmmword ptr [rsp+32], xmm8
	movdqu xmmword ptr [rsp+16], xmm9
	movdqu xmmword ptr [rsp+0], xmm10
	sub rsp, 80
	movdqu xmmword ptr [rsp+64], xmm11
	movdqu xmmword ptr [rsp+48], xmm12
	movdqu xmmword ptr [rsp+32], xmm13
	movdqu xmmword ptr [rsp+16], xmm14
	movdqu xmmword ptr [rsp+0], xmm15

	sub rsp,320
	lea rbp,[rsp+64]
	and rbp,-32
	vpxor xmm0,xmm0,xmm0
	xor edi,edi
	mov dword ptr [rbp],50462976
	mov r12,rdx
	mov dword ptr [rbp+4],169150212
	mov r14,rdx
	mov dword ptr [rbp+8],218436623
	shr r14,3
	and r12d,7
	mov dword ptr [rbp+12],135009046
	mov r13,r8
	mov byte ptr [rbp+16],9
	mov rsi,rcx
	mov ebx,edi
	vmovdqa ymmword ptr [rbp+32],ymm0
	vmovdqa ymmword ptr [rbp+64],ymm0
	vmovdqa ymmword ptr [rbp+96],ymm0
	vmovdqa ymmword ptr [rbp+128],ymm0
	vmovdqa ymmword ptr [rbp+160],ymm0
	vmovdqa ymmword ptr [rbp+192],ymm0
	vmovdqa ymmword ptr [rbp+224],ymm0
	test r14,r14
	je sha3_main_loop_end

sha3_main_loop:
	movzx eax,byte ptr [rbp+rbx]
	lea rcx,[rbp+32]
	lea rcx,[rcx+rax*8]
	mov rax,qword ptr [rsi]
	xor qword ptr [rcx],rax
	lea r15,[rbx+1]
	cmp rbx,16
	jne skip_keccak

	lea rcx,[rbp+32]
	call KeccakF1600_AVX2_ASM

skip_keccak:
	cmp rbx,16
	mov rax,rdi
	cmovne rax,r15
	add rsi,8
	mov rbx,rax
	sub r14,1
	jne sha3_main_loop

sha3_main_loop_end:
	mov rdx,rdi
	test r12,r12
	je sha3_tail_loop_end
	mov r8,rdi

sha3_tail_loop:
	movzx eax,byte ptr [rdx+rsi]
	inc rdx
	shlx rcx,rax,r8
	or rdi,rcx
	add r8,8
	cmp rdx,r12
	jb sha3_tail_loop

sha3_tail_loop_end:
	movzx eax,byte ptr [rbp+rbx]
	lea rdx,[rbp+32]
	lea rdx,[rdx+rax*8]
	mov ecx,6
	lea rax,[r12*8]
	shlx rcx,rcx,rax
	xor rcx,qword ptr [rdx]
	mov eax,1
	shl rax,63
	xor rcx,rdi
	mov qword ptr [rdx],rcx
	xor qword ptr [rbp+104],rax

	lea rcx,[rbp+32]
	call KeccakF1600_AVX2_ASM

	vmovups ymm0,ymmword ptr [rbp+32]
	vmovups ymmword ptr [r13],ymm0
	vzeroupper

	add rsp,320

	movdqu xmm15, xmmword ptr [rsp]
	movdqu xmm14, xmmword ptr [rsp+16]
	movdqu xmm13, xmmword ptr [rsp+32]
	movdqu xmm12, xmmword ptr [rsp+48]
	movdqu xmm11, xmmword ptr [rsp+64]
	add rsp, 80
	movdqu xmm10, xmmword ptr [rsp]
	movdqu xmm9, xmmword ptr [rsp+16]
	movdqu xmm8, xmmword ptr [rsp+32]
	movdqu xmm7, xmmword ptr [rsp+48]
	movdqu xmm6, xmmword ptr [rsp+64]
	add rsp, 80

	pop r15
	pop r14
	pop r13
	pop r12
	pop rbp
	mov rbx,qword ptr [rsp+8]
	mov rsi,qword ptr [rsp+16]
	mov rdi,qword ptr [rsp+24]

	ret
