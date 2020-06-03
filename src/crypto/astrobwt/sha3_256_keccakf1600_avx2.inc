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

	mov eax,24
	lea rcx,[rcx+96]
	vpbroadcastq ymm0,QWORD PTR [rcx-96]
	vmovdqu ymm1,YMMWORD PTR [rcx-88]
	vmovdqu ymm2,YMMWORD PTR [rcx-56]
	vmovdqu ymm3,YMMWORD PTR [rcx-24]
	vmovdqu ymm4,YMMWORD PTR [rcx+8]
	vmovdqu ymm5,YMMWORD PTR [rcx+40]
	vmovdqu ymm6,YMMWORD PTR [rcx+72]

ALIGN 64
Loop_avx2:
	vpshufd ymm13,ymm2,78
	vpxor ymm12,ymm5,ymm3
	vpxor ymm9,ymm4,ymm6
	vpxor ymm12,ymm12,ymm1
	vpxor ymm12,ymm12,ymm9
	vpermq ymm11,ymm12,147
	vpxor ymm13,ymm13,ymm2
	vpermq ymm7,ymm13,78
	vpsrlq ymm8,ymm12,63
	vpaddq ymm9,ymm12,ymm12
	vpor ymm8,ymm8,ymm9
	vpermq ymm15,ymm8,57
	vpxor ymm14,ymm8,ymm11
	vpermq ymm14,ymm14,0
	vpxor ymm13,ymm13,ymm0
	vpxor ymm13,ymm13,ymm7
	vpsrlq ymm7,ymm13,63
	vpaddq ymm8,ymm13,ymm13
	vpor ymm8,ymm8,ymm7
	vpxor ymm2,ymm2,ymm14
	vpxor ymm0,ymm0,ymm14
	vpblendd ymm15,ymm15,ymm8,192
	vpblendd ymm11,ymm11,ymm13,3
	vpxor ymm15,ymm15,ymm11
	vpsllvq ymm10,ymm2,YMMWORD PTR [r8-96]
	vpsrlvq ymm2,ymm2,YMMWORD PTR [r9-96]
	vpor ymm2,ymm2,ymm10
	vpxor ymm3,ymm3,ymm15
	vpsllvq ymm11,ymm3,YMMWORD PTR [r8-32]
	vpsrlvq ymm3,ymm3,YMMWORD PTR [r9-32]
	vpor ymm3,ymm3,ymm11
	vpxor ymm4,ymm4,ymm15
	vpsllvq ymm12,ymm4,YMMWORD PTR [r8]
	vpsrlvq ymm4,ymm4,YMMWORD PTR [r9]
	vpor ymm4,ymm4,ymm12
	vpxor ymm5,ymm5,ymm15
	vpsllvq ymm13,ymm5,YMMWORD PTR [r8+32]
	vpsrlvq ymm5,ymm5,YMMWORD PTR [r9+32]
	vpor ymm5,ymm5,ymm13
	vpxor ymm6,ymm6,ymm15
	vpermq ymm10,ymm2,141
	vpermq ymm11,ymm3,141
	vpsllvq ymm14,ymm6,YMMWORD PTR [r8+64]
	vpsrlvq ymm8,ymm6,YMMWORD PTR [r9+64]
	vpor ymm8,ymm8,ymm14
	vpxor ymm1,ymm1,ymm15
	vpermq ymm12,ymm4,27
	vpermq ymm13,ymm5,114
	vpsllvq ymm15,ymm1,YMMWORD PTR [r8-64]
	vpsrlvq ymm9,ymm1,YMMWORD PTR [r9-64]
	vpor ymm9,ymm9,ymm15
	vpsrldq ymm14,ymm8,8
	vpandn ymm7,ymm8,ymm14
	vpblendd ymm3,ymm9,ymm13,12
	vpblendd ymm15,ymm11,ymm9,12
	vpblendd ymm5,ymm10,ymm11,12
	vpblendd ymm14,ymm9,ymm10,12
	vpblendd ymm3,ymm3,ymm11,48
	vpblendd ymm15,ymm15,ymm12,48
	vpblendd ymm5,ymm5,ymm9,48
	vpblendd ymm14,ymm14,ymm13,48
	vpblendd ymm3,ymm3,ymm12,192
	vpblendd ymm15,ymm15,ymm13,192
	vpblendd ymm5,ymm5,ymm13,192
	vpblendd ymm14,ymm14,ymm11,192
	vpandn ymm3,ymm3,ymm15
	vpandn ymm5,ymm5,ymm14
	vpblendd ymm6,ymm12,ymm9,12
	vpblendd ymm15,ymm10,ymm12,12
	vpxor ymm3,ymm3,ymm10
	vpblendd ymm6,ymm6,ymm10,48
	vpblendd ymm15,ymm15,ymm11,48
	vpxor ymm5,ymm5,ymm12
	vpblendd ymm6,ymm6,ymm11,192
	vpblendd ymm15,ymm15,ymm9,192
	vpandn ymm6,ymm6,ymm15
	vpxor ymm6,ymm6,ymm13
	vpermq ymm4,ymm8,30
	vpblendd ymm15,ymm4,ymm0,48
	vpermq ymm1,ymm8,57
	vpblendd ymm1,ymm1,ymm0,192
	vpandn ymm1,ymm1,ymm15
	vpblendd ymm2,ymm11,ymm12,12
	vpblendd ymm14,ymm13,ymm11,12
	vpblendd ymm2,ymm2,ymm13,48
	vpblendd ymm14,ymm14,ymm10,48
	vpblendd ymm2,ymm2,ymm10,192
	vpblendd ymm14,ymm14,ymm12,192
	vpandn ymm2,ymm2,ymm14
	vpxor ymm2,ymm2,ymm9
	vpermq ymm7,ymm7,0
	vpermq ymm3,ymm3,27
	vpermq ymm5,ymm5,141
	vpermq ymm6,ymm6,114
	vpblendd ymm4,ymm13,ymm10,12
	vpblendd ymm14,ymm12,ymm13,12
	vpblendd ymm4,ymm4,ymm12,48
	vpblendd ymm14,ymm14,ymm9,48
	vpblendd ymm4,ymm4,ymm9,192
	vpblendd ymm14,ymm14,ymm10,192
	vpandn ymm4,ymm4,ymm14
	vpxor ymm0,ymm0,ymm7
	vpxor ymm1,ymm1,ymm8
	vpxor ymm4,ymm4,ymm11
	vpxor ymm0,ymm0,YMMWORD PTR [r10]
	lea r10,[r10+32]
	dec eax
	jnz Loop_avx2

	vmovq QWORD PTR [rcx-96],xmm0
	vmovdqu YMMWORD PTR [rcx-88],ymm1
	vmovdqu YMMWORD PTR [rcx-56],ymm2
	vmovdqu YMMWORD PTR [rcx-24],ymm3
	vmovdqu YMMWORD PTR [rcx+8],ymm4
	vmovdqu YMMWORD PTR [rcx+40],ymm5
	vmovdqu YMMWORD PTR [rcx+72],ymm6

	ret

ALIGN 32
rot_left:
	dq 3, 18, 36, 41
	dq 1, 62, 28, 27
	dq 45, 6, 56, 39
	dq 10, 61, 55, 8
	dq 2, 15, 25, 20
	dq 44, 43, 21, 14

ALIGN 32
rot_right:
	dq 64-3, 64-18, 64-36, 64-41
	dq 64-1, 64-62, 64-28, 64-27
	dq 64-45, 64-6, 64-56, 64-39
	dq 64-10, 64-61, 64-55, 64-8
	dq 64-2, 64-15, 64-25, 64-20
	dq 64-44, 64-43, 64-21, 64-14

ALIGN 32
rndc:
        dq 1, 1, 1, 1
        dq 32898, 32898, 32898, 32898
        dq 9223372036854808714, 9223372036854808714, 9223372036854808714, 9223372036854808714
        dq 9223372039002292224, 9223372039002292224, 9223372039002292224, 9223372039002292224
        dq 32907, 32907, 32907, 32907
        dq 2147483649, 2147483649, 2147483649, 2147483649
        dq 9223372039002292353, 9223372039002292353, 9223372039002292353, 9223372039002292353
        dq 9223372036854808585, 9223372036854808585, 9223372036854808585, 9223372036854808585
        dq 138, 138, 138, 138
        dq 136, 136, 136, 136
        dq 2147516425, 2147516425, 2147516425, 2147516425
        dq 2147483658, 2147483658, 2147483658, 2147483658
        dq 2147516555, 2147516555, 2147516555, 2147516555
        dq 9223372036854775947, 9223372036854775947, 9223372036854775947, 9223372036854775947
        dq 9223372036854808713, 9223372036854808713, 9223372036854808713, 9223372036854808713
        dq 9223372036854808579, 9223372036854808579, 9223372036854808579, 9223372036854808579
        dq 9223372036854808578, 9223372036854808578, 9223372036854808578, 9223372036854808578
        dq 9223372036854775936, 9223372036854775936, 9223372036854775936, 9223372036854775936
        dq 32778, 32778, 32778, 32778
        dq 9223372039002259466, 9223372039002259466, 9223372039002259466, 9223372039002259466
        dq 9223372039002292353, 9223372039002292353, 9223372039002292353, 9223372039002292353
        dq 9223372036854808704, 9223372036854808704, 9223372036854808704, 9223372036854808704
        dq 2147483649, 2147483649, 2147483649, 2147483649
        dq 9223372039002292232, 9223372039002292232, 9223372039002292232, 9223372039002292232
