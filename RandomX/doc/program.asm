randomx_isn_0:
	; ISMULH_R r0, r7
	mov rax, r8
	imul r15
	mov r8, rdx
randomx_isn_1:
	; IADD_RS r1, r2, SHFT 2
	lea r9, [r9+r10*4]
randomx_isn_2:
	; ISTORE L1[r6+1506176493], r4
	lea eax, [r14d+1506176493]
	and eax, 16376
	mov qword ptr [rsi+rax], r12
randomx_isn_3:
	; IMUL_R r5, r3
	imul r13, r11
randomx_isn_4:
	; IROR_R r3, r5
	mov ecx, r13d
	ror r11, cl
randomx_isn_5:
	; CBRANCH r7, -1891017657, COND 15
	add r15, -1886823353
	test r15, 2139095040
	jz randomx_isn_0
randomx_isn_6:
	; ISUB_M r3, L1[r7-1023302103]
	lea eax, [r15d-1023302103]
	and eax, 16376
	sub r11, qword ptr [rsi+rax]
randomx_isn_7:
	; IMUL_R r6, 220479013
	imul r14, 220479013
randomx_isn_8:
	; IADD_RS r5, r3, -669841607, SHFT 2
	lea r13, [r13+r11*4-669841607]
randomx_isn_9:
	; IADD_M r3, L3[532344]
	add r11, qword ptr [rsi+532344]
randomx_isn_10:
	; FADD_R f0, a3
	addpd xmm0, xmm11
randomx_isn_11:
	; CBRANCH r3, -1981570318, COND 4
	add r11, -1981566222
	test r11, 1044480
	jz randomx_isn_10
randomx_isn_12:
	; FSUB_R f0, a1
	subpd xmm0, xmm9
randomx_isn_13:
	; IADD_RS r1, r6, SHFT 2
	lea r9, [r9+r14*4]
randomx_isn_14:
	; FSQRT_R e2
	sqrtpd xmm6, xmm6
randomx_isn_15:
	; CBRANCH r5, -1278791788, COND 14
	add r13, -1278791788
	test r13, 1069547520
	jz randomx_isn_12
randomx_isn_16:
	; ISUB_R r3, -1310797453
	sub r11, -1310797453
randomx_isn_17:
	; IMUL_RCP r3, 2339914445
	mov rax, 16929713537937567113
	imul r11, rax
randomx_isn_18:
	; FADD_R f1, a2
	addpd xmm1, xmm10
randomx_isn_19:
	; FSUB_R f2, a2
	subpd xmm2, xmm10
randomx_isn_20:
	; IMUL_R r7, r0
	imul r15, r8
randomx_isn_21:
	; FADD_M f2, L2[r7-828505656]
	lea eax, [r15d-828505656]
	and eax, 262136
	cvtdq2pd xmm12, qword ptr [rsi+rax]
	addpd xmm2, xmm12
randomx_isn_22:
	; FDIV_M e1, L1[r1-1542605227]
	lea eax, [r9d-1542605227]
	and eax, 16376
	cvtdq2pd xmm12, qword ptr [rsi+rax]
	andps xmm12, xmm13
	orps xmm12, xmm14
	divpd xmm5, xmm12
randomx_isn_23:
	; IMUL_RCP r0, 1878277380
	mov rax, 10545322453154434729
	imul r8, rax
randomx_isn_24:
	; ISUB_R r6, r3
	sub r14, r11
randomx_isn_25:
	; IMUL_M r1, L1[r3-616171540]
	lea eax, [r11d-616171540]
	and eax, 16376
	imul r9, qword ptr [rsi+rax]
randomx_isn_26:
	; FSWAP_R f2
	shufpd xmm2, xmm2, 1
randomx_isn_27:
	; FSQRT_R e0
	sqrtpd xmm4, xmm4
randomx_isn_28:
	; IXOR_R r7, r5
	xor r15, r13
randomx_isn_29:
	; FADD_R f3, a3
	addpd xmm3, xmm11
randomx_isn_30:
	; FSUB_M f0, L2[r0+1880524670]
	lea eax, [r8d+1880524670]
	and eax, 262136
	cvtdq2pd xmm12, qword ptr [rsi+rax]
	subpd xmm0, xmm12
randomx_isn_31:
	; IADD_RS r0, r3, SHFT 3
	lea r8, [r8+r11*8]
randomx_isn_32:
	; FMUL_R e0, a2
	mulpd xmm4, xmm10
randomx_isn_33:
	; IMUL_M r1, L1[r4-588273594]
	lea eax, [r12d-588273594]
	and eax, 16376
	imul r9, qword ptr [rsi+rax]
randomx_isn_34:
	; IADD_M r4, L1[r6+999905907]
	lea eax, [r14d+999905907]
	and eax, 16376
	add r12, qword ptr [rsi+rax]
randomx_isn_35:
	; ISUB_R r4, r0
	sub r12, r8
randomx_isn_36:
	; FMUL_R e0, a3
	mulpd xmm4, xmm11
randomx_isn_37:
	; ISTORE L1[r4+2027210220], r3
	lea eax, [r12d+2027210220]
	and eax, 16376
	mov qword ptr [rsi+rax], r11
randomx_isn_38:
	; FADD_M f1, L2[r3+1451369534]
	lea eax, [r11d+1451369534]
	and eax, 262136
	cvtdq2pd xmm12, qword ptr [rsi+rax]
	addpd xmm1, xmm12
randomx_isn_39:
	; FMUL_R e1, a1
	mulpd xmm5, xmm9
randomx_isn_40:
	; FSUB_R f3, a2
	subpd xmm3, xmm10
randomx_isn_41:
	; IMULH_R r3, r3
	mov rax, r11
	mul r11
	mov r11, rdx
randomx_isn_42:
	; ISUB_R r4, r3
	sub r12, r11
randomx_isn_43:
	; CBRANCH r6, 335851892, COND 5
	add r14, 335847796
	test r14, 2088960
	jz randomx_isn_25
randomx_isn_44:
	; IADD_RS r7, r5, SHFT 3
	lea r15, [r15+r13*8]
randomx_isn_45:
	; CFROUND r6, 48
	mov rax, r14
	rol rax, 29
	and eax, 24576
	or eax, 40896
	push rax
	ldmxcsr dword ptr [rsp]
	pop rax
randomx_isn_46:
	; IMUL_RCP r6, 2070736307
	mov rax, 9565216276746377827
	imul r14, rax
randomx_isn_47:
	; IXOR_R r2, r4
	xor r10, r12
randomx_isn_48:
	; IMUL_R r0, r5
	imul r8, r13
randomx_isn_49:
	; CBRANCH r2, -272659465, COND 15
	add r10, -272659465
	test r10, 2139095040
	jz randomx_isn_48
randomx_isn_50:
	; ISTORE L1[r6+1414933948], r5
	lea eax, [r14d+1414933948]
	and eax, 16376
	mov qword ptr [rsi+rax], r13
randomx_isn_51:
	; ISTORE L1[r3-1336791747], r6
	lea eax, [r11d-1336791747]
	and eax, 16376
	mov qword ptr [rsi+rax], r14
randomx_isn_52:
	; FSCAL_R f1
	xorps xmm1, xmm15
randomx_isn_53:
	; CBRANCH r6, -2143810604, COND 1
	add r14, -2143810860
	test r14, 130560
	jz randomx_isn_50
randomx_isn_54:
	; ISUB_M r3, L1[r1-649360673]
	lea eax, [r9d-649360673]
	and eax, 16376
	sub r11, qword ptr [rsi+rax]
randomx_isn_55:
	; FADD_R f2, a3
	addpd xmm2, xmm11
randomx_isn_56:
	; CFROUND r3, 8
	mov rax, r11
	rol rax, 5
	and eax, 24576
	or eax, 40896
	push rax
	ldmxcsr dword ptr [rsp]
	pop rax
randomx_isn_57:
	; IROR_R r2, r0
	mov ecx, r8d
	ror r10, cl
randomx_isn_58:
	; IADD_RS r4, r2, SHFT 1
	lea r12, [r12+r10*2]
randomx_isn_59:
	; CBRANCH r6, -704407571, COND 10
	add r14, -704276499
	test r14, 66846720
	jz randomx_isn_54
randomx_isn_60:
	; FSUB_R f1, a3
	subpd xmm1, xmm11
randomx_isn_61:
	; ISUB_R r3, r7
	sub r11, r15
randomx_isn_62:
	; FMUL_R e2, a2
	mulpd xmm6, xmm10
randomx_isn_63:
	; FMUL_R e3, a1
	mulpd xmm7, xmm9
randomx_isn_64:
	; ISTORE L3[r2+845419810], r0
	lea eax, [r10d+845419810]
	and eax, 2097144
	mov qword ptr [rsi+rax], r8
randomx_isn_65:
	; CBRANCH r1, -67701844, COND 5
	add r9, -67705940
	test r9, 2088960
	jz randomx_isn_60
randomx_isn_66:
	; IROR_R r3, r1
	mov ecx, r9d
	ror r11, cl
randomx_isn_67:
	; IMUL_R r3, r1
	imul r11, r9
randomx_isn_68:
	; IROR_R r1, 40
	ror r9, 40
randomx_isn_69:
	; IMUL_R r3, r0
	imul r11, r8
randomx_isn_70:
	; IXOR_M r6, L3[1276704]
	xor r14, qword ptr [rsi+1276704]
randomx_isn_71:
	; FADD_M f0, L1[r1-1097746982]
	lea eax, [r9d-1097746982]
	and eax, 16376
	cvtdq2pd xmm12, qword ptr [rsi+rax]
	addpd xmm0, xmm12
randomx_isn_72:
	; IMUL_M r7, L1[r2+588700215]
	lea eax, [r10d+588700215]
	and eax, 16376
	imul r15, qword ptr [rsi+rax]
randomx_isn_73:
	; IXOR_M r2, L2[r3-1120252909]
	lea eax, [r11d-1120252909]
	and eax, 262136
	xor r10, qword ptr [rsi+rax]
randomx_isn_74:
	; FMUL_R e2, a0
	mulpd xmm6, xmm8
randomx_isn_75:
	; IMULH_R r2, r1
	mov rax, r10
	mul r9
	mov r10, rdx
randomx_isn_76:
	; FMUL_R e1, a2
	mulpd xmm5, xmm10
randomx_isn_77:
	; FSQRT_R e1
	sqrtpd xmm5, xmm5
randomx_isn_78:
	; FSCAL_R f1
	xorps xmm1, xmm15
randomx_isn_79:
	; FSWAP_R e1
	shufpd xmm5, xmm5, 1
randomx_isn_80:
	; IXOR_R r3, 721175561
	xor r11, 721175561
randomx_isn_81:
	; FSCAL_R f0
	xorps xmm0, xmm15
randomx_isn_82:
	; IADD_RS r3, r0, SHFT 1
	lea r11, [r11+r8*2]
randomx_isn_83:
	; ISUB_R r2, -691647438
	sub r10, -691647438
randomx_isn_84:
	; IXOR_R r1, r3
	xor r9, r11
randomx_isn_85:
	; IMULH_R r1, r7
	mov rax, r9
	mul r15
	mov r9, rdx
randomx_isn_86:
	; IMULH_R r3, r4
	mov rax, r11
	mul r12
	mov r11, rdx
randomx_isn_87:
	; CBRANCH r3, -1821955951, COND 5
	add r11, -1821955951
	test r11, 2088960
	jz randomx_isn_87
randomx_isn_88:
	; FADD_R f2, a3
	addpd xmm2, xmm11
randomx_isn_89:
	; IXOR_R r6, r3
	xor r14, r11
randomx_isn_90:
	; CBRANCH r4, -1780348372, COND 15
	add r12, -1784542676
	test r12, 2139095040
	jz randomx_isn_88
randomx_isn_91:
	; IROR_R r4, 55
	ror r12, 55
randomx_isn_92:
	; FSUB_R f3, a2
	subpd xmm3, xmm10
randomx_isn_93:
	; FSCAL_R f1
	xorps xmm1, xmm15
randomx_isn_94:
	; FADD_R f1, a0
	addpd xmm1, xmm8
randomx_isn_95:
	; ISUB_R r0, r3
	sub r8, r11
randomx_isn_96:
	; ISMULH_R r5, r7
	mov rax, r13
	imul r15
	mov r13, rdx
randomx_isn_97:
	; IADD_RS r0, r5, SHFT 1
	lea r8, [r8+r13*2]
randomx_isn_98:
	; IMUL_R r7, r3
	imul r15, r11
randomx_isn_99:
	; IADD_RS r2, r4, SHFT 2
	lea r10, [r10+r12*4]
randomx_isn_100:
	; ISTORE L3[r2+1641523310], r4
	lea eax, [r10d+1641523310]
	and eax, 2097144
	mov qword ptr [rsi+rax], r12
randomx_isn_101:
	; ISTORE L2[r5+1966751371], r5
	lea eax, [r13d+1966751371]
	and eax, 262136
	mov qword ptr [rsi+rax], r13
randomx_isn_102:
	; IXOR_R r4, r7
	xor r12, r15
randomx_isn_103:
	; CBRANCH r7, -607792642, COND 4
	add r15, -607792642
	test r15, 1044480
	jz randomx_isn_99
randomx_isn_104:
	; FMUL_R e1, a1
	mulpd xmm5, xmm9
randomx_isn_105:
	; IMUL_R r2, r3
	imul r10, r11
randomx_isn_106:
	; IADD_RS r5, r1, -1609896472, SHFT 3
	lea r13, [r13+r9*8-1609896472]
randomx_isn_107:
	; FMUL_R e2, a2
	mulpd xmm6, xmm10
randomx_isn_108:
	; ISUB_R r3, r6
	sub r11, r14
randomx_isn_109:
	; ISUB_R r0, r5
	sub r8, r13
randomx_isn_110:
	; IMUL_M r2, L3[1548384]
	imul r10, qword ptr [rsi+1548384]
randomx_isn_111:
	; FADD_R f2, a1
	addpd xmm2, xmm9
randomx_isn_112:
	; ISUB_M r6, L1[r7+1465746]
	lea eax, [r15d+1465746]
	and eax, 16376
	sub r14, qword ptr [rsi+rax]
randomx_isn_113:
	; IMULH_M r3, L1[r6-668730597]
	lea ecx, [r14d-668730597]
	and ecx, 16376
	mov rax, r11
	mul qword ptr [rsi+rcx]
	mov r11, rdx
randomx_isn_114:
	; IMUL_M r3, L2[r6-1549338697]
	lea eax, [r14d-1549338697]
	and eax, 262136
	imul r11, qword ptr [rsi+rax]
randomx_isn_115:
	; IMULH_M r4, L1[r6-82240335]
	lea ecx, [r14d-82240335]
	and ecx, 16376
	mov rax, r12
	mul qword ptr [rsi+rcx]
	mov r12, rdx
randomx_isn_116:
	; ISWAP_R r2, r4
	xchg r10, r12
randomx_isn_117:
	; IADD_RS r1, r0, SHFT 1
	lea r9, [r9+r8*2]
randomx_isn_118:
	; FSUB_R f0, a1
	subpd xmm0, xmm9
randomx_isn_119:
	; IADD_M r3, L1[r1-233433054]
	lea eax, [r9d-233433054]
	and eax, 16376
	add r11, qword ptr [rsi+rax]
randomx_isn_120:
	; FSUB_R f1, a0
	subpd xmm1, xmm8
randomx_isn_121:
	; ISUB_R r4, r3
	sub r12, r11
randomx_isn_122:
	; IXOR_M r6, L2[r1-425418413]
	lea eax, [r9d-425418413]
	and eax, 262136
	xor r14, qword ptr [rsi+rax]
randomx_isn_123:
	; FSQRT_R e2
	sqrtpd xmm6, xmm6
randomx_isn_124:
	; CBRANCH r1, -1807592127, COND 12
	add r9, -1806543551
	test r9, 267386880
	jz randomx_isn_118
randomx_isn_125:
	; IADD_RS r4, r4, SHFT 0
	lea r12, [r12+r12*1]
randomx_isn_126:
	; ISTORE L2[r5-104490218], r0
	lea eax, [r13d-104490218]
	and eax, 262136
	mov qword ptr [rsi+rax], r8
randomx_isn_127:
	; IXOR_R r5, r0
	xor r13, r8
randomx_isn_128:
	; IMUL_M r6, L1[r2-603755642]
	lea eax, [r10d-603755642]
	and eax, 16376
	imul r14, qword ptr [rsi+rax]
randomx_isn_129:
	; INEG_R r5
	neg r13
randomx_isn_130:
	; FMUL_R e0, a0
	mulpd xmm4, xmm8
randomx_isn_131:
	; ISUB_R r0, -525100988
	sub r8, -525100988
randomx_isn_132:
	; IMUL_RCP r0, 3636489804
	mov rax, 10893494383940851768
	imul r8, rax
randomx_isn_133:
	; FADD_M f2, L1[r3-768193829]
	lea eax, [r11d-768193829]
	and eax, 16376
	cvtdq2pd xmm12, qword ptr [rsi+rax]
	addpd xmm2, xmm12
randomx_isn_134:
	; IADD_RS r7, r7, SHFT 3
	lea r15, [r15+r15*8]
randomx_isn_135:
	; IROR_R r3, r2
	mov ecx, r10d
	ror r11, cl
randomx_isn_136:
	; ISUB_R r1, r4
	sub r9, r12
randomx_isn_137:
	; FADD_M f2, L1[r3+1221716517]
	lea eax, [r11d+1221716517]
	and eax, 16376
	cvtdq2pd xmm12, qword ptr [rsi+rax]
	addpd xmm2, xmm12
randomx_isn_138:
	; FDIV_M e2, L1[r3-1258284098]
	lea eax, [r11d-1258284098]
	and eax, 16376
	cvtdq2pd xmm12, qword ptr [rsi+rax]
	andps xmm12, xmm13
	orps xmm12, xmm14
	divpd xmm6, xmm12
randomx_isn_139:
	; FSUB_R f1, a0
	subpd xmm1, xmm8
randomx_isn_140:
	; IADD_RS r5, r6, -1773817530, SHFT 3
	lea r13, [r13+r14*8-1773817530]
randomx_isn_141:
	; IADD_M r0, L3[540376]
	add r8, qword ptr [rsi+540376]
randomx_isn_142:
	; FMUL_R e1, a3
	mulpd xmm5, xmm11
randomx_isn_143:
	; IADD_RS r6, r3, SHFT 2
	lea r14, [r14+r11*4]
randomx_isn_144:
	; ISTORE L1[r6+1837899146], r5
	lea eax, [r14d+1837899146]
	and eax, 16376
	mov qword ptr [rsi+rax], r13
randomx_isn_145:
	; FSWAP_R f2
	shufpd xmm2, xmm2, 1
randomx_isn_146:
	; FMUL_R e0, a0
	mulpd xmm4, xmm8
randomx_isn_147:
	; IADD_RS r1, r4, SHFT 3
	lea r9, [r9+r12*8]
randomx_isn_148:
	; ISUB_M r1, L2[r6-326072101]
	lea eax, [r14d-326072101]
	and eax, 262136
	sub r9, qword ptr [rsi+rax]
randomx_isn_149:
	; FSUB_R f1, a1
	subpd xmm1, xmm9
randomx_isn_150:
	; FADD_M f0, L2[r5+1123208251]
	lea eax, [r13d+1123208251]
	and eax, 262136
	cvtdq2pd xmm12, qword ptr [rsi+rax]
	addpd xmm0, xmm12
randomx_isn_151:
	; FSWAP_R f1
	shufpd xmm1, xmm1, 1
randomx_isn_152:
	; IMUL_M r3, L1[r4+522054565]
	lea eax, [r12d+522054565]
	and eax, 16376
	imul r11, qword ptr [rsi+rax]
randomx_isn_153:
	; IADD_RS r0, r0, SHFT 1
	lea r8, [r8+r8*2]
randomx_isn_154:
	; FMUL_R e2, a3
	mulpd xmm6, xmm11
randomx_isn_155:
	; FSUB_R f1, a2
	subpd xmm1, xmm10
randomx_isn_156:
	; ISTORE L1[r6+1559762664], r7
	lea eax, [r14d+1559762664]
	and eax, 16376
	mov qword ptr [rsi+rax], r15
randomx_isn_157:
	; FSUB_R f0, a1
	subpd xmm0, xmm9
randomx_isn_158:
	; ISUB_R r5, r6
	sub r13, r14
randomx_isn_159:
	; FADD_R f0, a0
	addpd xmm0, xmm8
randomx_isn_160:
	; FMUL_R e1, a0
	mulpd xmm5, xmm8
randomx_isn_161:
	; FSUB_R f2, a1
	subpd xmm2, xmm9
randomx_isn_162:
	; ISUB_R r5, r7
	sub r13, r15
randomx_isn_163:
	; FDIV_M e3, L2[r4-1912085642]
	lea eax, [r12d-1912085642]
	and eax, 262136
	cvtdq2pd xmm12, qword ptr [rsi+rax]
	andps xmm12, xmm13
	orps xmm12, xmm14
	divpd xmm7, xmm12
randomx_isn_164:
	; IXOR_M r3, L1[r0-858372123]
	lea eax, [r8d-858372123]
	and eax, 16376
	xor r11, qword ptr [rsi+rax]
randomx_isn_165:
	; IXOR_R r4, r6
	xor r12, r14
randomx_isn_166:
	; IADD_RS r3, r6, SHFT 0
	lea r11, [r11+r14*1]
randomx_isn_167:
	; FMUL_R e1, a1
	mulpd xmm5, xmm9
randomx_isn_168:
	; IADD_RS r5, r2, -371238437, SHFT 1
	lea r13, [r13+r10*2-371238437]
randomx_isn_169:
	; ISTORE L2[r5-633500019], r5
	lea eax, [r13d-633500019]
	and eax, 262136
	mov qword ptr [rsi+rax], r13
randomx_isn_170:
	; IXOR_R r4, -246154334
	xor r12, -246154334
randomx_isn_171:
	; IROR_R r7, r5
	mov ecx, r13d
	ror r15, cl
randomx_isn_172:
	; ISTORE L1[r5+4726218], r2
	lea eax, [r13d+4726218]
	and eax, 16376
	mov qword ptr [rsi+rax], r10
randomx_isn_173:
	; IADD_RS r2, r0, SHFT 3
	lea r10, [r10+r8*8]
randomx_isn_174:
	; IXOR_R r2, r6
	xor r10, r14
randomx_isn_175:
	; IADD_RS r0, r7, SHFT 0
	lea r8, [r8+r15*1]
randomx_isn_176:
	; FMUL_R e1, a1
	mulpd xmm5, xmm9
randomx_isn_177:
	; ISTORE L1[r1+962725405], r0
	lea eax, [r9d+962725405]
	and eax, 16376
	mov qword ptr [rsi+rax], r8
randomx_isn_178:
	; ISTORE L1[r5-1472969684], r4
	lea eax, [r13d-1472969684]
	and eax, 16376
	mov qword ptr [rsi+rax], r12
randomx_isn_179:
	; FSCAL_R f3
	xorps xmm3, xmm15
randomx_isn_180:
	; IXOR_M r7, L1[r5+1728657403]
	lea eax, [r13d+1728657403]
	and eax, 16376
	xor r15, qword ptr [rsi+rax]
randomx_isn_181:
	; CBRANCH r2, -759703940, COND 2
	add r10, -759704452
	test r10, 261120
	jz randomx_isn_175
randomx_isn_182:
	; FADD_R f1, a2
	addpd xmm1, xmm10
randomx_isn_183:
	; IMULH_R r5, r1
	mov rax, r13
	mul r9
	mov r13, rdx
randomx_isn_184:
	; FSUB_R f3, a2
	subpd xmm3, xmm10
randomx_isn_185:
	; IMUL_R r6, r2
	imul r14, r10
randomx_isn_186:
	; IROR_R r2, r6
	mov ecx, r14d
	ror r10, cl
randomx_isn_187:
	; FADD_R f2, a3
	addpd xmm2, xmm11
randomx_isn_188:
	; FSUB_R f3, a2
	subpd xmm3, xmm10
randomx_isn_189:
	; FSUB_R f0, a1
	subpd xmm0, xmm9
randomx_isn_190:
	; FSUB_R f1, a2
	subpd xmm1, xmm10
randomx_isn_191:
	; ISTORE L2[r0+519974891], r5
	lea eax, [r8d+519974891]
	and eax, 262136
	mov qword ptr [rsi+rax], r13
randomx_isn_192:
	; IXOR_R r3, r0
	xor r11, r8
randomx_isn_193:
	; IMUL_RCP r3, 2631645861
	mov rax, 15052968123180221777
	imul r11, rax
randomx_isn_194:
	; FSCAL_R f2
	xorps xmm2, xmm15
randomx_isn_195:
	; IMUL_RCP r6, 3565118466
	mov rax, 11111575010739676440
	imul r14, rax
randomx_isn_196:
	; IMUL_RCP r7, 2240276148
	mov rax, 17682677777245240213
	imul r15, rax
randomx_isn_197:
	; FADD_R f3, a0
	addpd xmm3, xmm8
randomx_isn_198:
	; ISTORE L3[r7-908286266], r0
	lea eax, [r15d-908286266]
	and eax, 2097144
	mov qword ptr [rsi+rax], r8
randomx_isn_199:
	; FMUL_R e0, a1
	mulpd xmm4, xmm9
randomx_isn_200:
	; FADD_R f1, a2
	addpd xmm1, xmm10
randomx_isn_201:
	; IADD_RS r3, r2, SHFT 3
	lea r11, [r11+r10*8]
randomx_isn_202:
	; FSUB_R f0, a0
	subpd xmm0, xmm8
randomx_isn_203:
	; CBRANCH r1, -1282235504, COND 2
	add r9, -1282234992
	test r9, 261120
	jz randomx_isn_182
randomx_isn_204:
	; IMUL_M r1, L3[176744]
	imul r9, qword ptr [rsi+176744]
randomx_isn_205:
	; FSWAP_R e1
	shufpd xmm5, xmm5, 1
randomx_isn_206:
	; CBRANCH r0, -1557284726, COND 14
	add r8, -1555187574
	test r8, 1069547520
	jz randomx_isn_204
randomx_isn_207:
	; IADD_M r3, L1[r0+72267507]
	lea eax, [r8d+72267507]
	and eax, 16376
	add r11, qword ptr [rsi+rax]
randomx_isn_208:
	; ISUB_R r7, r0
	sub r15, r8
randomx_isn_209:
	; IROR_R r3, r2
	mov ecx, r10d
	ror r11, cl
randomx_isn_210:
	; ISUB_R r0, r3
	sub r8, r11
randomx_isn_211:
	; IMUL_RCP r7, 3271526781
	mov rax, 12108744298594255889
	imul r15, rax
randomx_isn_212:
	; FSQRT_R e2
	sqrtpd xmm6, xmm6
randomx_isn_213:
	; IMUL_R r0, r4
	imul r8, r12
randomx_isn_214:
	; FSWAP_R f3
	shufpd xmm3, xmm3, 1
randomx_isn_215:
	; FADD_R f2, a1
	addpd xmm2, xmm9
randomx_isn_216:
	; ISMULH_M r5, L1[r4-1702277076]
	lea ecx, [r12d-1702277076]
	and ecx, 16376
	mov rax, r13
	imul qword ptr [rsi+rcx]
	mov r13, rdx
randomx_isn_217:
	; ISUB_R r4, r2
	sub r12, r10
randomx_isn_218:
	; FMUL_R e1, a2
	mulpd xmm5, xmm10
randomx_isn_219:
	; FSUB_R f3, a1
	subpd xmm3, xmm9
randomx_isn_220:
	; ISTORE L2[r1+1067932664], r3
	lea eax, [r9d+1067932664]
	and eax, 262136
	mov qword ptr [rsi+rax], r11
randomx_isn_221:
	; IROR_R r6, r4
	mov ecx, r12d
	ror r14, cl
randomx_isn_222:
	; FSUB_R f1, a1
	subpd xmm1, xmm9
randomx_isn_223:
	; ISUB_R r2, r5
	sub r10, r13
randomx_isn_224:
	; IXOR_R r2, r7
	xor r10, r15
randomx_isn_225:
	; IXOR_R r7, r5
	xor r15, r13
randomx_isn_226:
	; IMUL_RCP r4, 1021824288
	mov rax, 9691999329617659469
	imul r12, rax
randomx_isn_227:
	; IROR_R r1, 48
	ror r9, 48
randomx_isn_228:
	; IMUL_RCP r4, 4042529026
	mov rax, 9799331310263836012
	imul r12, rax
randomx_isn_229:
	; FSQRT_R e1
	sqrtpd xmm5, xmm5
randomx_isn_230:
	; IROR_R r3, r6
	mov ecx, r14d
	ror r11, cl
randomx_isn_231:
	; FMUL_R e2, a1
	mulpd xmm6, xmm9
randomx_isn_232:
	; IMULH_M r4, L1[r6+396272725]
	lea ecx, [r14d+396272725]
	and ecx, 16376
	mov rax, r12
	mul qword ptr [rsi+rcx]
	mov r12, rdx
randomx_isn_233:
	; FSUB_R f0, a0
	subpd xmm0, xmm8
randomx_isn_234:
	; FADD_R f3, a2
	addpd xmm3, xmm10
randomx_isn_235:
	; IADD_RS r7, r3, SHFT 1
	lea r15, [r15+r11*2]
randomx_isn_236:
	; ISUB_R r6, r3
	sub r14, r11
randomx_isn_237:
	; IADD_RS r4, r4, SHFT 2
	lea r12, [r12+r12*4]
randomx_isn_238:
	; ISUB_R r7, r1
	sub r15, r9
randomx_isn_239:
	; ISMULH_R r2, r5
	mov rax, r10
	imul r13
	mov r10, rdx
randomx_isn_240:
	; FMUL_R e1, a2
	mulpd xmm5, xmm10
randomx_isn_241:
	; IADD_RS r1, r4, SHFT 2
	lea r9, [r9+r12*4]
randomx_isn_242:
	; FDIV_M e2, L2[r6+259737107]
	lea eax, [r14d+259737107]
	and eax, 262136
	cvtdq2pd xmm12, qword ptr [rsi+rax]
	andps xmm12, xmm13
	orps xmm12, xmm14
	divpd xmm6, xmm12
randomx_isn_243:
	; IADD_M r0, L1[r1+789576070]
	lea eax, [r9d+789576070]
	and eax, 16376
	add r8, qword ptr [rsi+rax]
randomx_isn_244:
	; IMUL_R r3, r4
	imul r11, r12
randomx_isn_245:
	; IMUL_R r3, r1
	imul r11, r9
randomx_isn_246:
	; IMUL_RCP r4, 1001661150
	mov rax, 9887096364157721599
	imul r12, rax
randomx_isn_247:
	; CBRANCH r3, -722123512, COND 2
	add r11, -722123512
	test r11, 261120
	jz randomx_isn_246
randomx_isn_248:
	; ISMULH_R r7, r6
	mov rax, r15
	imul r14
	mov r15, rdx
randomx_isn_249:
	; IADD_M r5, L3[1870552]
	add r13, qword ptr [rsi+1870552]
randomx_isn_250:
	; ISUB_R r0, r1
	sub r8, r9
randomx_isn_251:
	; IMULH_R r0, r5
	mov rax, r8
	mul r13
	mov r8, rdx
randomx_isn_252:
	; FSUB_R f1, a1
	subpd xmm1, xmm9
randomx_isn_253:
	; ISTORE L2[r3-2010380786], r5
	lea eax, [r11d-2010380786]
	and eax, 262136
	mov qword ptr [rsi+rax], r13
randomx_isn_254:
	; FMUL_R e3, a2
	mulpd xmm7, xmm10
randomx_isn_255:
	; CBRANCH r7, -2007380935, COND 9
	add r15, -2007315399
	test r15, 33423360
	jz randomx_isn_249
