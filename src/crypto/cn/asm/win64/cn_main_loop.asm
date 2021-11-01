_TEXT_CNV2_MAINLOOP SEGMENT PAGE READ EXECUTE
PUBLIC cnv2_mainloop_ivybridge_asm
PUBLIC cnv2_mainloop_ryzen_asm
PUBLIC cnv2_mainloop_bulldozer_asm
PUBLIC cnv2_double_mainloop_sandybridge_asm
PUBLIC cnv2_rwz_mainloop_asm
PUBLIC cnv2_rwz_double_mainloop_asm

ALIGN 64
cnv2_mainloop_ivybridge_asm PROC
	INCLUDE cn2/cnv2_main_loop_ivybridge.inc
	ret 0
	mov eax, 3735929054
cnv2_mainloop_ivybridge_asm ENDP

ALIGN 64
cnv2_mainloop_ryzen_asm PROC
	INCLUDE cn2/cnv2_main_loop_ryzen.inc
	ret 0
	mov eax, 3735929054
cnv2_mainloop_ryzen_asm ENDP

ALIGN 64
cnv2_mainloop_bulldozer_asm PROC
	INCLUDE cn2/cnv2_main_loop_bulldozer.inc
	ret 0
	mov eax, 3735929054
cnv2_mainloop_bulldozer_asm ENDP

ALIGN 64
cnv2_double_mainloop_sandybridge_asm PROC
	INCLUDE cn2/cnv2_double_main_loop_sandybridge.inc
	ret 0
	mov eax, 3735929054
cnv2_double_mainloop_sandybridge_asm ENDP

ALIGN(64)
cnv2_rwz_mainloop_asm PROC
	INCLUDE cn2/cnv2_rwz_main_loop.inc
	ret 0
	mov eax, 3735929054
cnv2_rwz_mainloop_asm ENDP

ALIGN(64)
cnv2_rwz_double_mainloop_asm PROC
	INCLUDE cn2/cnv2_rwz_double_main_loop.inc
	ret 0
	mov eax, 3735929054
cnv2_rwz_double_mainloop_asm ENDP

ALIGN(64)
cnv2_upx_double_mainloop_zen3_asm PROC
	INCLUDE cn2/cnv2_upx_double_mainloop_zen3.inc
	ret 0
	mov eax, 3735929054
cnv2_upx_double_mainloop_zen3_asm ENDP

_TEXT_CNV2_MAINLOOP ENDS
END
