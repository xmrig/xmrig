_TEXT_CNV2_MAINLOOP SEGMENT PAGE READ EXECUTE
PUBLIC cnv2_mainloop_ivybridge_asm
PUBLIC cnv2_mainloop_ryzen_asm
PUBLIC cnv2_double_mainloop_sandybridge_asm

ALIGN 64
cnv2_mainloop_ivybridge_asm PROC
	INCLUDE cnv2_main_loop_ivybridge.inc
	ret 0
cnv2_mainloop_ivybridge_asm ENDP

ALIGN 64
cnv2_mainloop_ryzen_asm PROC
	INCLUDE cnv2_main_loop_ryzen.inc
	ret 0
cnv2_mainloop_ryzen_asm ENDP

ALIGN 64
cnv2_double_mainloop_sandybridge_asm PROC
	INCLUDE cnv2_double_main_loop_sandybridge.inc
	ret 0
cnv2_double_mainloop_sandybridge_asm ENDP

_TEXT_CNV2_MAINLOOP ENDS
END
