_TEXT_CN_MAINLOOP SEGMENT PAGE READ EXECUTE
PUBLIC cnv1_mainloop_sandybridge_asm
PUBLIC cnv2_mainloop_ivybridge_asm
PUBLIC cnv2_mainloop_ryzen_asm
PUBLIC cnv2_double_mainloop_sandybridge_asm

PUBLIC cnv1_mainloop_soft_aes_sandybridge_asm
PUBLIC cnv2_mainloop_soft_aes_sandybridge_asm

#ifdef __APPLE__
ALIGN 16
#else
ALIGN 64
#endif
cnv1_mainloop_sandybridge_asm PROC
	INCLUDE cnv1_mainloop_sandybridge.inc
	ret 0
cnv1_mainloop_sandybridge_asm ENDP

#ifdef __APPLE__
ALIGN 16
#else
ALIGN 64
#endif
cnv2_mainloop_ivybridge_asm PROC
	INCLUDE cnv2_main_loop_ivybridge.inc
	ret 0
cnv2_mainloop_ivybridge_asm ENDP

#ifdef __APPLE__
ALIGN 16
#else
ALIGN 64
#endif
cnv2_mainloop_ryzen_asm PROC
	INCLUDE cnv2_main_loop_ryzen.inc
	ret 0
cnv2_mainloop_ryzen_asm ENDP

#ifdef __APPLE__
ALIGN 16
#else
ALIGN 64
#endif
cnv2_double_mainloop_sandybridge_asm PROC
	INCLUDE cnv2_double_main_loop_sandybridge.inc
	ret 0
cnv2_double_mainloop_sandybridge_asm ENDP

#ifdef __APPLE__
ALIGN 16
#else
ALIGN 64
#endif
cnv1_mainloop_soft_aes_sandybridge_asm PROC
	INCLUDE cnv1_mainloop_soft_aes_sandybridge.inc
	ret 0
cnv1_mainloop_soft_aes_sandybridge_asm ENDP

#ifdef __APPLE__
ALIGN 16
#else
ALIGN 64
#endif
cnv2_mainloop_soft_aes_sandybridge_asm PROC
	INCLUDE cnv2_mainloop_soft_aes_sandybridge.inc
	ret 0
cnv2_mainloop_soft_aes_sandybridge_asm ENDP

_TEXT_CN_MAINLOOP ENDS
END
