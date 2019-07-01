	;# restore callee-saved registers - System V AMD64 ABI
	pop r15
	pop r14
	pop r13
	pop r12
	pop rbp
	pop rbx

	;# program finished
	ret 0