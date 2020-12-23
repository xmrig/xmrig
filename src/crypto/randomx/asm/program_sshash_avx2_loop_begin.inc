	;# prefetch RandomX dataset lines
	prefetchnta byte ptr [rsi]
	prefetchnta byte ptr [rsi+64]
	prefetchnta byte ptr [rsi+128]
	prefetchnta byte ptr [rsi+192]
	prefetchnta byte ptr [rsi+256]

	;# prefetch RandomX cache lines
	mov rbx, rbp
	and rbx, RANDOMX_CACHE_MASK
	shl rbx, 6
	add rbx, rdi
	prefetchnta byte ptr [rbx]
	lea rax, [rbp+1]
	and rax, RANDOMX_CACHE_MASK
	shl rax, 6
	add rax, rdi
	prefetchnta byte ptr [rax]
	mov [rsp], rax
	lea rax, [rbp+2]
	and rax, RANDOMX_CACHE_MASK
	shl rax, 6
	add rax, rdi
	prefetchnta byte ptr [rax]
	mov [rsp+8], rax
	lea rax, [rbp+3]
	and rax, RANDOMX_CACHE_MASK
	shl rax, 6
	add rax, rdi
	prefetchnta byte ptr [rax]
	mov [rsp+16], rax
	lea rax, [rbp+4]
	and rax, RANDOMX_CACHE_MASK
	shl rax, 6
	add rax, rdi
	prefetchnta byte ptr [rax]
	mov [rsp+24], rax
