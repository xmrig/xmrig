	vmovdqu ymmword ptr [rsp], ymm0

	mov rax, [rsp]
	and rax, RANDOMX_CACHE_MASK
	shl rax, 6
	add rax, rdi
	mov [rsp], rax
	prefetchnta byte ptr [rax]

	mov rax, [rsp+8]
	and rax, RANDOMX_CACHE_MASK
	shl rax, 6
	add rax, rdi
	mov [rsp+8], rax
	prefetchnta byte ptr [rax]

	mov rax, [rsp+16]
	and rax, RANDOMX_CACHE_MASK
	shl rax, 6
	add rax, rdi
	mov [rsp+16], rax
	prefetchnta byte ptr [rax]

	mov rax, [rsp+24]
	and rax, RANDOMX_CACHE_MASK
	shl rax, 6
	add rax, rdi
	mov [rsp+24], rax
	prefetchnta byte ptr [rax]
