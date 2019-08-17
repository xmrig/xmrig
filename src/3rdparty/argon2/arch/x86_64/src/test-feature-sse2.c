#include <x86intrin.h>

void function_sse2(__m128i *dst, const __m128i *a, const __m128i *b)
{
    *dst = _mm_xor_si128(*a, *b);
}

int main(void) { return 0; }
