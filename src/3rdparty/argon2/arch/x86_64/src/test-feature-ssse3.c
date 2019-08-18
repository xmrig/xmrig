#include <x86intrin.h>

void function_ssse3(__m128i *dst, const __m128i *a, const __m128i *b)
{
    *dst = _mm_shuffle_epi8(*a, *b);
}

int main(void) { return 0; }
