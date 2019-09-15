#include <x86intrin.h>

void function_xop(__m128i *dst, const __m128i *a, int b)
{
    *dst = _mm_roti_epi64(*a, b);
}

int main(void) { return 0; }
