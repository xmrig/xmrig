#include <x86intrin.h>

void function_avx512f(__m512i *dst, const __m512i *a)
{
    *dst = _mm512_ror_epi64(*a, 57);
}

int main(void) { return 0; }
