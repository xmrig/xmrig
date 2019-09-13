#include <x86intrin.h>

void function_avx2(__m256i *dst, const __m256i *a, const __m256i *b)
{
    *dst = _mm256_xor_si256(*a, *b);
}

int main(void) { return 0; }
