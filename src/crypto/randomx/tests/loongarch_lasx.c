/*
 * Probe built once by CMake to find out whether this LoongArch CPU
 * really executes the 256 bit LASX vector instructions.
 *
 * Not every LoongArch CPU carries LASX, therefore the 256 bit BLAKE2b code of
 * RandomX must not be linked or run when this probe fails, see
 * XMRIG_FEATURE_LASX in cmake/cpu.cmake.
 */

#include <lasxintrin.h>


int main(void)
{
    long long buf[4] = { 1, 2, 3, 4 };

    __m256i a = __lasx_xvld(buf, 0);
    __m256i b = __lasx_xvadd_d(a, a);

    __lasx_xvst(b, buf, 0);

    return buf[0] == 2 ? 0 : 1;
}
