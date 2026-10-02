/*
 * Probe built once by CMake to find out whether this LoongArch CPU
 * really executes the 128 bit LSX vector instructions.
 *
 * LSX is an extension of the LoongArch base ISA, it is not implied by it, so
 * the only safe way is to compile one instruction with -mlsx and run it.
 * A CPU without LSX kills the process with SIGILL and CMake reports the run as
 * failed, see XMRIG_FEATURE_LSX in cmake/cpu.cmake.
 */

#include <lsxintrin.h>


int main(void)
{
    long long buf[2] = { 1, 2 };

    __m128i a = __lsx_vld(buf, 0);
    __m128i b = __lsx_vadd_d(a, a);

    __lsx_vst(b, buf, 0);

    return buf[0] == 2 ? 0 : 1;
}
