/*
 * Copyright (c) 2026, RESBI <https://github.com/RESBI>

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
	* Redistributions of source code must retain the above copyright
	  notice, this list of conditions and the following disclaimer.
	* Redistributions in binary form must reproduce the above copyright
	  notice, this list of conditions and the following disclaimer in the
	  documentation and/or other materials provided with the distribution.
	* Neither the name of the copyright holder nor the
	  names of its contributors may be used to endorse or promote products
	  derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

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
