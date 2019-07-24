dnl ---------------------------------------------------------------------
dnl Copyright (C) 2015, Ondrej Mosnacek <omosnacek@gmail.com>
dnl
dnl This program is free software: you can redistribute it and/or
dnl modify it under the terms of the GNU General Public License
dnl as published by the Free Software Foundation: either version 2
dnl of the License, or (at your option) any later version.
dnl
dnl This program is distributed in the hope that it will be useful,
dnl but WITHOUT ANY WARRANTY; without even the implied warranty of
dnl MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
dnl GNU General Public License for more details.
dnl
dnl You should have received a copy of the GNU General Public License
dnl along with this program.  If not, see <http://www.gnu.org/licenses/>.
dnl ---------------------------------------------------------------------

AC_CONFIG_MACRO_DIR([m4])

AC_INIT([argon2], [0.1], [])
LT_INIT
AM_INIT_AUTOMAKE([foreign subdir-objects])
AM_SILENT_RULES([yes])

AC_PROG_CC
AC_PROG_CC_C89
AM_PROG_AS
AX_PTHREAD

AC_CANONICAL_HOST

AS_CASE([$host_cpu],
    dnl [i?86], [ARCH=i386],
    [x86_64], [ARCH=x86_64],
    [ARCH=generic
    AC_MSG_WARN("No code for architecture $host_cpu; using generic implementation")]
)
AC_SUBST([ARCH])

AM_CONDITIONAL([ARCH_X86_64],  [test "$ARCH" = 'x86_64'])
AM_CONDITIONAL([ARCH_GENERIC], [test "$ARCH" = 'generic'])

# AX_CHECK_COMPILER_FEATURE(NAME, FLAG, TEST_SOURCE)
# --------------------------
AC_DEFUN([AX_CHECK_COMPILER_FEATURE], [{
    AX_CHECK_COMPILE_FLAG([-m$2], [HAVE_FLAG=1], [HAVE_FLAG=0])
    HAVE_FEATURE=0
    AS_IF([test "$HAVE_FLAG" = '1'], [{
        AC_MSG_CHECKING("whether C compiler supports $1 with -m$2...")

        CFLAGS_BACKUP="$CFLAGS"
        CFLAGS="-m$2"

        AC_COMPILE_IFELSE([AC_LANG_SOURCE([$3])], [HAVE_FEATURE=1])

        CFLAGS="$CFLAGS_BACKUP"
        AS_IF([test "$HAVE_FEATURE" = '1'], [RESULT='yes'], [RESULT='no'])
        AC_MSG_RESULT([$RESULT])

    }])
    HAVE_$1=HAVE_FEATURE
    AM_CONDITIONAL([HAVE_$1], [test "$HAVE_FEATURE" = '1'])
}])

AX_CHECK_COMPILER_FEATURE([SSE2], [sse2], [[
#include <x86intrin.h>

void function_sse2(__m128i *dst, const __m128i *a, const __m128i *b)
{
    *dst = _mm_xor_si128(*a, *b);
}
]])
AX_CHECK_COMPILER_FEATURE([SSSE3], [ssse3], [[
#include <x86intrin.h>

void function_ssse3(__m128i *dst, const __m128i *a, const __m128i *b)
{
    *dst = _mm_shuffle_epi8(*a, *b);
}
]])
AX_CHECK_COMPILER_FEATURE([XOP], [xop], [[
#include <x86intrin.h>

void function_xop(__m128i *dst, const __m128i *a, int b)
{
    *dst = _mm_roti_epi64(*a, b);
}
]])
AX_CHECK_COMPILER_FEATURE([AVX2], [avx2], [[
#include <x86intrin.h>

void function_avx2(__m256i *dst, const __m256i *a, const __m256i *b)
{
    *dst = _mm256_xor_si256(*a, *b);
}
]])
AX_CHECK_COMPILER_FEATURE([AVX512F], [avx512f], [[
#include <x86intrin.h>

void function_avx512f(__m512i *dst, const __m512i *a)
{
    *dst = _mm512_ror_epi64(*a, 57);
}
]])

AC_CONFIG_FILES([Makefile])

AC_OUTPUT
