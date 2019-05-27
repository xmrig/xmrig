ACLOCAL_AMFLAGS = -I m4

include_HEADERS = include/argon2.h
lib_LTLIBRARIES = libargon2.la
noinst_LTLIBRARIES =

bin_PROGRAMS = argon2
noinst_PROGRAMS = argon2-genkat argon2-bench2 argon2-test

TESTS = argon2-test

AM_CPPFLAGS = \
	-I$(srcdir)/include \
	-I$(srcdir)/lib \
	-I$(srcdir)/arch/@ARCH@/include \
	-I$(srcdir)/arch/@ARCH@/lib

libargon2_la_CFLAGS = @PTHREAD_CFLAGS@
libargon2_la_LIBADD = @PTHREAD_LIBS@
libargon2_la_SOURCES = \
	lib/argon2.c \
	lib/core.c \
	lib/encoding.c \
	lib/genkat.c \
	lib/impl-select.c \
	lib/thread.c \
	lib/blake2/blake2.c \
	lib/argon2-template-64.h \
	lib/core.h \
	lib/encoding.h \
	lib/genkat.h \
	lib/impl-select.h \
	lib/thread.h \
	lib/blake2/blake2.h \
	lib/blake2/blake2-impl.h

if ARCH_X86_64
noinst_LTLIBRARIES  += \
	libargon2-sse2.la \
	libargon2-ssse3.la \
	libargon2-xop.la \
	libargon2-avx2.la \
	libargon2-avx512f.la
libargon2_la_LIBADD += \
	libargon2-sse2.la \
	libargon2-ssse3.la \
	libargon2-xop.la \
	libargon2-avx2.la \
	libargon2-avx512f.la

libargon2_sse2_la_SOURCES = \
	arch/@ARCH@/lib/argon2-sse2.c \
	arch/@ARCH@/lib/argon2-sse2.h \
	arch/@ARCH@/lib/argon2-template-128.h
libargon2_sse2_la_CPPFLAGS = $(AM_CPPFLAGS)
libargon2_sse2_la_CFLAGS   = $(AM_CFLAGS)
if HAVE_SSE2
libargon2_sse2_la_CPPFLAGS += -DHAVE_SSE2
libargon2_sse2_la_CFLAGS   += -msse2
endif # HAVE_SSE2

libargon2_ssse3_la_SOURCES = \
	arch/@ARCH@/lib/argon2-ssse3.c \
	arch/@ARCH@/lib/argon2-ssse3.h \
	arch/@ARCH@/lib/argon2-template-128.h
libargon2_ssse3_la_CPPFLAGS = $(AM_CPPFLAGS)
libargon2_ssse3_la_CFLAGS   = $(AM_CFLAGS)
if HAVE_SSSE3
libargon2_ssse3_la_CPPFLAGS += -DHAVE_SSSE3
libargon2_ssse3_la_CFLAGS   += -mssse3
endif # HAVE_SSSE3

libargon2_xop_la_SOURCES = \
	arch/@ARCH@/lib/argon2-xop.c \
	arch/@ARCH@/lib/argon2-xop.h \
	arch/@ARCH@/lib/argon2-template-128.h
libargon2_xop_la_CPPFLAGS = $(AM_CPPFLAGS)
libargon2_xop_la_CFLAGS   = $(AM_CFLAGS)
if HAVE_XOP
libargon2_xop_la_CPPFLAGS += -DHAVE_XOP
libargon2_xop_la_CFLAGS   += -mxop
endif # HAVE_XOP

libargon2_avx2_la_SOURCES = \
	arch/@ARCH@/lib/argon2-avx2.c \
	arch/@ARCH@/lib/argon2-avx2.h
libargon2_avx2_la_CPPFLAGS = $(AM_CPPFLAGS)
libargon2_avx2_la_CFLAGS   = $(AM_CFLAGS)
if HAVE_AVX2
libargon2_avx2_la_CPPFLAGS += -DHAVE_AVX2
libargon2_avx2_la_CFLAGS   += -mavx2
endif # HAVE_AVX2

libargon2_avx512f_la_SOURCES = \
	arch/@ARCH@/lib/argon2-avx512f.c \
	arch/@ARCH@/lib/argon2-avx512f.h
libargon2_avx512f_la_CPPFLAGS = $(AM_CPPFLAGS)
libargon2_avx512f_la_CFLAGS   = $(AM_CFLAGS)
if HAVE_AVX512F
libargon2_avx512f_la_CPPFLAGS += -DHAVE_AVX512F
libargon2_avx512f_la_CFLAGS   += -mavx512f
endif # HAVE_AVX512F

libargon2_la_SOURCES += \
	arch/@ARCH@/lib/argon2-arch.c \
	arch/@ARCH@/lib/cpu-flags.c \
	arch/@ARCH@/lib/cpu-flags.h
endif # ARCH_X86_64

if ARCH_GENERIC
libargon2_la_SOURCES += \
	arch/@ARCH@/lib/argon2-arch.c
endif # ARCH_GENERIC

argon2_LDADD = libargon2.la
argon2_SOURCES = src/run.c lib/core.h

argon2_genkat_LDADD = libargon2.la
argon2_genkat_SOURCES = src/genkat.c

argon2_bench2_LDADD = libargon2.la -lrt
argon2_bench2_SOURCES = src/bench2.c src/timing.h

argon2_test_LDADD = libargon2.la
argon2_test_SOURCES = tests/test.c
