QT       -= core gui

TARGET = argon2-sse2
TEMPLATE = lib
CONFIG += staticlib

ARGON2_ROOT = ../../../..

INCLUDEPATH += \
    $$ARGON2_ROOT/include \
    $$ARGON2_ROOT/lib \
    $$ARGON2_ROOT/arch/$$ARCH/lib

USE_SSE2 | USE_SSSE3 | USE_XOP | USE_AVX2 {
    DEFINES += HAVE_SSE2
    QMAKE_CFLAGS += -msse2
}

SOURCES += \
    $$ARGON2_ROOT/arch/x86_64/lib/argon2-sse2.c

HEADERS += \
    $$ARGON2_ROOT/arch/x86_64/lib/argon2-sse2.h \
    $$ARGON2_ROOT/arch/x86_64/lib/argon2-template-128.h
