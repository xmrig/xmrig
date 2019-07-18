QT       -= core gui

TARGET = argon2-avx512f
TEMPLATE = lib
CONFIG += staticlib

ARGON2_ROOT = ../../../..

INCLUDEPATH += \
    $$ARGON2_ROOT/include \
    $$ARGON2_ROOT/lib \
    $$ARGON2_ROOT/arch/$$ARCH/lib

USE_AVX512F {
    DEFINES += HAVE_AVX512F
    QMAKE_CFLAGS += -mavx512f
}

SOURCES += \
    $$ARGON2_ROOT/arch/x86_64/lib/argon2-avx512f.c

HEADERS += \
    $$ARGON2_ROOT/arch/x86_64/lib/argon2-avx512f.h
