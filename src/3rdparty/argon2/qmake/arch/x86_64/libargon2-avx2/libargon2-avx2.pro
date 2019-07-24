QT       -= core gui

TARGET = argon2-avx2
TEMPLATE = lib
CONFIG += staticlib

ARGON2_ROOT = ../../../..

INCLUDEPATH += \
    $$ARGON2_ROOT/include \
    $$ARGON2_ROOT/lib \
    $$ARGON2_ROOT/arch/$$ARCH/lib

USE_AVX2 {
    DEFINES += HAVE_AVX2
    QMAKE_CFLAGS += -mavx2
}

SOURCES += \
    $$ARGON2_ROOT/arch/x86_64/lib/argon2-avx2.c

HEADERS += \
    $$ARGON2_ROOT/arch/x86_64/lib/argon2-avx2.h
