QT       -= core gui

TARGET = argon2-ssse3
TEMPLATE = lib
CONFIG += staticlib

ARGON2_ROOT = ../../../..

INCLUDEPATH += \
    $$ARGON2_ROOT/include \
    $$ARGON2_ROOT/lib \
    $$ARGON2_ROOT/arch/$$ARCH/lib

USE_SSSE3 | USE_XOP | USE_AVX2 {
    DEFINES += HAVE_SSSE3
    QMAKE_CFLAGS += -mssse3
}

SOURCES += \
    $$ARGON2_ROOT/arch/x86_64/lib/argon2-ssse3.c

HEADERS += \
    $$ARGON2_ROOT/arch/x86_64/lib/argon2-ssse3.h \
    $$ARGON2_ROOT/arch/x86_64/lib/argon2-template-128.h
