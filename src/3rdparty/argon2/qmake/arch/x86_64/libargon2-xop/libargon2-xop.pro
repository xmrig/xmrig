QT       -= core gui

TARGET = argon2-xop
TEMPLATE = lib
CONFIG += staticlib

ARGON2_ROOT = ../../../..

INCLUDEPATH += \
    $$ARGON2_ROOT/include \
    $$ARGON2_ROOT/lib \
    $$ARGON2_ROOT/arch/$$ARCH/lib

USE_XOP {
    DEFINES += HAVE_XOP
    QMAKE_CFLAGS += -mxop
}

SOURCES += \
    $$ARGON2_ROOT/arch/x86_64/lib/argon2-xop.c

HEADERS += \
    $$ARGON2_ROOT/arch/x86_64/lib/argon2-xop.h \
    $$ARGON2_ROOT/arch/x86_64/lib/argon2-template-128.h
