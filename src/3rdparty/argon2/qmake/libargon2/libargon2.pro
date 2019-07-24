#-------------------------------------------------
#
# Project created by QtCreator 2016-08-08T17:43:00
#
#-------------------------------------------------

QT       -= core gui

TARGET = argon2
TEMPLATE = lib

ARGON2_ROOT = ../..

INCLUDEPATH += \
    $$ARGON2_ROOT/include \
    $$ARGON2_ROOT/lib

SOURCES += \
    $$ARGON2_ROOT/lib/argon2.c \
    $$ARGON2_ROOT/lib/core.c \
    $$ARGON2_ROOT/lib/encoding.c \
    $$ARGON2_ROOT/lib/genkat.c \
    $$ARGON2_ROOT/lib/impl-select.c \
    $$ARGON2_ROOT/lib/thread.c \
    $$ARGON2_ROOT/lib/blake2/blake2.c

HEADERS += \
    $$ARGON2_ROOT/include/argon2.h \
    $$ARGON2_ROOT/lib/argon2-template-64.h \
    $$ARGON2_ROOT/lib/core.h \
    $$ARGON2_ROOT/lib/encoding.h \
    $$ARGON2_ROOT/lib/genkat.h \
    $$ARGON2_ROOT/lib/impl-select.h \
    $$ARGON2_ROOT/lib/thread.h \
    $$ARGON2_ROOT/lib/blake2/blake2.h \
    $$ARGON2_ROOT/lib/blake2/blake2-impl.h

equals(ARCH, x86_64) {
    SOURCES += \
        $$ARGON2_ROOT/arch/$$ARCH/lib/cpu-flags.c \
        $$ARGON2_ROOT/arch/$$ARCH/lib/argon2-arch.c

    HEADERS += \
        $$ARGON2_ROOT/arch/$$ARCH/lib/cpu-flags.h

    # libargon2-sse2.a:
    win32:CONFIG(release, debug|release): LIBS += -L$$OUT_PWD/../arch/x86_64/libargon2-sse2/release/ -largon2-sse2
    else:win32:CONFIG(debug, debug|release): LIBS += -L$$OUT_PWD/../arch/x86_64/libargon2-sse2/debug/ -largon2-sse2
    else:unix: LIBS += -L$$OUT_PWD/../arch/x86_64/libargon2-sse2/ -largon2-sse2

    DEPENDPATH += $$PWD/../arch/x86_64/libargon2-sse2

    win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$OUT_PWD/../arch/x86_64/libargon2-sse2/release/libargon2-sse2.a
    else:win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$OUT_PWD/../arch/x86_64/libargon2-sse2/debug/libargon2-sse2.a
    else:win32:!win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$OUT_PWD/../arch/x86_64/libargon2-sse2/release/argon2-sse2.lib
    else:win32:!win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$OUT_PWD/../arch/x86_64/libargon2-sse2/debug/argon2-sse2.lib
    else:unix: PRE_TARGETDEPS += $$OUT_PWD/../arch/x86_64/libargon2-sse2/libargon2-sse2.a

    # libargon2-ssse3.a:
    win32:CONFIG(release, debug|release): LIBS += -L$$OUT_PWD/../arch/x86_64/libargon2-ssse3/release/ -largon2-ssse3
    else:win32:CONFIG(debug, debug|release): LIBS += -L$$OUT_PWD/../arch/x86_64/libargon2-ssse3/debug/ -largon2-ssse3
    else:unix: LIBS += -L$$OUT_PWD/../arch/x86_64/libargon2-ssse3/ -largon2-ssse3

    DEPENDPATH += $$PWD/../arch/x86_64/libargon2-ssse3

    win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$OUT_PWD/../arch/x86_64/libargon2-ssse3/release/libargon2-ssse3.a
    else:win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$OUT_PWD/../arch/x86_64/libargon2-ssse3/debug/libargon2-ssse3.a
    else:win32:!win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$OUT_PWD/../arch/x86_64/libargon2-ssse3/release/argon2-ssse3.lib
    else:win32:!win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$OUT_PWD/../arch/x86_64/libargon2-ssse3/debug/argon2-ssse3.lib
    else:unix: PRE_TARGETDEPS += $$OUT_PWD/../arch/x86_64/libargon2-ssse3/libargon2-ssse3.a

    # libargon2-xop.a:
    win32:CONFIG(release, debug|release): LIBS += -L$$OUT_PWD/../arch/x86_64/libargon2-xop/release/ -largon2-xop
    else:win32:CONFIG(debug, debug|release): LIBS += -L$$OUT_PWD/../arch/x86_64/libargon2-xop/debug/ -largon2-xop
    else:unix: LIBS += -L$$OUT_PWD/../arch/x86_64/libargon2-xop/ -largon2-xop

    DEPENDPATH += $$PWD/../arch/x86_64/libargon2-xop

    win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$OUT_PWD/../arch/x86_64/libargon2-xop/release/libargon2-xop.a
    else:win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$OUT_PWD/../arch/x86_64/libargon2-xop/debug/libargon2-xop.a
    else:win32:!win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$OUT_PWD/../arch/x86_64/libargon2-xop/release/argon2-xop.lib
    else:win32:!win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$OUT_PWD/../arch/x86_64/libargon2-xop/debug/argon2-xop.lib
    else:unix: PRE_TARGETDEPS += $$OUT_PWD/../arch/x86_64/libargon2-xop/libargon2-xop.a

    # libargon2-avx2.a:
    win32:CONFIG(release, debug|release): LIBS += -L$$OUT_PWD/../arch/x86_64/libargon2-avx2/release/ -largon2-avx2
    else:win32:CONFIG(debug, debug|release): LIBS += -L$$OUT_PWD/../arch/x86_64/libargon2-avx2/debug/ -largon2-avx2
    else:unix: LIBS += -L$$OUT_PWD/../arch/x86_64/libargon2-avx2/ -largon2-avx2

    DEPENDPATH += $$PWD/../arch/x86_64/libargon2-avx2

    win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$OUT_PWD/../arch/x86_64/libargon2-avx2/release/libargon2-avx2.a
    else:win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$OUT_PWD/../arch/x86_64/libargon2-avx2/debug/libargon2-avx2.a
    else:win32:!win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$OUT_PWD/../arch/x86_64/libargon2-avx2/release/argon2-avx2.lib
    else:win32:!win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$OUT_PWD/../arch/x86_64/libargon2-avx2/debug/argon2-avx2.lib
    else:unix: PRE_TARGETDEPS += $$OUT_PWD/../arch/x86_64/libargon2-avx2/libargon2-avx2.a

    # libargon2-avx512f.a:
    win32:CONFIG(release, debug|release): LIBS += -L$$OUT_PWD/../arch/x86_64/libargon2-avx512f/release/ -largon2-avx512f
    else:win32:CONFIG(debug, debug|release): LIBS += -L$$OUT_PWD/../arch/x86_64/libargon2-avx512f/debug/ -largon2-avx512f
    else:unix: LIBS += -L$$OUT_PWD/../arch/x86_64/libargon2-avx512f/ -largon2-avx512f

    DEPENDPATH += $$PWD/../arch/x86_64/libargon2-avx512f

    win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$OUT_PWD/../arch/x86_64/libargon2-avx512f/release/libargon2-avx512f.a
    else:win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$OUT_PWD/../arch/x86_64/libargon2-avx512f/debug/libargon2-avx512f.a
    else:win32:!win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$OUT_PWD/../arch/x86_64/libargon2-avx512f/release/argon2-avx512f.lib
    else:win32:!win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$OUT_PWD/../arch/x86_64/libargon2-avx512f/debug/argon2-avx512f.lib
    else:unix: PRE_TARGETDEPS += $$OUT_PWD/../arch/x86_64/libargon2-avx512f/libargon2-avx512f.a
}
equals(ARCH, generic) {
    SOURCES += \
        $$ARGON2_ROOT/arch/$$ARCH/lib/argon2-arch.c
}

unix {
    target.path = /usr/lib
    INSTALLS += target
}
