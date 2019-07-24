TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

ARGON2_ROOT = ../..

SOURCES += \
    $$ARGON2_ROOT/tests/test.c

win32:CONFIG(release, debug|release): LIBS += -L$$OUT_PWD/../libargon2/release/ -largon2
else:win32:CONFIG(debug, debug|release): LIBS += -L$$OUT_PWD/../libargon2/debug/ -largon2
else:unix: LIBS += -L$$OUT_PWD/../libargon2/ -largon2

INCLUDEPATH += $$PWD/../../include
DEPENDPATH  += $$PWD/../../include
