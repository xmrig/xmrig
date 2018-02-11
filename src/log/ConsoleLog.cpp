#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef WIN32
#   include <winsock2.h>
#   include <windows.h>
#endif


#include "log/ConsoleLog.h"
#include "log/Log.h"
#include "Options.h"


ConsoleLog::ConsoleLog(bool colors) :
    m_colors(colors),
    m_stream(nullptr){}
void ConsoleLog::message(int level, const char* fmt, va_list args){}
void ConsoleLog::text(const char* fmt, va_list args){}
bool ConsoleLog::isWritable() const{}
void ConsoleLog::print(va_list args){}