#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "interfaces/ILogBackend.h"
#include "log/Log.h"
Log *Log::m_self = nullptr;

void Log::text(const char* fmt, ...){}
Log::~Log()
{
    for (auto backend : m_backends) {
        delete backend;
    }
}
