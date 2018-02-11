#ifndef __ILOGBACKEND_H__
#define __ILOGBACKEND_H__
#include <stdarg.h>
class ILogBackend
{
public:
    virtual ~ILogBackend() {}

    virtual void message(int level, const char* fmt, va_list args) = 0;
    virtual void text(const char* fmt, va_list args)               = 0;
};


#endif