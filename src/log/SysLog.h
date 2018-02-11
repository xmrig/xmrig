#ifndef __SYSLOG_H__
#define __SYSLOG_H__
#include "interfaces/ILogBackend.h"
class SysLog : public ILogBackend
{
public:
    SysLog();

    void message(int level, const char *fmt, va_list args) override;
    void text(const char *fmt, va_list args) override;
};

#endif