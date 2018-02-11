#include <syslog.h>
#include "log/SysLog.h"
#include "version.h"
SysLog::SysLog() { openlog(APP_ID, LOG_PID, LOG_USER);}
void SysLog::message(int level, const char *fmt, va_list args) { vsyslog(level, fmt, args); }
void SysLog::text(const char *fmt, va_list args) { message(LOG_INFO, fmt, args); }