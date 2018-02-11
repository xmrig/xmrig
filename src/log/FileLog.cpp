#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "log/FileLog.h"
FileLog::FileLog(const char *fileName){}
void FileLog::message(int level, const char* fmt, va_list args){}
void FileLog::text(const char* fmt, va_list args){}
void FileLog::onWrite(uv_fs_t *req){}
void FileLog::write(char *data, size_t size){}