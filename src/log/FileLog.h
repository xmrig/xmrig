#ifndef __FILELOG_H__
#define __FILELOG_H__
#include <uv.h>
#include "interfaces/ILogBackend.h"
class FileLog : public ILogBackend
{
public:
    FileLog(const char *fileName);

    void message(int level, const char* fmt, va_list args) override;
    void text(const char* fmt, va_list args) override;

private:
    static void onWrite(uv_fs_t *req);

    void write(char *data, size_t size);

    int m_file;
};

#endif