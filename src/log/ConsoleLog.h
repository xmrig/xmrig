#ifndef __CONSOLELOG_H__
#define __CONSOLELOG_H__
#include <uv.h>
#include "interfaces/ILogBackend.h"
class ConsoleLog : public ILogBackend
{
public:
    ConsoleLog(bool colors);

    void message(int level, const char *fmt, va_list args) override;
    void text(const char *fmt, va_list args) override;

private:
    bool isWritable() const;
    void print(va_list args);

    bool m_colors;
    char m_buf[512];
    char m_fmt[256];
    uv_buf_t m_uvBuf;
    uv_stream_t *m_stream;
    uv_tty_t m_tty;
};

#endif