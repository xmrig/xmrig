#ifndef __LOG_H__
#define __LOG_H__


#include <uv.h>
#include <vector>


class ILogBackend;


class Log
{
public:
    enum Level {
        ERR,
        WARNING,
        NOTICE,
        INFO,
        DEBUG
    };

    constexpr static const char* kCL_N      = "\x1B[0m";
    constexpr static const char* kCL_RED    = "\x1B[31m";
    constexpr static const char* kCL_YELLOW = "\x1B[33m";
    constexpr static const char* kCL_WHITE  = "\x1B[01;37m";

#   ifdef WIN32
    constexpr static const char* kCL_GRAY = "\x1B[01;30m";
#   else
    constexpr static const char* kCL_GRAY = "\x1B[90m";
#   endif

    static inline Log* i()                       { return m_self; }
    static inline void add(ILogBackend *backend) { i()->m_backends.push_back(backend); }
    static inline void init()                    { if (!m_self) { m_self = new Log();} }
    static inline void release()                 { delete m_self; }

    void message(Level level, const char* fmt, ...);
    void text(const char* fmt, ...);

private:
    inline Log() {}
    ~Log();

    static Log *m_self;
    std::vector<ILogBackend*> m_backends;
};


#define LOG_ERR(x, ...)    Log::i()->message(Log::ERR,     x, ##__VA_ARGS__)
#define LOG_WARN(x, ...)   Log::i()->message(Log::WARNING, x, ##__VA_ARGS__)
#define LOG_NOTICE(x, ...) Log::i()->message(Log::NOTICE,  x, ##__VA_ARGS__)
#define LOG_INFO(x, ...)   Log::i()->message(Log::INFO,    x, ##__VA_ARGS__)

#ifdef APP_DEBUG
#   define LOG_DEBUG(x, ...)      Log::i()->message(Log::DEBUG,   x, ##__VA_ARGS__)
#else
#   define LOG_DEBUG(x, ...)
#endif

#if defined(APP_DEBUG) || defined(APP_DEVEL)
#   define LOG_DEBUG_ERR(x, ...)  Log::i()->message(Log::ERR,     x, ##__VA_ARGS__)
#   define LOG_DEBUG_WARN(x, ...) Log::i()->message(Log::WARNING, x, ##__VA_ARGS__)
#else
#   define LOG_DEBUG_ERR(x, ...)
#   define LOG_DEBUG_WARN(x, ...)
#endif

#endif