#ifndef __PLATFORM_H__
#define __PLATFORM_H__

class Platform
{
public:
    static const char *defaultConfigName();
    static void init(const char *userAgent);
    static void release();
    static void setProcessPriority(int priority);
    static void setThreadPriority(int priority);

    static inline const char *userAgent() { return m_userAgent; }

private:
    static char *m_defaultConfigName;
    static char *m_userAgent;
};
#endif /* __PLATFORM_H__ */
