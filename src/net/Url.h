#ifndef __URL_H__
#define __URL_H__
#include <stdint.h>
class Url
{
public:
    constexpr static const char *kDefaultPassword = "x";
    constexpr static const char *kDefaultUser     = "x";
    constexpr static uint16_t kDefaultPort        = 3333;

    Url();
    Url(const char *url);
    Url(const char *host, uint16_t port, const char *user = nullptr, const char *password = nullptr, bool keepAlive = false, bool nicehash = false  );
    ~Url();

    inline bool isKeepAlive() const          { return m_keepAlive; }
    inline bool isNicehash() const           { return m_nicehash; }
    inline bool isValid() const              { return m_host && m_port > 0; }
    inline const char *host() const          { return m_host; }
    inline const char *password() const      { return m_password ? m_password : kDefaultPassword; }
    inline const char *user() const          { return m_user ? m_user : kDefaultUser; }
    inline uint16_t port() const             { return m_port; }
    inline void setKeepAlive(bool keepAlive) { m_keepAlive = keepAlive; }
    inline void setNicehash(bool nicehash)   { m_nicehash = nicehash; }

    bool parse(const char *url);
    bool setUserpass(const char *userpass);
    void applyExceptions();
    void setPassword(const char *password);
    void setUser(const char *user);

    Url &operator=(const Url *other);

private:
    bool m_keepAlive;
    bool m_nicehash;
    char *m_host;
    char *m_password;
    char *m_user;
    uint16_t m_port;
};

#endif