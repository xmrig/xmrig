#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include "net/Url.h"
#ifdef _MSC_VER
#   define strncasecmp(x,y,z) _strnicmp(x,y,z)
#endif
Url::Url() :
    m_keepAlive(false),
    m_nicehash(false),
    m_host(nullptr),
    m_password(nullptr),
    m_user(nullptr),
    m_port(kDefaultPort)
{
}
Url::Url(const char *url) :
    m_keepAlive(false),
    m_nicehash(false),
    m_host(nullptr),
    m_password(nullptr),
    m_user(nullptr),
    m_port(kDefaultPort)
{
    parse(url);
}
Url::Url(const char *host, uint16_t port, const char *user, const char *password, bool keepAlive, bool nicehash) :
    m_keepAlive(keepAlive),
    m_nicehash(nicehash),
    m_password(password ? strdup(password) : nullptr),
    m_user(user ? strdup(user) : nullptr),
    m_port(port)
{
    m_host = strdup(host);
}
Url::~Url()
{
    free(m_host);
    free(m_password);
    free(m_user);
}
bool Url::parse(const char *url)
{
    const char *base = url;
    if (!strlen(base) || *base == '/') { return false; }
    const char *port = strchr(base, ':');
    if (!port) {
        m_host = strdup(base);
        return false;
    }
    const size_t size = port++ - base + 1;
    m_host = static_cast<char*>(malloc(size));
    memcpy(m_host, base, size - 1);
    m_host[size - 1] = '\0';
    m_port = (uint16_t) strtol(port, nullptr, 10);
    return true;
}
bool Url::setUserpass(const char *userpass)
{
    const char *p = strchr(userpass, ':');
    if (!p) { return false; }
    free(m_user);
    free(m_password);
    m_user = static_cast<char*>(calloc(p - userpass + 1, 1));
    strncpy(m_user, userpass, p - userpass);
    m_password = strdup(p + 1);
    return true;
}
void Url::applyExceptions()
{
    if (!isValid()) { return; }
}
void Url::setPassword(const char *password)
{
	if (!password) {return;}
	free(m_password);
	m_password = strdup(password);
}
void Url::setUser(const char *user)
{
	if (!user) { return; }
	free(m_user);
	m_user = strdup(user);
}
Url &Url::operator=(const Url *other)
{
    m_keepAlive = other->m_keepAlive;
    m_nicehash  = other->m_nicehash;
    m_port      = other->m_port;
    free(m_host);
    m_host = strdup(other->m_host);
    setPassword(other->m_password);
    setUser(other->m_user);
    return *this;
}