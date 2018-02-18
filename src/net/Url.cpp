/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2016-2017 XMRig       <support@xmrig.com>
 *
 *
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#include <string.h>
#include <stdlib.h>
#include <algorithm>

#include "net/Url.h"

#ifdef _MSC_VER
#define strncasecmp(x,y,z) _strnicmp(x,y,z)
#endif

Url::Url()
	: m_keepAlive(false),
	  m_nicehash(false),
	  m_host(),
	  m_password(),
	  m_user(),
	  m_port(kDefaultPort),
	  m_proxy_host(),
	  m_proxy_port(kDefaultProxyPort),
	  m_keystream()
{
}

Url::Url(const std::string & url)
	: m_keepAlive(false),
	  m_nicehash(false),
	  m_host(),
	  m_password(),
	  m_user(),
	  m_port(kDefaultPort),
	  m_proxy_host(),
	  m_proxy_port(kDefaultProxyPort),
	  m_keystream()
{
	parse(url);
}

Url::Url(const std::string & host,
         uint16_t port,
         const std::string & user,
         const std::string & password,
         bool keepAlive,
         bool nicehash)
	: m_keepAlive(keepAlive),
	  m_nicehash(nicehash),
	  m_host(host),
	  m_password(password),
	  m_user(user),
	  m_port(port),
	  m_proxy_host(),
	  m_proxy_port(kDefaultProxyPort),
	  m_keystream()
{
}

Url::~Url()
{
}

bool Url::isNicehash() const
{
	return isValid() && (m_nicehash || m_host.find(".nicehash.com") != std::string::npos);
}

/**
 * @brief Parse url.
 *
 * Valid urls:
 * example.com
 * example.com:3333
 * example.com:3333#keystream
 * example.com:3333@proxy
 * example.com:3333@proxy:8080
 * example.com:3333#keystream@proxy
 * example.com:3333#keystream@proxy:8080
 * stratum+tcp://example.com
 * stratum+tcp://example.com:3333
 * stratum+tcp://example.com:3333#keystream
 * stratum+tcp://example.com:3333@proxy
 * stratum+tcp://example.com:3333@proxy:8080
 * stratum+tcp://example.com:3333#keystream@proxy
 * stratum+tcp://example.com:3333#keystream@proxy:8080
 *
 * @param url
 */
bool Url::parse(const std::string & url)
{
	size_t base = 0;

	const size_t p = url.find("://");
	if(p != std::string::npos)
	{
		static const std::string STRATUM_PREFIX = "stratum+tcp://";
		if(strncasecmp(url.c_str(), STRATUM_PREFIX.c_str(), STRATUM_PREFIX.size()))
		{
			return false;
		}

		base = STRATUM_PREFIX.size();
	}

	const std::string path = url.substr(base);
	if(path.empty() || path[0] == '/')
	{
		return false;
	}

	const size_t port = path.find_first_of(':');
	size_t portini = port;
	if(port != std::string::npos)
	{
		m_host = path.substr(0, port);
		m_port = (uint16_t) strtol(path.substr(port + 1).c_str(), nullptr, 10);
	}
	else
	{
		portini = 0;
	}

	const size_t proxy = path.find_first_of('@', portini);
	const size_t keystream = path.find_first_of('#', portini);
	if(keystream != std::string::npos)
	{
		if(port == std::string::npos)
		{
			m_host = path.substr(0, keystream);
		}
		if(proxy != std::string::npos)
		{
			m_keystream = path.substr(keystream + 1, proxy - keystream - 1);
		}
		else
		{
			m_keystream = path.substr(keystream + 1);
		}
	}

	if(proxy == std::string::npos)
	{
		if(port == std::string::npos && keystream == std::string::npos)
		{
			m_host = path;
		}
		return true;
	}
	else
	{
		if(port == std::string::npos && keystream == std::string::npos)
		{
			m_host = path.substr(0, proxy);
		}
	}

	const size_t proxyini = proxy + 1;

	const size_t proxyport = path.find_first_of(':', proxyini);
	if(proxyport == std::string::npos)
	{
		m_proxy_host = path.substr(proxyini);
		return false;
	}

	m_proxy_host = path.substr(proxyini, proxyport - proxyini);
	m_proxy_port = (uint16_t) strtol(path.substr(proxyport + 1).c_str(), nullptr, 10);

	return true;
}

bool Url::setUserpass(const std::string & userpass)
{
	const size_t p = userpass.find_first_of(':');
	if(p == std::string::npos)
	{
		return false;
	}

	m_user = userpass.substr(0, p);
	m_password = userpass.substr(p + 1);

	return true;
}


void Url::applyExceptions()
{
	if(!isValid())
	{
		return;
	}

	if(m_host.find(".nicehash.com") != std::string::npos)
	{
		m_keepAlive = false;
		m_nicehash  = true;
	}

	if(m_host.find(".minergate.com") != std::string::npos)
	{
		m_keepAlive = false;
	}
}



void Url::setPassword(const std::string & password)
{
	m_password = password;
}

void Url::setUser(const std::string & user)
{
	m_user = user;
}

void Url::copyKeystream(char* keystreamDest, const size_t keystreamLen) const
{
	if(hasKeystream())
	{
		memset(keystreamDest, 1, keystreamLen);
		memcpy(keystreamDest, m_keystream.c_str(), std::min(keystreamLen, m_keystream.size()));
	}
}
