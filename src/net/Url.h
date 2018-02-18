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

#ifndef __URL_H__
#define __URL_H__

#include <stdint.h>

#include <string>

#include "interfaces/interface.h"

class Url
{
public:
	static const std::string & DefaultPassword()
	{
		static const std::string kDefaultPassword = "x";
		return kDefaultPassword;
	}
	static const std::string & DefaultUser()
	{
		static const std::string kDefaultUser = "";
		return kDefaultUser;
	}

	enum
	{
		kDefaultPort        = 3333,
		kDefaultProxyPort   = 8080,
	};

	Url();
	Url(const std::string & url);
	Url(const std::string & host, uint16_t port, const std::string & user = "",
	    const std::string & password = "",
	    bool keepAlive = false, bool nicehash = false);
	~Url();

	inline bool isKeepAlive() const
	{
		return m_keepAlive;
	}
	inline bool isValid() const
	{
		return m_host.size() > 0 && m_port > 0;
	}
	inline bool hasKeystream() const
	{
		return m_keystream.size() > 0;
	}
	inline const std::string & host() const
	{
		return isProxyed() ? proxyHost() : finalHost();
	}
	inline const std::string & password() const
	{
		return m_password.empty() ? DefaultPassword() : m_password;
	}
	inline const std::string & user() const
	{
		return m_user.empty() ? DefaultUser() : m_user;
	}
	inline uint16_t port() const
	{
		return isProxyed() ? proxyPort() : finalPort();
	}
	inline bool isProxyed() const
	{
		return proxyHost().size() > 0;
	}
	inline const std::string & finalHost() const
	{
		return m_host;
	}
	inline uint16_t finalPort() const
	{
		return m_port;
	}
	inline const std::string & proxyHost() const
	{
		return m_proxy_host;
	}
	inline uint16_t proxyPort() const
	{
		return m_proxy_port;
	}
	inline void setProxyHost(const std::string & value)
	{
		m_proxy_host = value;
	}
	inline void setProxyPort(const uint16_t value)
	{
		m_proxy_port = value;
	}
	inline void setKeepAlive(bool keepAlive)
	{
		m_keepAlive = keepAlive;
	}
	inline void setNicehash(bool nicehash)
	{
		m_nicehash = nicehash;
	}

	bool isNicehash() const;
	bool parse(const std::string & url);
	bool setUserpass(const std::string & userpass);
	void applyExceptions();
	void setPassword(const std::string & password);
	void setUser(const std::string & user);
	void copyKeystream(char* keystreamDest, const size_t keystreamLen) const;

private:
	bool m_keepAlive;
	bool m_nicehash;
	std::string m_host;
	std::string m_password;
	std::string m_user;
	uint16_t m_port;
	std::string m_proxy_host;
	uint16_t m_proxy_port;
	std::string m_keystream;
};

#endif /* __URL_H__ */
