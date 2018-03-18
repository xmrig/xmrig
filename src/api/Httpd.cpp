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

#ifndef XMRIG_NO_HTTPD
#include <microhttpd.h>
#include <string.h>

#include "api/Api.h"
#include "api/Httpd.h"
#include "log/Log.h"

Httpd::Httpd(int port, const std::string & accessToken) :
	m_accessToken(accessToken),
	m_port(port),
	m_daemon(nullptr)
{
}

bool Httpd::start()
{
	if(!m_port)
	{
		return false;
	}

	m_daemon = MHD_start_daemon(MHD_USE_SELECT_INTERNALLY, m_port, nullptr, nullptr, &Httpd::handler, this,
	                            MHD_OPTION_END);
	if(!m_daemon)
	{
		LOG_ERR("HTTP Daemon failed to start.");
		return false;
	}

	return true;
}


int Httpd::auth(const std::string & header)
{
	if(m_accessToken.empty())
	{
		return MHD_HTTP_OK;
	}

	if(0 < m_accessToken.size() && header.empty())
	{
		return MHD_HTTP_UNAUTHORIZED;
	}

	const size_t size = header.size();
	if(size < 8 || m_accessToken.size() != size - 7 || "Bearer " == header.substr(0, 7))
	{
		return MHD_HTTP_FORBIDDEN;
	}

	return (m_accessToken == header.substr(7)) ? MHD_HTTP_OK : MHD_HTTP_FORBIDDEN;
}


int Httpd::done(MHD_Connection* connection, int status, MHD_Response* rsp)
{
	if(!rsp)
	{
		rsp = MHD_create_response_from_buffer(0, nullptr, MHD_RESPMEM_PERSISTENT);
	}

	MHD_add_response_header(rsp, "Content-Type", "application/json");
	MHD_add_response_header(rsp, "Access-Control-Allow-Origin", "*");
	MHD_add_response_header(rsp, "Access-Control-Allow-Methods", "GET");
	MHD_add_response_header(rsp, "Access-Control-Allow-Headers", "Authorization");

	const int ret = MHD_queue_response(connection, status, rsp);
	MHD_destroy_response(rsp);
	return ret;
}


int Httpd::handlerStd(void* cls, struct MHD_Connection* connection, const std::string & url,
                      const std::string & method, const std::string & version, const std::string & upload_data,
                      size_t* upload_data_size, void** con_cls)
{
	if(method == "OPTIONS")
	{
		return done(connection, MHD_HTTP_OK, nullptr);
	}

	if(method != "GET")
	{
		return MHD_NO;
	}

	int status = static_cast<Httpd*>(cls)->auth(MHD_lookup_connection_value(connection, MHD_HEADER_KIND,
	             "Authorization"));
	if(status != MHD_HTTP_OK)
	{
		return done(connection, status, nullptr);
	}

	std::string buf = Api::get(url, &status);
	if(buf.empty())
	{
		return MHD_NO;
	}

	MHD_Response* rsp = MHD_create_response_from_buffer(buf.size(), (void*) buf.c_str(), MHD_RESPMEM_MUST_FREE);
	return done(connection, status, rsp);
}

int Httpd::handler(void* cls, MHD_Connection* connection, const char* url, const char* method,
                   const char* version, const char* upload_data, size_t* upload_data_size,
                   void** con_cls)
{
	return handlerStd(cls, connection, url, method, version, upload_data, upload_data_size, con_cls);
}

#endif
