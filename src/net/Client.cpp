/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2016-2018 XMRig       <support@xmrig.com>
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

#include <inttypes.h>
#include <iterator>
#include <stdio.h>
#include <string.h>
#include <utility>


#include "interfaces/IClientListener.h"
#include "log/Log.h"
#include "net/Client.h"
#include "net/Url.h"
#include "rapidjson/document.h"
#include "rapidjson/error/en.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

#ifdef XMRIG_PROXY_PROJECT
#include "proxy/JobResult.h"
#else
#include "net/JobResult.h"
#endif


#ifdef _MSC_VER
#define strncasecmp(x,y,z) _strnicmp(x,y,z)
#endif


int64_t Client::m_sequence = 1;


Client::Client(int id, const std::string & agent, IClientListener* listener) :
	m_quiet(false),
	m_encrypted(false),
	m_agent(agent),
	m_listener(listener),
	m_id(id),
	m_retryPause(5000),
	m_failures(0),
	m_recvBufPos(0),
	m_state(UnconnectedState),
	m_expire(0),
	m_stream(nullptr),
	m_socket(nullptr)
{
	memset(&m_hints, 0, sizeof(m_hints));
	memset(m_keystream, 0, sizeof(m_keystream));

	m_resolver.data = this;

	m_hints.ai_family   = PF_INET;
	m_hints.ai_socktype = SOCK_STREAM;
	m_hints.ai_protocol = IPPROTO_TCP;

	m_recvBuf.base = m_buf;
	m_recvBuf.len  = sizeof(m_buf);

#ifndef XMRIG_PROXY_PROJECT
	m_keepAliveTimer.data = this;
	uv_timer_init(uv_default_loop(), &m_keepAliveTimer);
#endif
}


Client::~Client()
{
	delete m_socket;
}


void Client::connect()
{
	resolve(m_url.host());
}


/**
 * @brief Connect to server.
 *
 * @param url
 */
void Client::connect(const Url & url)
{
	setUrl(url);
	resolve(m_url.host());
}


void Client::disconnect()
{
#ifndef XMRIG_PROXY_PROJECT
	uv_timer_stop(&m_keepAliveTimer);
#endif

	m_expire   = 0;
	m_failures = -1;

	close();
}


void Client::setUrl(const Url & url)
{
	if(false == url.isValid())
	{
		return;
	}

	if(url.hasKeystream())
	{
		url.copyKeystream(m_keystream, sizeof(m_keystream));
		m_encrypted = true;
	}
	else
	{
		m_encrypted = false;
	}

	m_url = url;
}


void Client::tick(uint64_t now)
{
	if(m_expire == 0 || now < m_expire)
	{
		return;
	}

	if(m_state == ConnectedState)
	{
		LOG_DEBUG_ERR("[" << m_url.host() << ":" << m_url.port() << "] timeout");
		close();
	}


	if(m_state == ConnectingState)
	{
		connect();
	}
}


int64_t Client::submit(const JobResult & result)
{
#ifdef XMRIG_PROXY_PROJECT
	const std::string nonce = result.nonce;
	const std::string data  = result.result;
#else
	char nonce_buffer[9];
	char data_buffer[65];

	char* nonceChar = (char*)(&result.nonce);
	Job::toHex(std::string(nonceChar, 4), nonce_buffer);
	nonce_buffer[8] = '\0';

	char* resultChar = (char*)(&result.result);
	Job::toHex(std::string(resultChar, 32), data_buffer);
	data_buffer[64] = '\0';

	const std::string nonce = nonce_buffer;
	const std::string data  = data_buffer;
#endif

	const size_t size = snprintf(m_sendBuf, sizeof(m_sendBuf),
	                             "{\"id\":%" PRIu64
	                             ",\"jsonrpc\":\"2.0\",\"method\":\"submit\",\"params\":{\"id\":\"%s\",\"job_id\":\"%s\",\"nonce\":\"%s\",\"result\":\"%s\"}}\n",
	                             m_sequence, m_rpcId,
	                             result.jobId.data().c_str(), nonce.c_str(), data.c_str());

	m_results[m_sequence] = SubmitResult(m_sequence, result.diff, result.actualDiff());
	return send(size);
}


bool Client::isCriticalError(const std::string & message)
{
	if(message.empty())
	{
		return false;
	}

	if(message == "Unauthenticated")
	{
		return true;
	}

	if(message == "your IP is banned")
	{
		return true;
	}

	if(message == "IP Address currently banned")
	{
		return true;
	}

	return false;
}


bool Client::parseJob(const rapidjson::Value & params, int* code)
{
	if(!params.IsObject())
	{
		*code = 2;
		return false;
	}

	Job job(m_id, m_url.isNicehash());
	if(!job.setId(params["job_id"].GetString()))
	{
		*code = 3;
		return false;
	}

	if(!job.setBlob(params["blob"].GetString()))
	{
		*code = 4;
		return false;
	}

	if(!job.setTarget(params["target"].GetString()))
	{
		*code = 5;
		return false;
	}

	if(m_job == job)
	{
		if(!m_quiet)
		{
			LOG_WARN("[" << m_url.host() << ":" << m_url.port() << "] duplicate job received, reconnect");
		}

		close();
		return false;
	}

	m_job = std::move(job);
	return true;
}


bool Client::parseLogin(const rapidjson::Value & result, int* code)
{
	const char* id = result["id"].GetString();
	if(!id || strlen(id) >= sizeof(m_rpcId))
	{
		*code = 1;
		return false;
	}

	memset(m_rpcId, 0, sizeof(m_rpcId));
	memcpy(m_rpcId, id, strlen(id));

	return parseJob(result["job"], code);
}


int Client::resolve(const std::string & host)
{
	setState(HostLookupState);

	m_expire     = 0;
	m_recvBufPos = 0;

	if(m_failures == -1)
	{
		m_failures = 0;
	}

	const int r = uv_getaddrinfo(uv_default_loop(), &m_resolver, Client::onResolved, host.c_str(), NULL, &m_hints);
	if(r)
	{
		if(!m_quiet)
		{
			LOG_ERR("[" << host << ":" << m_url.port() << "] getaddrinfo error: \"" << uv_strerror(r) << "\"");
		}
		return 1;
	}

	return 0;
}


int64_t Client::send(size_t size, const bool encrypted)
{
	LOG_DEBUG("[" << m_url.host() << ":" << m_url.port() << "] send(" << size << " bytes): \"" << m_sendBuf <<
	          "\"");
	if((state() != ConnectedState && state() != ProxingState) || !uv_is_writable(m_stream))
	{
		LOG_DEBUG_ERR("[" << m_url.host() << ":" << m_url.port() << "] send failed, invalid state: " << m_state);
		return -1;
	}

	if(encrypted && m_encrypted)
	{
		// Encrypt
		for(size_t i = 0; i < std::min(size, sizeof(SendBuf)); ++i)
		{
			m_sendBuf[i] ^= m_keystream[i];
		}

		char* send_encr_hex = static_cast<char*>(malloc(size * 2 + 1));
		memset(send_encr_hex, 0, size * 2 + 1);
		Job::toHex(std::string(m_sendBuf, size), send_encr_hex);
		send_encr_hex[size * 2] = '\0';
		LOG_DEBUG("[" << m_url.host() << ":" << m_url.port() << "] send encr.(" << size << " bytes): \"0x"  <<
		          send_encr_hex << "\"");
		free(send_encr_hex);
	}

	uv_buf_t buf = uv_buf_init(m_sendBuf, (unsigned int) size);

	if(uv_try_write(m_stream, &buf, 1) < 0)
	{
		close();
		return -1;
	}

	m_expire = uv_now(uv_default_loop()) + kResponseTimeout;
	return m_sequence++;
}


void Client::close()
{
	if(m_state == UnconnectedState || m_state == ClosingState || !m_socket)
	{
		return;
	}

	setState(ClosingState);

	if(uv_is_closing(reinterpret_cast<uv_handle_t*>(m_socket)) == 0)
	{
		uv_close(reinterpret_cast<uv_handle_t*>(m_socket), Client::onClose);
	}
}


void Client::connect(struct sockaddr* addr)
{
	setState(ConnectingState);

	reinterpret_cast<struct sockaddr_in*>(addr)->sin_port = htons(m_url.port());
	delete m_socket;

	uv_connect_t* req = new uv_connect_t;
	req->data = this;

	m_socket = new uv_tcp_t;
	m_socket->data = this;

	uv_tcp_init(uv_default_loop(), m_socket);
	uv_tcp_nodelay(m_socket, 1);

#ifndef WIN32
	uv_tcp_keepalive(m_socket, 1, 60);
#endif

	uv_tcp_connect(req, m_socket, reinterpret_cast<const sockaddr*>(addr), Client::onConnect);
}

void Client::prelogin()
{
	if(m_url.isProxyed())
	{
		setState(ProxingState);
		const std::string buffer = std::string("CONNECT ") + m_url.finalHost() + ":" +
		                           std::to_string((unsigned long long)(m_url.finalPort())) + " HTTP/1.1\n";

		const size_t size = buffer.size();
		memcpy(m_sendBuf, buffer.c_str(), size);
		m_sendBuf[size]     = '\n';
		m_sendBuf[size + 1] = '\0';

		LOG_DEBUG("Prelogin send (" << size << " bytes): \"" << m_sendBuf << "\"");
		send(size + 1, false);
	}
	else
	{
		setState(ConnectedState);
		login();
	}
}

void Client::login()
{
	m_results.clear();

	rapidjson::Document doc;
	doc.SetObject();

	auto & allocator = doc.GetAllocator();

	doc.AddMember("id",      1,       allocator);
	doc.AddMember("jsonrpc", "2.0",   allocator);
	doc.AddMember("method",  "login", allocator);

	rapidjson::Value params(rapidjson::kObjectType);
	params.AddMember("login", rapidjson::StringRef(m_url.user().c_str()),     allocator);
	params.AddMember("pass",  rapidjson::StringRef(m_url.password().c_str()), allocator);
	params.AddMember("agent", rapidjson::StringRef(m_agent.c_str()),	      allocator);

	doc.AddMember("params", params, allocator);

	rapidjson::StringBuffer buffer(0, 512);
	rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
	doc.Accept(writer);

	const size_t size = buffer.GetSize();
	if(size > (sizeof(m_buf) - 2))
	{
		return;
	}

	memcpy(m_sendBuf, buffer.GetString(), size);
	m_sendBuf[size]     = '\n';
	m_sendBuf[size + 1] = '\0';

	send(size + 1);
}


void Client::parse(char* line, size_t len)
{
	startTimeout();

	line[len - 1] = '\0';

	LOG_DEBUG("[" << m_url.host() << ":" << m_url.port() << "] received (" << len << " bytes): \"" << line <<
	          "\"");

	rapidjson::Document doc;
	if(doc.ParseInsitu(line).HasParseError())
	{
		if(!m_quiet)
		{
			LOG_ERR("[" << m_url.host() << ":" << m_url.port() << "] JSON decode failed: \"" <<
			        rapidjson::GetParseError_En(doc.GetParseError()) << "\"");
		}

		return;
	}

	if(!doc.IsObject())
	{
		return;
	}

	const rapidjson::Value & id = doc["id"];
	if(id.IsInt64())
	{
		parseResponse(id.GetInt64(), doc["result"], doc["error"]);
	}
	else
	{
		parseNotification(doc["method"].GetString(), doc["params"], doc["error"]);
	}
}


void Client::parseNotification(const char* method, const rapidjson::Value & params,
                               const rapidjson::Value & error)
{
	if(error.IsObject())
	{
		if(!m_quiet)
		{
			LOG_ERR("[" << m_url.host() << ":" << m_url.port() << "] error: \"" << error["message"].GetString() <<
			        "\", code: " << error["code"].GetInt());
		}
		return;
	}

	if(!method)
	{
		return;
	}

	if(strcmp(method, "job") == 0)
	{
		int code = -1;
		if(parseJob(params, &code))
		{
			m_listener->onJobReceived(this, m_job);
		}

		return;
	}

	LOG_WARN("[" << m_url.host() << ":" << m_url.port() << "] unsupported method: \"" << method << "\"");
}


void Client::parseResponse(int64_t id, const rapidjson::Value & result, const rapidjson::Value & error)
{
	if(error.IsObject())
	{
		const char* message = error["message"].GetString();

		auto it = m_results.find(id);
		if(it != m_results.end())
		{
			it->second.done();
			m_listener->onResultAccepted(this, it->second, message);
			m_results.erase(it);
		}
		else if(!m_quiet)
		{
			LOG_ERR("[" << m_url.host() << ":" << m_url.port() << "] error: \"" << message << "\", code: " <<
			        error["code"].GetInt());
		}

		if(id == 1 || isCriticalError(message))
		{
			close();
		}

		return;
	}

	if(!result.IsObject())
	{
		return;
	}

	if(id == 1)
	{
		int code = -1;
		if(!parseLogin(result, &code))
		{
			if(!m_quiet)
			{
				LOG_ERR("[" << m_url.host() << ":" << m_url.port() << "] login error code: " << code);
			}

			return close();
		}

		m_failures = 0;
		m_listener->onLoginSuccess(this);
		m_listener->onJobReceived(this, m_job);
		return;
	}

	auto it = m_results.find(id);
	if(it != m_results.end())
	{
		it->second.done();
		m_listener->onResultAccepted(this, it->second, "");
		m_results.erase(it);
	}
}


void Client::ping()
{
	send(snprintf(m_sendBuf, sizeof(m_sendBuf),
	              "{\"id\":%" PRId64 ",\"jsonrpc\":\"2.0\",\"method\":\"keepalived\",\"params\":{\"id\":\"%s\"}}\n", m_sequence,
	              m_rpcId));
}


void Client::reconnect()
{
	setState(ConnectingState);

#ifndef XMRIG_PROXY_PROJECT
	if(m_url.isKeepAlive())
	{
		uv_timer_stop(&m_keepAliveTimer);
	}
#endif

	if(m_failures == -1)
	{
		return m_listener->onClose(this, -1);
	}

	m_failures++;
	m_listener->onClose(this, (int) m_failures);

	m_expire = uv_now(uv_default_loop()) + m_retryPause;
}


void Client::setState(SocketState state)
{
	LOG_DEBUG("[" << m_url.host() << ":" << m_url.port() << "] state: " << state);

	if(m_state == state)
	{
		return;
	}

	m_state = state;
}


void Client::startTimeout()
{
	m_expire = 0;

#ifndef XMRIG_PROXY_PROJECT
	if(!m_url.isKeepAlive())
	{
		return;
	}

	uv_timer_start(&m_keepAliveTimer, &Client::onTimeout, kKeepAliveTimeout, 0);
#endif
}

void Client::onTimeout(uv_timer_t* handle)
{
	getClient(handle->data)->ping();
}

void Client::onAllocBuffer(uv_handle_t* handle, size_t suggested_size, uv_buf_t* buf)
{
	auto client = getClient(handle->data);

	buf->base = &client->m_recvBuf.base[client->m_recvBufPos];
	buf->len  = client->m_recvBuf.len - (unsigned long)client->m_recvBufPos;
}


void Client::onClose(uv_handle_t* handle)
{
	auto client = getClient(handle->data);

	delete client->m_socket;

	client->m_stream = nullptr;
	client->m_socket = nullptr;
	client->setState(UnconnectedState);

	client->reconnect();
}


void Client::onConnect(uv_connect_t* req, int status)
{
	auto client = getClient(req->data);
	if(status < 0)
	{
		if(!client->m_quiet)
		{
			LOG_ERR("[" << client->m_url.host() << ":" << client->m_url.port() << "] connect error: \"" << uv_strerror(
			            status) << "\"");
		}

		delete req;
		client->close();
		return;
	}

	client->m_stream = static_cast<uv_stream_t*>(req->handle);
	client->m_stream->data = req->data;

	uv_read_start(client->m_stream, Client::onAllocBuffer, Client::onRead);
	delete req;

	client->prelogin();
}


void Client::onRead(uv_stream_t* stream, ssize_t nread, const uv_buf_t* buf)
{
	auto client = getClient(stream->data);
	if(nread < 0)
	{
		if(nread != UV_EOF && !client->m_quiet)
		{
			LOG_ERR("[" << client->m_url.host() << ":" << client->m_url.port() << "] read error: \"" << uv_strerror((
			            int) nread) << "\"");
		}

		return client->close();
	}

	if((size_t) nread > (sizeof(Buf) - 8 - client->m_recvBufPos))
	{
		return client->close();
	}

	if(client->state() == ProxingState)
	{
		const char* const content = buf->base;
		LOG_DEBUG("[" << client->m_url.host() << ":" << client->m_url.port() << "] received from proxy (" << nread <<
		          " bytes): \"" << content << "\"");

		if(content == strstr(content, "HTTP/1.1 200"))
		{
			LOG_INFO("[" << client->m_url.host() << ":" << client->m_url.port() << "] Proxy connected to " <<
			         client->m_url.finalHost() << ":" << client->m_url.finalPort() << "!");
			client->setState(ConnectedState);
			client->login();
		}
		return;
	}

	client->m_recvBufPos += nread;

	char* end;
	char* start = buf->base;
	size_t remaining = client->m_recvBufPos;

	if(client->m_encrypted)
	{
		char* read_encr_hex = static_cast<char*>(malloc(nread * 2 + 1));
		memset(read_encr_hex, 0, nread * 2 + 1);
		Job::toHex(std::string(start, nread), read_encr_hex);
		LOG_DEBUG("[" <<  client->m_ip << "] read encr. (" << nread << "  bytes): 0x\"" << read_encr_hex << "\"");
		free(read_encr_hex);

		// DeEncrypt
		for(int i = 0; i < (int)nread; ++i)
		{
			start[i] ^= client->m_keystream[i];
		}
	}

	while((end = static_cast<char*>(memchr(start, '\n', remaining))) != nullptr)
	{
		end++;
		size_t len = end - start;
		client->parse(start, len);

		remaining -= len;
		start = end;
	}

	if(remaining == 0)
	{
		client->m_recvBufPos = 0;
		return;
	}

	if(start == buf->base)
	{
		return;
	}

	memcpy(buf->base, start, remaining);
	client->m_recvBufPos = remaining;
}


void Client::onResolved(uv_getaddrinfo_t* req, int status, struct addrinfo* res)
{
	auto client = getClient(req->data);
	if(status < 0)
	{
		LOG_ERR("[" << client->m_url.host() << ":" << client->m_url.port() << "] DNS error: \"" << uv_strerror(
		            status) << "\"");
		return client->reconnect();
	}

	addrinfo* ptr = res;
	std::vector<addrinfo*> ipv4;

	while(ptr != nullptr)
	{
		if(ptr->ai_family == AF_INET)
		{
			ipv4.push_back(ptr);
		}

		ptr = ptr->ai_next;
	}

	if(ipv4.empty())
	{
		LOG_ERR("[" << client->m_url.host() << ":" << client->m_url.port() <<
		        "] DNS error: \"No IPv4 records found\"");
		return client->reconnect();
	}

	ptr = ipv4[rand() % ipv4.size()];

	char buf[16];
	uv_ip4_name(reinterpret_cast<sockaddr_in*>(ptr->ai_addr), buf, sizeof(buf));
	client->setIP(buf);

	client->connect(ptr->ai_addr);
	uv_freeaddrinfo(res);
}
