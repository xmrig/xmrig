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
#include "net/JobResult.h"
#ifdef _MSC_VER
#   define strncasecmp(x,y,z) _strnicmp(x,y,z)
#endif
int64_t Client::m_sequence = 1;
Client::Client(int id, const char *agent, IClientListener *listener) :
    m_quiet(false),
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
    memset(m_ip, 0, sizeof(m_ip));
    memset(&m_hints, 0, sizeof(m_hints));
    m_resolver.data = this;
    m_hints.ai_family   = PF_INET;
    m_hints.ai_socktype = SOCK_STREAM;
    m_hints.ai_protocol = IPPROTO_TCP;
    m_recvBuf.base = m_buf;
    m_recvBuf.len  = sizeof(m_buf);
#   ifndef XMRIG_PROXY_PROJECT
    m_keepAliveTimer.data = this;
    uv_timer_init(uv_default_loop(), &m_keepAliveTimer);
#   endif
}
Client::~Client() { delete m_socket; }
void Client::connect() { resolve(m_url.host()); }
void Client::connect(const Url *url)
{
    setUrl(url);
    resolve(m_url.host());
}
void Client::disconnect()
{
    m_expire   = 0;
    m_failures = -1;
    close();
}
void Client::setUrl(const Url *url)
{
    if (!url || !url->isValid()) { return; }
    m_url = url;
}
void Client::tick(uint64_t now)
{
    if (m_expire == 0 || now < m_expire) { return; }
    if (m_state == ConnectedState) { close(); }
    if (m_state == ConnectingState) { connect(); }
}
int64_t Client::submit(const JobResult &result)
{
    char nonce[9];
    char data[65];
    Job::toHex(reinterpret_cast<const unsigned char*>(&result.nonce), 4, nonce);
    nonce[8] = '\0';
    Job::toHex(result.result, 32, data);
    data[64] = '\0';
    const size_t size = snprintf(m_sendBuf, sizeof(m_sendBuf), "{\"id\":%" PRIu64 ",\"jsonrpc\":\"2.0\",\"method\":\"submit\",\"params\":{\"id\":\"%s\",\"job_id\":\"%s\",\"nonce\":\"%s\",\"result\":\"%s\"}}\n",
                                 m_sequence, m_rpcId, result.jobId.data(), nonce, data);
    m_results[m_sequence] = SubmitResult(m_sequence, result.diff, result.actualDiff());
    return send(size);
}
bool Client::parseJob(const rapidjson::Value &params, int *code)
{
    if (!params.IsObject()) {
        *code = 2;
        return false;
    }
    Job job(m_id, m_url.isNicehash());
    if (!job.setId(params["job_id"].GetString())) {
        *code = 3;
        return false;
    }
    if (!job.setBlob(params["blob"].GetString())) {
        *code = 4;
        return false;
    }
    if (!job.setTarget(params["target"].GetString())) {
        *code = 5;
        return false;
    }
    if (m_job == job) {
        if (!m_quiet) {}
        close();
        return false;
    }
    m_job = std::move(job);
    return true;
}
bool Client::parseLogin(const rapidjson::Value &result, int *code)
{
    const char *id = result["id"].GetString();
    if (!id || strlen(id) >= sizeof(m_rpcId)) {
        *code = 1;
        return false;
    }
    memset(m_rpcId, 0, sizeof(m_rpcId));
    memcpy(m_rpcId, id, strlen(id));
    return parseJob(result["job"], code);
}
int Client::resolve(const char *host)
{
    setState(HostLookupState);
    m_expire     = 0;
    m_recvBufPos = 0;
    if (m_failures == -1) {
        m_failures = 0;
    }
    const int r = uv_getaddrinfo(uv_default_loop(), &m_resolver, Client::onResolved, host, NULL, &m_hints);
    if (r) {
        if (!m_quiet) {}
        return 1;
    }
   return 0;
}
int64_t Client::send(size_t size)
{
    if (state() != ConnectedState || !uv_is_writable(m_stream)) {
        return -1;
    }
    uv_buf_t buf = uv_buf_init(m_sendBuf, (unsigned int) size);
    if (uv_try_write(m_stream, &buf, 1) < 0) {
        close();
        return -1;
    }
   m_expire = uv_now(uv_default_loop()) + kResponseTimeout;
    return m_sequence++;
}
void Client::close()
{
    if (m_state == UnconnectedState || m_state == ClosingState || !m_socket) {
        return;
    }
    setState(ClosingState);
    if (uv_is_closing(reinterpret_cast<uv_handle_t*>(m_socket)) == 0) {
        uv_close(reinterpret_cast<uv_handle_t*>(m_socket), Client::onClose);
    }
}
void Client::connect(struct sockaddr *addr)
{
    setState(ConnectingState);
    reinterpret_cast<struct sockaddr_in*>(addr)->sin_port = htons(m_url.port());
    delete m_socket;
    uv_connect_t *req = new uv_connect_t;
    req->data = this;
    m_socket = new uv_tcp_t;
    m_socket->data = this;
    uv_tcp_init(uv_default_loop(), m_socket);
    uv_tcp_nodelay(m_socket, 1);
#   ifndef WIN32
    uv_tcp_keepalive(m_socket, 1, 60);
#   endif
    uv_tcp_connect(req, m_socket, reinterpret_cast<const sockaddr*>(addr), Client::onConnect);
}
void Client::login()
{
    m_results.clear();
    rapidjson::Document doc;
    doc.SetObject();
    auto &allocator = doc.GetAllocator();
    doc.AddMember("id",      1,       allocator);
    doc.AddMember("jsonrpc", "2.0",   allocator);
    doc.AddMember("method",  "login", allocator);
    rapidjson::Value params(rapidjson::kObjectType);
    params.AddMember("login", rapidjson::StringRef(m_url.user()),     allocator);
    params.AddMember("pass",  rapidjson::StringRef(m_url.password()), allocator);
    params.AddMember("agent", rapidjson::StringRef(m_agent),          allocator);
    doc.AddMember("params", params, allocator);
    rapidjson::StringBuffer buffer(0, 512);
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    doc.Accept(writer);
    const size_t size = buffer.GetSize();
    if (size > (sizeof(m_buf) - 2)) {
        return;
    }
    memcpy(m_sendBuf, buffer.GetString(), size);
    m_sendBuf[size]     = '\n';
    m_sendBuf[size + 1] = '\0';
    send(size + 1);
}
void Client::parse(char *line, size_t len)
{
    startTimeout();
    line[len - 1] = '\0';
    rapidjson::Document doc;
    if (doc.ParseInsitu(line).HasParseError()) {
        if (!m_quiet) {}
        return;
    }
    if (!doc.IsObject()) {
        return;
    }
    const rapidjson::Value &id = doc["id"];
    if (id.IsInt64()) {
        parseResponse(id.GetInt64(), doc["result"], doc["error"]);
    }
    else {
        parseNotification(doc["method"].GetString(), doc["params"], doc["error"]);
    }
}
void Client::parseNotification(const char *method, const rapidjson::Value &params, const rapidjson::Value &error)
{
    if (error.IsObject()) {
        if (!m_quiet) {}
        return;
    }
    if (!method) {
        return;
    }
    if (strcmp(method, "job") == 0) {
        int code = -1;
        if (parseJob(params, &code)) {
            m_listener->onJobReceived(this, m_job);
        }
        return;
    }
}
void Client::parseResponse(int64_t id, const rapidjson::Value &result, const rapidjson::Value &error)
{
    if (error.IsObject()) {
        const char *message = error["message"].GetString();
		auto it = m_results.find(id);
        if (it != m_results.end()) {
            it->second.done();
            m_listener->onResultAccepted(this, it->second, message);
            m_results.erase(it);
        }
        else if (!m_quiet) {}
        if (id == 1) { close(); }
        return;
    }

    if (!result.IsObject()) { return; }
    if (id == 1) {
        int code = -1;
        if (!parseLogin(result, &code)) {
            if (!m_quiet) {}
            return close();
        }
        m_failures = 0;
        m_listener->onLoginSuccess(this);
        m_listener->onJobReceived(this, m_job);
        return;
    }
    auto it = m_results.find(id);
    if (it != m_results.end()) {
        it->second.done();
        m_listener->onResultAccepted(this, it->second, nullptr);
        m_results.erase(it);
    }
}
void Client::ping()
{
    send(snprintf(m_sendBuf, sizeof(m_sendBuf), "{\"id\":%" PRId64 ",\"jsonrpc\":\"2.0\",\"method\":\"keepalived\",\"params\":{\"id\":\"%s\"}}\n", m_sequence, m_rpcId));
}
void Client::reconnect()
{
    setState(ConnectingState);
#   ifndef XMRIG_PROXY_PROJECT
    if (m_url.isKeepAlive()) {
        uv_timer_stop(&m_keepAliveTimer);
    }
#   endif
    if (m_failures == -1) { return m_listener->onClose(this, -1); }
    m_failures++;
    m_listener->onClose(this, (int) m_failures);
   m_expire = uv_now(uv_default_loop()) + m_retryPause;
}
void Client::setState(SocketState state)
{
    if (m_state == state) { return; }
    m_state = state;
}
void Client::startTimeout()
{
    m_expire = 0;
#   ifndef XMRIG_PROXY_PROJECT
    if (!m_url.isKeepAlive()) { return; }
    uv_timer_start(&m_keepAliveTimer, [](uv_timer_t *handle) { getClient(handle->data)->ping(); }, kKeepAliveTimeout, 0);
#   endif
}
void Client::onAllocBuffer(uv_handle_t *handle, size_t suggested_size, uv_buf_t *buf)
{
    auto client = getClient(handle->data);
    buf->base = &client->m_recvBuf.base[client->m_recvBufPos];
	buf->len = client->m_recvBuf.len - (ULONG)client->m_recvBufPos;
}
void Client::onClose(uv_handle_t *handle)
{
    auto client = getClient(handle->data);
    delete client->m_socket;
    client->m_stream = nullptr;
    client->m_socket = nullptr;
    client->setState(UnconnectedState);
    client->reconnect();
}
void Client::onConnect(uv_connect_t *req, int status)
{
    auto client = getClient(req->data);
    if (status < 0) {
        if (!client->m_quiet) {}
        delete req;
        client->close();
        return;
    }
    client->m_stream = static_cast<uv_stream_t*>(req->handle);
    client->m_stream->data = req->data;
    client->setState(ConnectedState);
    uv_read_start(client->m_stream, Client::onAllocBuffer, Client::onRead);
    delete req;
    client->login();
}
void Client::onRead(uv_stream_t *stream, ssize_t nread, const uv_buf_t *buf)
{
    auto client = getClient(stream->data);
    if (nread < 0) {
        if (nread != UV_EOF && !client->m_quiet) {}
        return client->close();;
    }
    if ((size_t) nread > (sizeof(m_buf) - 8 - client->m_recvBufPos)) { return client->close();; }
    client->m_recvBufPos += nread;
    char* end;
    char* start = buf->base;
    size_t remaining = client->m_recvBufPos;
    while ((end = static_cast<char*>(memchr(start, '\n', remaining))) != nullptr) {
        end++;
        size_t len = end - start;
        client->parse(start, len);
        remaining -= len;
        start = end;
    }
    if (remaining == 0) {
        client->m_recvBufPos = 0;
        return;
    }
    if (start == buf->base) { return; }
    memcpy(buf->base, start, remaining);
    client->m_recvBufPos = remaining;
}
void Client::onResolved(uv_getaddrinfo_t *req, int status, struct addrinfo *res)
{
    auto client = getClient(req->data);
    if (status < 0) { return client->reconnect();; }
    addrinfo *ptr = res;
    std::vector<addrinfo*> ipv4;
    while (ptr != nullptr) {
        if (ptr->ai_family == AF_INET) { ipv4.push_back(ptr); }
        ptr = ptr->ai_next;
    }
    if (ipv4.empty()) { return client->reconnect(); }
    ptr = ipv4[rand() % ipv4.size()];
    uv_ip4_name(reinterpret_cast<sockaddr_in*>(ptr->ai_addr), client->m_ip, 16);
    client->connect(ptr->ai_addr);
    uv_freeaddrinfo(res);
}
