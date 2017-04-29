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
#include <pthread.h>
#include <jansson.h>
#include <unistd.h>

#if defined(WIN32)
#   include <winsock2.h>
#   include <mstcpip.h>
#else
#   include <errno.h>
#   include <netinet/tcp.h>
#   include <poll.h>
#endif

#include "stratum.h"
#include "version.h"
#include "stats.h"
#include "util.h"
#include "utils/applog.h"


#ifdef WIN32
#   define socket_blocks() (WSAGetLastError() == WSAEWOULDBLOCK)
#   define poll(fdarray, nfds, timeout) WSAPoll(fdarray, nfds, timeout)
#   define SHUT_RDWR SD_BOTH
#else
#   define socket_blocks() (errno == EAGAIN || errno == EWOULDBLOCK)
#   define closesocket(x) close((x))
#endif

#define RBUFSIZE 2048
#define RECVSIZE (RBUFSIZE - 4)

#define unlikely(expr) (__builtin_expect(!!(expr), 0))


static struct work work;


static bool send_line(curl_socket_t sock, char *s);
static bool socket_full(curl_socket_t sock, int timeout);
static void buffer_append(struct stratum_ctx *sctx, const char *s);
static bool job(struct stratum_ctx *sctx, json_t *params);
static int sockopt_keepalive_cb(void *userdata, curl_socket_t fd, curlsocktype purpose);
static curl_socket_t opensocket_grab_cb(void *clientp, curlsocktype purpose, struct curl_sockaddr *addr);
static int closesocket_cb(void *clientp, curl_socket_t item);
static bool login_decode(struct stratum_ctx *sctx, const json_t *val);
static bool job_decode(const json_t *job);
static bool jobj_binary(const json_t *obj, const char *key, void *buf, size_t buflen);


/**
 * @brief stratum_socket_full
 * @param sctx
 * @param timeout
 * @return
 */
bool stratum_socket_full(struct stratum_ctx *sctx, int timeout)
{
    return strlen(sctx->sockbuf) || socket_full(sctx->sock, timeout);
}


/**
 * @brief stratum_send_line
 * @param sctx
 * @param s
 * @return
 */
bool stratum_send_line(struct stratum_ctx *sctx, char *s)
{
    bool ret = false;

    pthread_mutex_lock(&sctx->sock_lock);
    ret = send_line(sctx->sock, s);
    pthread_mutex_unlock(&sctx->sock_lock);

    return ret;
}


/**
 * @brief stratum_recv_line
 * @param sctx
 * @return
 */
char *stratum_recv_line(struct stratum_ctx *sctx)
{
    if (!strstr(sctx->sockbuf, "\n")) {
        bool ret = true;
        time_t rstart;

        time(&rstart);

        if (!socket_full(sctx->sock, 60)) {
            applog(LOG_ERR, "stratum_recv_line timed out");
            return NULL;
        }

        do {
            char s[RBUFSIZE];
            ssize_t n;

            memset(s, 0, RBUFSIZE);
            n = recv(sctx->sock, s, RECVSIZE, 0);
            if (!n) {
                ret = false;
                break;
            }

            if (n < 0) {
                if (!socket_blocks() || !socket_full(sctx->sock, 1)) {
                    ret = false;
                    break;
                }
            } else {
                buffer_append(sctx, s);
            }
        } while (time(NULL) - rstart < 60 && !strstr(sctx->sockbuf, "\n"));

        if (!ret) {
            applog(LOG_ERR, "stratum_recv_line failed");
            return NULL;
        }
    }

    ssize_t buflen = strlen(sctx->sockbuf);
    char *tok = strtok(sctx->sockbuf, "\n");

    if (!tok) {
        applog(LOG_ERR, "stratum_recv_line failed to parse a newline-terminated string");
        return NULL;
    }

    char *sret = strdup(tok);
    ssize_t len = strlen(sret);

    if (buflen > len + 1) {
        memmove(sctx->sockbuf, sctx->sockbuf + len + 1, buflen - len + 1);
    }
    else {
        sctx->sockbuf[0] = '\0';
    }

    return sret;
}


/**
 * @brief stratum_disconnect
 * @param sctx
 */
void stratum_disconnect(struct stratum_ctx *sctx)
{
    pthread_mutex_lock(&sctx->sock_lock);

    sctx->ready = false;

    if (sctx->curl) {
        curl_easy_cleanup(sctx->curl);
        sctx->curl = NULL;
        sctx->sockbuf[0] = '\0';
    }

    pthread_mutex_unlock(&sctx->sock_lock);
}


/**
 * @brief stratum_handle_method
 * @param sctx
 * @param s
 * @return
 */
bool stratum_handle_method(struct stratum_ctx *sctx, const char *s)
{
    bool ret = false;
    const char *method;
    json_t *val = json_decode(s);

    if (!val) {
        return false;
    }

    if (method = json_string_value(json_object_get(val, "method"))) {
        if (!strcasecmp(method, "job")) {
            ret = job(sctx, json_object_get(val, "params"));
        }
        else {
            applog(LOG_WARNING, "Unknown method: \"%s\"", method);
        }
    }

    json_decref(val);
    return ret;
}


/**
 * @brief stratum_handle_response
 * @param buf
 * @return
 */
bool stratum_handle_response(char *buf) {
    bool valid = false;

    json_t *val = json_decode(buf);
    if (!val) {
       return false;
    }

    json_t *res_val = json_object_get(val, "result");
    json_t *err_val = json_object_get(val, "error");
    json_t *id_val = json_object_get(val, "id");

    if (!id_val || json_is_null(id_val) || !res_val) {
       json_decref(val);
       return false;
    }

    json_t *status = json_object_get(res_val, "status");

    if (!strcmp(json_string_value(status), "KEEPALIVED") ) {
        applog(LOG_DEBUG, "Keepalived receveid");
        json_decref(val);
        return true;
    }

    if (status) {
       valid = !strcmp(json_string_value(status), "OK") && json_is_null(err_val);
    } else {
       valid = json_is_null(err_val);
    }

    stats_share_result(valid);
    json_decref(val);
    return true;
}


/**
 * @brief stratum_keepalived
 * @param sctx
 * @return
 */
bool stratum_keepalived(struct stratum_ctx *sctx)
{
    char *s = malloc(128);
    snprintf(s, 128, "{\"method\":\"keepalived\",\"params\":{\"id\":\"%s\"},\"id\":1}", sctx->id);
    bool ret = stratum_send_line(sctx, s);

    free(s);
    return ret;
}


/**
 * @brief stratum_authorize
 * @param sctx
 * @param user
 * @param pass
 * @return
 */
bool stratum_authorize(struct stratum_ctx *sctx, const char *user, const char *pass)
{
    char *sret;
    json_error_t err;

    char *req = malloc(128 + strlen(user) + strlen(pass));
    sprintf(req, "{\"method\":\"login\",\"params\":{\"login\":\"%s\",\"pass\":\"%s\",\"agent\":\"%s/%s\"},\"id\":1}", user, pass, APP_NAME, APP_VERSION);

    if (!stratum_send_line(sctx, req)) {
        free(req);
        return false;
    }

    free(req);

    while (1) {
        sret = stratum_recv_line(sctx);
        if (!sret) {
            return false;
        }

        if (!stratum_handle_method(sctx, sret)) {
            break;
        }

        free(sret);
    }

    json_t *val = json_decode(sret);
    free(sret);

    if (!val) {
        return false;
    }

    json_t *result = json_object_get(val, "result");
    json_t *error  = json_object_get(val, "error");

    if (!result || json_is_false(result) || (error && !json_is_null(error)))  {
        applog(LOG_ERR, "Stratum authentication failed");
        json_decref(val);
        return false;
    }

    if (login_decode(sctx, val) && job(sctx, json_object_get(result, "job"))) {
        pthread_mutex_lock(&sctx->sock_lock);
        sctx->ready = true;
        pthread_mutex_unlock(&sctx->sock_lock);
    }

    json_decref(val);
    return true;
}


/**
 * @brief stratum_connect
 * @param sctx
 * @param url
 * @return
 */
bool stratum_connect(struct stratum_ctx *sctx, const char *url)
{
    CURL *curl;

    pthread_mutex_lock(&sctx->sock_lock);
    sctx->ready = false;

    if (sctx->curl) {
        curl_easy_cleanup(sctx->curl);
    }

    sctx->curl = curl_easy_init();
    if (!sctx->curl) {
        applog(LOG_ERR, "CURL initialization failed");
        pthread_mutex_unlock(&sctx->sock_lock);
        return false;
    }

    curl = sctx->curl;
    if (!sctx->sockbuf) {
        sctx->sockbuf = calloc(RBUFSIZE, 1);
        sctx->sockbuf_size = RBUFSIZE;
    }

    sctx->sockbuf[0] = '\0';
    pthread_mutex_unlock(&sctx->sock_lock);

    if (url != sctx->url) {
        free(sctx->url);
        sctx->url = strdup(url);
    }

    free(sctx->curl_url);
    sctx->curl_url = malloc(strlen(url));
    sprintf(sctx->curl_url, "http%s/", strstr(url, "://"));

    curl_easy_setopt(curl, CURLOPT_VERBOSE, 0);
    curl_easy_setopt(curl, CURLOPT_URL, sctx->curl_url);
    curl_easy_setopt(curl, CURLOPT_FRESH_CONNECT, 1);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 30);
    curl_easy_setopt(curl, CURLOPT_ERRORBUFFER, sctx->curl_err_str);
    curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1);
    curl_easy_setopt(curl, CURLOPT_TCP_NODELAY, 1);
    curl_easy_setopt(curl, CURLOPT_SOCKOPTFUNCTION, sockopt_keepalive_cb);
    curl_easy_setopt(curl, CURLOPT_OPENSOCKETFUNCTION, opensocket_grab_cb);
    curl_easy_setopt(curl, CURLOPT_CLOSESOCKETFUNCTION, closesocket_cb);
    curl_easy_setopt(curl, CURLOPT_OPENSOCKETDATA, &sctx->sock);
    curl_easy_setopt(curl, CURLOPT_CONNECT_ONLY, 1);
    curl_easy_setopt(curl, CURLOPT_IPRESOLVE, CURL_IPRESOLVE_V4);

    int rc = curl_easy_perform(curl);
    if (rc) {
        applog(LOG_ERR, "Stratum connection failed: code: %d, text: %s", rc, sctx->curl_err_str);
        curl_easy_cleanup(curl);
        sctx->curl = NULL;
        return false;
    }

    return true;
}


/**
 * @brief send_line
 * @param sock
 * @param s
 * @return
 */
static bool send_line(curl_socket_t sock, char *s)
{
    ssize_t len, sent = 0;

    len = strlen(s);
    s[len++] = '\n';

    while (len > 0) {
        struct pollfd pfd;
        pfd.fd = sock;
        pfd.events = POLLOUT;

        if (poll(&pfd, 1, 0) < 1) {
            return false;
        }

        ssize_t n = send(sock, s + sent, len, 0);
        if (n < 0) {
            if (!socket_blocks()) {
                return false;
            }

            n = 0;
        }

        sent += n;
        len -= n;
    }

    return true;
}


/**
 * @brief socket_full
 * @param sock
 * @param timeout
 * @return
 */
static bool socket_full(curl_socket_t sock, int timeout)
{
    struct pollfd pfd;
    pfd.fd = sock;
    pfd.events = POLLIN;

    return poll(&pfd, 1, timeout * 1000) > 0;
}


/**
 * @brief buffer_append
 * @param sctx
 * @param s
 */
static void buffer_append(struct stratum_ctx *sctx, const char *s)
{
    size_t old, new;

    old = strlen(sctx->sockbuf);
    new = old + strlen(s) + 1;

    if (new >= sctx->sockbuf_size) {
        sctx->sockbuf_size = new + (RBUFSIZE - (new % RBUFSIZE));
        sctx->sockbuf = realloc(sctx->sockbuf, sctx->sockbuf_size);
    }

    strcpy(sctx->sockbuf + old, s);
}


/**
 * @brief job
 * @param sctx
 * @param params
 * @return
 */
static bool job(struct stratum_ctx *sctx, json_t *params)
{
    if (!job_decode(params)) {
        return false;
    }

    pthread_mutex_lock(&sctx->work_lock);

    if (sctx->work.target != work.target) {
        stats_set_target(work.target);
    }

    memcpy(&sctx->work, &work, sizeof(struct work));
    pthread_mutex_unlock(&sctx->work_lock);

    return true;
}


/**
 * @brief sockopt_keepalive_cb
 * @param userdata
 * @param fd
 * @param purpose
 * @return
 */
static int sockopt_keepalive_cb(void *userdata, curl_socket_t fd, curlsocktype purpose)
{
    int keepalive = 1;
    int tcp_keepcnt = 3;
    int tcp_keepidle = 50;
    int tcp_keepintvl = 50;

#ifndef WIN32
    if (unlikely(setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &keepalive,
        sizeof(keepalive))))
        return 1;
#ifdef __linux
    if (unlikely(setsockopt(fd, SOL_TCP, TCP_KEEPCNT,
        &tcp_keepcnt, sizeof(tcp_keepcnt))))
        return 1;
    if (unlikely(setsockopt(fd, SOL_TCP, TCP_KEEPIDLE,
        &tcp_keepidle, sizeof(tcp_keepidle))))
        return 1;
    if (unlikely(setsockopt(fd, SOL_TCP, TCP_KEEPINTVL,
        &tcp_keepintvl, sizeof(tcp_keepintvl))))
        return 1;
#endif /* __linux */
#ifdef __APPLE_CC__
    if (unlikely(setsockopt(fd, IPPROTO_TCP, TCP_KEEPALIVE,
        &tcp_keepintvl, sizeof(tcp_keepintvl))))
        return 1;
#endif /* __APPLE_CC__ */
#else /* WIN32 */
    struct tcp_keepalive vals;
    vals.onoff = 1;
    vals.keepalivetime = tcp_keepidle * 1000;
    vals.keepaliveinterval = tcp_keepintvl * 1000;
    DWORD outputBytes;
    if (unlikely(WSAIoctl(fd, SIO_KEEPALIVE_VALS, &vals, sizeof(vals), NULL, 0, &outputBytes, NULL, NULL))) {
        return 1;
    }

#endif /* WIN32 */

    return 0;
}


static int closesocket_cb(void *clientp, curl_socket_t item) {
    shutdown(item, SHUT_RDWR);
    return closesocket(item);
}


/**
 * @brief opensocket_grab_cb
 * @param clientp
 * @param purpose
 * @param addr
 * @return
 */
static curl_socket_t opensocket_grab_cb(void *clientp, curlsocktype purpose, struct curl_sockaddr *addr)
{
    curl_socket_t *sock = clientp;
    *sock = socket(addr->family, addr->socktype, addr->protocol);
    return *sock;
}


/**
 * @brief login_decode
 * @param sctx
 * @param val
 * @return
 */
static bool login_decode(struct stratum_ctx *sctx, const json_t *val) {
    json_t *res = json_object_get(val, "result");
    if (!res) {
        applog(LOG_ERR, "JSON invalid result");
        return false;
    }

    const char *id = json_string_value(json_object_get(res, "id"));
    if (!id || strlen(id) >= (sizeof(sctx->id))) {
        applog(LOG_ERR, "JSON invalid id");
        return false;
    }

    memset(&sctx->id, 0, sizeof(sctx->id));
    memcpy(&sctx->id, id, strlen(id));

    const char *s = json_string_value(json_object_get(res, "status"));
    if (!s) {
        applog(LOG_ERR, "JSON invalid status");
        return false;
    }

    if (strcmp(s, "OK")) {
        applog(LOG_ERR, "JSON returned status \"%s\"", s);
        return false;
    }

    return true;
}


/**
 * @brief job_decode
 * @param sctx
 * @param job
 * @param work
 * @return
 */
static bool job_decode(const json_t *job) {
    const char *job_id = json_string_value(json_object_get(job, "job_id"));
    if (!job_id || strlen(job_id) >= sizeof(work.job_id)) {
        applog(LOG_ERR, "JSON invalid job id");
        return false;
    }

    const char *blob = json_string_value(json_object_get(job, "blob"));
    if (!blob) {
        applog(LOG_ERR, "JSON invalid blob");
        return false;
    }

    work.blob_size = strlen(blob);
    if (work.blob_size % 2 != 0) {
        applog(LOG_ERR, "JSON invalid blob length");
        return false;
    }

    work.blob_size /= 2;
    if (work.blob_size < 76 || work.blob_size > (sizeof(work.blob))) {
        applog(LOG_ERR, "JSON invalid blob length");
        return false;
    }

    if (!hex2bin((unsigned char *) work.blob, blob, work.blob_size)) {
        applog(LOG_ERR, "JSON invalid blob");
        return false;
    }

    jobj_binary(job, "target", &work.target, 4);

    memset(work.job_id, 0, sizeof(work.job_id));
    memcpy(work.job_id, job_id, strlen(job_id));

    return true;
}


/**
 * @brief jobj_binary
 * @param obj
 * @param key
 * @param buf
 * @param buflen
 * @return
 */
static bool jobj_binary(const json_t *obj, const char *key, void *buf, size_t buflen) {
    const char *hexstr;
    json_t *tmp;

    tmp = json_object_get(obj, key);
    if (unlikely(!tmp)) {
        applog(LOG_ERR, "JSON key '%s' not found", key);
        return false;
    }

    hexstr = json_string_value(tmp);
    if (unlikely(!hexstr)) {
        applog(LOG_ERR, "JSON key '%s' is not a string", key);
        return false;
    }


    if (!hex2bin(buf, hexstr, buflen)) {
        return false;
    }

    return true;
}
