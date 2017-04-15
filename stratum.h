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

#ifndef __STRATUM_H__
#define __STRATUM_H__

#include <stdbool.h>
#include <inttypes.h>
#include <curl/curl.h>


struct work {
    uint32_t data[19];
    uint32_t target[8];
    uint32_t hash[8];

    char *job_id;
    size_t xnonce2_len;
    unsigned char *xnonce2;
};


struct stratum_ctx {
    char *url;

    CURL *curl;
    char *curl_url;
    char curl_err_str[CURL_ERROR_SIZE];
    curl_socket_t sock;
    size_t sockbuf_size;
    char *sockbuf;
    pthread_mutex_t sock_lock;
    bool ready;

    char id[64];
    char blob[76];
    uint32_t target;

    struct work work;
    struct work g_work;
    time_t g_work_time;
    pthread_mutex_t work_lock;
};


bool stratum_send_line(struct stratum_ctx *sctx, char *s);
bool stratum_socket_full(struct stratum_ctx *sctx, int timeout);
char *stratum_recv_line(struct stratum_ctx *sctx);
bool stratum_connect(struct stratum_ctx *sctx, const char *url);
void stratum_disconnect(struct stratum_ctx *sctx);
bool stratum_authorize(struct stratum_ctx *sctx, const char *user, const char *pass);
bool stratum_handle_method(struct stratum_ctx *sctx, const char *s);
bool stratum_handle_response(char *buf);
bool stratum_keepalived(struct stratum_ctx *sctx);

#endif /* __STRATUM_H__ */
