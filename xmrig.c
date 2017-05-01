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
#include <unistd.h>
#include <jansson.h>
#include <sys/time.h>

#ifdef WIN32
#   include <winsock2.h>
#   include <windows.h>
#endif

#include <jansson.h>
#include <curl/curl.h>
#include <pthread.h>

#include "compat.h"
#include "xmrig.h"
#include "algo/cryptonight/cryptonight.h"
#include "options.h"
#include "cpu.h"
#include "persistent_memory.h"
#include "stratum.h"
#include "stats.h"
#include "util.h"
#include "utils/summary.h"
#include "utils/applog.h"

#define LP_SCANTIME  60
#define JSON_BUF_LEN 345


struct workio_cmd {
    struct thr_info *thr;
    struct work *work;
};


struct thr_info *thr_info;
static int work_thr_id = -1;
static int timer_thr_id = -1;
static int stratum_thr_id = -1;
struct work_restart *work_restart = NULL;
static struct stratum_ctx *stratum_ctx = NULL;
static bool backup_active = false;
static bool g_want_donate = false;


static void workio_cmd_free(struct workio_cmd *wc);


/**
 * @brief work_copy
 * @param dest
 * @param src
 */
static inline void work_copy(struct work *dest, const struct work *src) {
    memcpy(dest, src, sizeof(struct work));
}


/**
 * @brief restart_threads
 */
static inline void restart_threads(void) {
    for (int i = 0; i < opt_n_threads; i++) {
        work_restart[i].restart = 1;
    }
}


/**
 * @brief gen_workify
 * @param sctx
 * @param work
 */
static inline void gen_workify(struct stratum_ctx *sctx) {
    pthread_mutex_lock(&stratum_ctx->work_lock);

    if (stratum_ctx->work.job_id && (!stratum_ctx->g_work_time || strcmp(stratum_ctx->work.job_id, stratum_ctx->g_work.job_id))) {
        memcpy(&sctx->g_work, &sctx->work, sizeof(struct work));
        time(&stratum_ctx->g_work_time);

        pthread_mutex_unlock(&stratum_ctx->work_lock);

        applog(LOG_DEBUG, "Stratum detected new block");
        restart_threads();

        return;
    }

    pthread_mutex_unlock(&stratum_ctx->work_lock);
}


/**
 * @brief submit_upstream_work
 * @param work
 * @return
 */
static bool submit_upstream_work(struct work *work) {
    char s[JSON_BUF_LEN];

    /* pass if the previous hash is not the current previous hash */
    if (memcmp(work->blob + 1, stratum_ctx->g_work.blob + 1, 32)) {
        return true;
    }

    char *noncestr = bin2hex(((const unsigned char*) work->blob) + 39, 4);
    char *hashhex  = bin2hex((const unsigned char *) work->hash, 32);

    snprintf(s, JSON_BUF_LEN,
            "{\"method\":\"submit\",\"params\":{\"id\":\"%s\",\"job_id\":\"%s\",\"nonce\":\"%s\",\"result\":\"%s\"},\"id\":1}",
            stratum_ctx->id, work->job_id, noncestr, hashhex);

    free(hashhex);
    free(noncestr);

    if (unlikely(!stratum_send_line(stratum_ctx, s))) {
        return false;
    }

    return true;
}



/**
 * @brief workio_cmd_free
 * @param wc
 */
static void workio_cmd_free(struct workio_cmd *wc) {
    if (!wc) {
        return;
    }

    free(wc->work);

    memset(wc, 0, sizeof(*wc)); /* poison */
    free(wc);
}


/**
 * @brief workio_submit_work
 * @param wc
 * @param curl
 * @return
 */
static bool workio_submit_work(struct workio_cmd *wc) {
    while (!submit_upstream_work(wc->work)) {
        sleep(opt_retry_pause);
    }

    return true;
}


/**
 * @brief workio_thread
 * @param userdata
 * @return
 */
static void *workio_thread(void *userdata) {
    struct thr_info *mythr = userdata;
    bool ok = true;

    while (ok) {
        struct workio_cmd *wc;

        /* wait for workio_cmd sent to us, on our queue */
        wc = tq_pop(mythr->q, NULL );
        if (!wc) {
            ok = false;
            break;
        }

        workio_submit_work(wc);
        workio_cmd_free(wc);
    }

    tq_freeze(mythr->q);

    return NULL ;
}


/**
 * @brief submit_work
 * @param thr
 * @param work_in
 * @return
 */
static bool submit_work(struct thr_info *thr, const struct work *work_in) {
    struct workio_cmd *wc;

    /* fill out work request message */
    wc = calloc(1, sizeof(*wc));
    wc->work = malloc(sizeof(*work_in));

    if (likely(wc->work)) {
        wc->thr = thr;
        work_copy(wc->work, work_in);

        if (likely(tq_push(thr_info[work_thr_id].q, wc))) {
            return true;
        }
    }

    workio_cmd_free(wc);
    return false;
}


static bool should_pause(int thr_id) {
    bool ret = false;

    pthread_mutex_lock(&stratum_ctx->sock_lock);

    if (!stratum_ctx->ready) {
        ret = true;
    }

    pthread_mutex_unlock(&stratum_ctx->sock_lock);

    return ret;
}


/**
 * @brief miner_thread
 * @param userdata
 * @return
 */
static void *miner_thread(void *userdata) {
    struct thr_info *mythr = userdata;
    const int thr_id = mythr->id;
    struct work work = { { 0 } };
    uint32_t max_nonce;
    uint32_t end_nonce = 0xffffffffU / opt_n_threads * (thr_id + 1) - 0x20;

    struct cryptonight_ctx *persistentctx = (struct cryptonight_ctx *) create_persistent_ctx(thr_id);

    if (cpu_info.count > 1 && opt_affinity != -1L) {
        affine_to_cpu_mask(thr_id, (unsigned long) opt_affinity);
    }

    uint32_t *nonceptr = NULL;
    uint32_t hash[8] __attribute__((aligned(32)));

    while (1) {
        if (should_pause(thr_id)) {
            sleep(1);
            continue;
        }

        pthread_mutex_lock(&stratum_ctx->work_lock);

        if (memcmp(work.job_id, stratum_ctx->g_work.job_id, 64)) {
            work_copy(&work, &stratum_ctx->g_work);
            nonceptr = (uint32_t*) (((char*) work.blob) + 39);
            *nonceptr = 0xffffffffU / opt_n_threads * thr_id;
        }

        pthread_mutex_unlock(&stratum_ctx->work_lock);

        work_restart[thr_id].restart = 0;

        if (*nonceptr + LP_SCANTIME > end_nonce) {
            max_nonce = end_nonce;
        } else {
            max_nonce = *nonceptr + LP_SCANTIME;
        }

        unsigned long hashes_done = 0;

        struct timeval tv_start;
        gettimeofday(&tv_start, NULL);

        /* scan nonces for a proof-of-work hash */
        const int rc = scanhash_cryptonight(thr_id, hash, work.blob, work.blob_size, work.target, max_nonce, &hashes_done, persistentctx);
        stats_add_hashes(thr_id, &tv_start, hashes_done);

        if (!rc) {
            continue;
        }

        memcpy(work.hash, hash, 32);
        submit_work(mythr, &work);
        ++(*nonceptr);
    }

    tq_freeze(mythr->q);
    return NULL;
}


/**
 * @brief miner_thread_double
 * @param userdata
 * @return
 */
static void *miner_thread_double(void *userdata) {
    struct thr_info *mythr = userdata;
    const int thr_id = mythr->id;
    struct work work = { { 0 } };
    uint32_t max_nonce;
    uint32_t end_nonce = 0xffffffffU / opt_n_threads * (thr_id + 1) - 0x20;

    struct cryptonight_ctx *persistentctx = (struct cryptonight_ctx *) create_persistent_ctx(thr_id);

    if (cpu_info.count > 1 && opt_affinity != -1L) {
        affine_to_cpu_mask(thr_id, (unsigned long) opt_affinity);
    }

    uint32_t *nonceptr0 = NULL;
    uint32_t *nonceptr1 = NULL;
    uint8_t double_hash[64];
    uint8_t double_blob[sizeof(work.blob) * 2];

    while (1) {
        if (should_pause(thr_id)) {
            sleep(1);
            continue;
        }

        pthread_mutex_lock(&stratum_ctx->work_lock);

        if (memcmp(work.job_id, stratum_ctx->g_work.job_id, 64)) {
            work_copy(&work, &stratum_ctx->g_work);

            memcpy(double_blob,                  work.blob, work.blob_size);
            memcpy(double_blob + work.blob_size, work.blob, work.blob_size);

            nonceptr0 = (uint32_t*) (((char*) double_blob) + 39);
            nonceptr1 = (uint32_t*) (((char*) double_blob) + 39 + work.blob_size);

            *nonceptr0 = 0xffffffffU / (opt_n_threads * 2) * thr_id;
            *nonceptr1 = 0xffffffffU / (opt_n_threads * 2) * (thr_id + opt_n_threads);
        }

        pthread_mutex_unlock(&stratum_ctx->work_lock);

        work_restart[thr_id].restart = 0;

        if (*nonceptr0 + (LP_SCANTIME / 2) > end_nonce) {
            max_nonce = end_nonce;
        } else {
            max_nonce = *nonceptr0 + (LP_SCANTIME / 2);
        }

        unsigned long hashes_done = 0;

        struct timeval tv_start;
        gettimeofday(&tv_start, NULL);

        /* scan nonces for a proof-of-work hash */
        const int rc = scanhash_cryptonight_double(thr_id, (uint32_t *) double_hash, double_blob, work.blob_size, work.target, max_nonce, &hashes_done, persistentctx);
        stats_add_hashes(thr_id, &tv_start, hashes_done);

        if (!rc) {
            continue;
        }

        if (rc & 1) {
            memcpy(work.hash, double_hash, 32);
            memcpy(work.blob, double_blob, work.blob_size);
            submit_work(mythr, &work);
        }

        if (rc & 2) {
            memcpy(work.hash, double_hash + 32, 32);
            memcpy(work.blob, double_blob + work.blob_size, work.blob_size);
            submit_work(mythr, &work);
        }

        ++(*nonceptr0);
        ++(*nonceptr1);
    }

    tq_freeze(mythr->q);
    return NULL;
}



/**
 * @brief stratum_thread
 * @param userdata
 * @return
 */
static void *timer_thread(void *userdata) {
    const int max_user_time  = 100 - opt_donate_level;
    int user_time_remaning   = max_user_time;
    int donate_time_remaning = 0;


    while (1) {
        sleep(60);

        if (user_time_remaning > 0) {
            if (--user_time_remaning == 0) {
                g_want_donate = true;

                donate_time_remaning = opt_donate_level;
                stratum_disconnect(stratum_ctx);
                continue;
            }
        }

        if (donate_time_remaning > 0) {
            if (--donate_time_remaning == 0) {
                g_want_donate = false;

                user_time_remaning = max_user_time;
                stratum_disconnect(stratum_ctx);
                continue;
            }
        }
    }
}


static void switch_stratum() {
    static bool want_donate = false;

    if (g_want_donate && !want_donate) {
        stratum_ctx->url = "stratum+tcp://donate.xmrig.com:443";
        applog(LOG_NOTICE, "Switching to dev pool");
        want_donate = true;
    }

    if (!g_want_donate && want_donate) {
        stratum_ctx->url = backup_active ? opt_backup_url : opt_url;
        applog(LOG_NOTICE, "Switching to user pool: \"%s\"", stratum_ctx->url);
        want_donate = false;
    }
}



/**
 * @brief stratum_thread
 * @param userdata
 * @return
 */
static void *stratum_thread(void *userdata) {
    char *s;

    stratum_ctx->url   = opt_url;
    stratum_ctx->ready = false;

    while (1) {
        int failures = 0;
        switch_stratum();

        while (!stratum_ctx->curl) {
            pthread_mutex_lock(&stratum_ctx->work_lock);
            stratum_ctx->g_work_time = 0;
            pthread_mutex_unlock(&stratum_ctx->work_lock);

            restart_threads();
            switch_stratum();

            if (!stratum_connect(stratum_ctx, stratum_ctx->url) || !stratum_authorize(stratum_ctx, opt_user, opt_pass)) {
                stratum_disconnect(stratum_ctx);
                failures++;

                if (failures > opt_retries && opt_backup_url) {
                    failures = 0;

                    backup_active = !backup_active;
                    stratum_ctx->url = backup_active ? opt_backup_url : opt_url;
                    sleep(opt_retry_pause);

                    applog(LOG_WARNING, "Switch to: \"%s\"", stratum_ctx->url);
                    continue;
                }

                applog(LOG_ERR, "...retry after %d seconds", opt_retry_pause);
                sleep(opt_retry_pause);
            }
        }

        gen_workify(stratum_ctx);

        if (opt_keepalive && !stratum_socket_full(stratum_ctx, 90)) {
            stratum_keepalived(stratum_ctx);
        }

        if (!stratum_socket_full(stratum_ctx, 300)) {
            applog(LOG_ERR, "Stratum connection timed out");
            s = NULL;
        } else {
            s = stratum_recv_line(stratum_ctx);
        }

        if (!s) {
            stratum_disconnect(stratum_ctx);
            applog(LOG_ERR, "Stratum connection interrupted");
            continue;
        }

        if (!stratum_handle_method(stratum_ctx, s)) {
            stratum_handle_response(s);
        }

        free(s);
    }

    return NULL ;
}


/**
 * @brief start work I/O thread
 * @return
 */
static bool start_workio() {
    work_thr_id = opt_n_threads;

    struct thr_info *thr = &thr_info[work_thr_id];
    thr->id = work_thr_id;
    thr->q = tq_new();

    if (unlikely(!thr->q || pthread_create(&thr->pth, NULL, workio_thread, thr))) {
        return false;
    }

    return true;
}


/**
 * @brief start_stratum
 * @return
 */
static bool start_stratum() {
    stratum_thr_id = opt_n_threads + 1;

    stratum_ctx = persistent_calloc(1, sizeof(struct stratum_ctx));
    pthread_mutex_init(&stratum_ctx->work_lock, NULL);
    pthread_mutex_init(&stratum_ctx->sock_lock, NULL);

    struct thr_info *thr = &thr_info[stratum_thr_id];
    thr->id = stratum_thr_id;
    thr->q = tq_new();

    if (unlikely(!thr->q || pthread_create(&thr->pth, NULL, stratum_thread, thr))) {
        return false;
    }

     tq_push(thr_info[stratum_thr_id].q, strdup(opt_url));
     return true;
}


/**
 * @brief start_timer
 * @return
 */
static bool start_timer() {
    timer_thr_id = opt_n_threads + 2;

    if (opt_donate_level < 1) {
        return true;
    }

    struct thr_info *thr = &thr_info[timer_thr_id];
    thr->id = timer_thr_id;
    thr->q = tq_new();

    if (unlikely(!thr->q || pthread_create(&thr->pth, NULL, timer_thread, thr))) {
        return false;
    }

    return true;
}


/**
 * @brief start_mining
 * @return
 */
static bool start_mining() {
    for (int i = 0; i < opt_n_threads; i++) {
        struct thr_info *thr = &thr_info[i];

        thr->id = i;
        thr->q = tq_new();

        if (unlikely(!thr->q || pthread_create(&thr->pth, NULL, opt_double_hash ? miner_thread_double : miner_thread, thr))) {
            applog(LOG_ERR, "thread %d create failed", i);
            return false;
        }
    }

    return true;
}


/**
 * @brief main
 * @param argc
 * @param argv
 * @return
 */
int main(int argc, char *argv[]) {
    cpu_init();
    applog_init();
    parse_cmdline(argc, argv);
    persistent_memory_allocate();
    print_summary();

    stats_init();
    os_specific_init();

    work_restart = persistent_calloc(opt_n_threads, sizeof(*work_restart));
    thr_info     = persistent_calloc(opt_n_threads + 3, sizeof(struct thr_info));

    if (!start_workio()) {
        applog(LOG_ERR, "workio thread create failed");
        return 1;
    }

    if (!start_stratum()) {
        applog(LOG_ERR, "stratum thread create failed");
        return 1;
    }

    start_timer();

    if (!start_mining()) {
        return 1;
    }

    pthread_join(thr_info[work_thr_id].pth, NULL);
    applog(LOG_INFO, "workio thread dead, exiting.");
    persistent_memory_free();
    return 0;
}

