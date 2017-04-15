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
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <pthread.h>

#include "util.h"
#include "elist.h"
#include "utils/applog.h"


struct tq_ent {
    void *data;
    struct list_head q_node;
};


struct thread_q {
    struct list_head q;
    bool frozen;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
};


json_t *json_decode(const char *s)
{
    json_error_t err;
    json_t *val = json_loads(s, 0, &err);

    if (!val) {
        applog(LOG_ERR, "JSON decode failed(%d): %s", err.line, err.text);
    }

    return val;
}


/**
 * @brief bin2hex
 * @param p
 * @param len
 * @return
 */
char *bin2hex(const unsigned char *p, size_t len)
{
    char *s = malloc((len * 2) + 1);
    if (!s) {
        return NULL;
    }

    for (int i = 0; i < len; i++) {
        sprintf(s + (i * 2), "%02x", (unsigned int) p[i]);
    }

    return s;
}


/**
 * @brief hex2bin
 * @param p
 * @param hexstr
 * @param len
 * @return
 */
bool hex2bin(unsigned char *p, const char *hexstr, size_t len)
{
    char hex_byte[3];
    char *ep;

    hex_byte[2] = '\0';

    while (*hexstr && len) {
        if (!hexstr[1]) {
            applog(LOG_ERR, "hex2bin str truncated");
            return false;
        }

        hex_byte[0] = hexstr[0];
        hex_byte[1] = hexstr[1];
        *p = (unsigned char) strtol(hex_byte, &ep, 16);
        if (*ep) {
            applog(LOG_ERR, "hex2bin failed on '%s'", hex_byte);
            return false;
        }

        p++;
        hexstr += 2;
        len--;
    }

    return (len == 0 && *hexstr == 0) ? true : false;
}


/**
 * @brief tq_new
 * @return
 */
struct thread_q *tq_new(void)
{
    struct thread_q *tq;

    tq = calloc(1, sizeof(*tq));
    if (!tq)
        return NULL;

    INIT_LIST_HEAD(&tq->q);
    pthread_mutex_init(&tq->mutex, NULL);
    pthread_cond_init(&tq->cond, NULL);

    return tq;
}


/**
 * @brief tq_free
 * @param tq
 */
void tq_free(struct thread_q *tq)
{
    struct tq_ent *ent, *iter;

    if (!tq)
        return;

    list_for_each_entry_safe(ent, iter, &tq->q, q_node) {
        list_del(&ent->q_node);
        free(ent);
    }

    pthread_cond_destroy(&tq->cond);
    pthread_mutex_destroy(&tq->mutex);

    memset(tq, 0, sizeof(*tq)); /* poison */
    free(tq);
}


/**
 * @brief tq_freezethaw
 * @param tq
 * @param frozen
 */
static void tq_freezethaw(struct thread_q *tq, bool frozen)
{
    pthread_mutex_lock(&tq->mutex);

    tq->frozen = frozen;

    pthread_cond_signal(&tq->cond);
    pthread_mutex_unlock(&tq->mutex);
}


/**
 * @brief tq_freeze
 * @param tq
 */
void tq_freeze(struct thread_q *tq)
{
    tq_freezethaw(tq, true);
}


/**
 * @brief tq_thaw
 * @param tq
 */
void tq_thaw(struct thread_q *tq)
{
    tq_freezethaw(tq, false);
}


/**
 * @brief tq_push
 * @param tq
 * @param data
 * @return
 */
bool tq_push(struct thread_q *tq, void *data)
{
    struct tq_ent *ent;
    bool rc = true;

    ent = calloc(1, sizeof(*ent));
    if (!ent)
        return false;

    ent->data = data;
    INIT_LIST_HEAD(&ent->q_node);

    pthread_mutex_lock(&tq->mutex);

    if (!tq->frozen) {
        list_add_tail(&ent->q_node, &tq->q);
    } else {
        free(ent);
        rc = false;
    }

    pthread_cond_signal(&tq->cond);
    pthread_mutex_unlock(&tq->mutex);

    return rc;
}


/**
 * @brief tq_pop
 * @param tq
 * @param abstime
 * @return
 */
void *tq_pop(struct thread_q *tq, const struct timespec *abstime)
{
    struct tq_ent *ent;
    void *rval = NULL;
    int rc;

    pthread_mutex_lock(&tq->mutex);

    if (!list_empty(&tq->q))
        goto pop;

    if (abstime)
        rc = pthread_cond_timedwait(&tq->cond, &tq->mutex, abstime);
    else
        rc = pthread_cond_wait(&tq->cond, &tq->mutex);
    if (rc)
        goto out;
    if (list_empty(&tq->q))
        goto out;

pop:
    ent = list_entry(tq->q.next, struct tq_ent, q_node);
    rval = ent->data;

    list_del(&ent->q_node);
    free(ent);

out:
    pthread_mutex_unlock(&tq->mutex);
    return rval;
}
