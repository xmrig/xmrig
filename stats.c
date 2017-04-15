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

#include <pthread.h>
#include <sys/time.h>
#include <stdlib.h>

#include "stats.h"
#include "options.h"
#include "utils/applog.h"
#include "persistent_memory.h"


static unsigned long accepted_count = 0L;
static unsigned long rejected_count = 0L;
static double *thr_hashrates;
static double *thr_times;
static uint32_t target = 0;

pthread_mutex_t stats_lock;


static int timeval_subtract(struct timeval *result, struct timeval *x, struct timeval *y);


/**
 * @brief stats_init
 */
void stats_init() {
    pthread_mutex_init(&stats_lock, NULL);

    thr_hashrates = (double *) persistent_calloc(opt_n_threads, sizeof(double));
    thr_times = (double *) persistent_calloc(opt_n_threads, sizeof(double));
}



/**
 * @brief stats_set_target
 * @param target
 */
void stats_set_target(uint32_t new_target)
{
    target = new_target;

    applog(LOG_DEBUG, "Pool set diff to %g", ((double) 0xffffffff) / target);
}


/**
 * @brief stats_share_result
 * @param result
 */
void stats_share_result(bool success)
{
    double hashrate = 0.0;

    pthread_mutex_lock(&stats_lock);

    for (int i = 0; i < opt_n_threads; i++) {
        if (thr_times[i] > 0) {
            hashrate += thr_hashrates[i] / thr_times[i];
        }
    }

    success ? accepted_count++ : rejected_count++;
    pthread_mutex_unlock(&stats_lock);

    applog(LOG_INFO, "accepted: %lu/%lu (%.2f%%), %.2f H/s at diff %g",
            accepted_count, accepted_count + rejected_count,
            100. * accepted_count / (accepted_count + rejected_count), hashrate,
            (((double) 0xffffffff) / target));
}


void stats_add_hashes(int thr_id, struct timeval *tv_start, unsigned long hashes_done)
{
    struct timeval tv_end, diff;

    /* record scanhash elapsed time */
    gettimeofday(&tv_end, NULL);
    timeval_subtract(&diff, &tv_end, tv_start);

    if (diff.tv_usec || diff.tv_sec) {
        pthread_mutex_lock(&stats_lock);
        thr_hashrates[thr_id] = hashes_done;
        thr_times[thr_id] = (diff.tv_sec + 1e-6 * diff.tv_usec);
        pthread_mutex_unlock(&stats_lock);
    }
}


/* Subtract the `struct timeval' values X and Y,
   storing the result in RESULT.
   Return 1 if the difference is negative, otherwise 0.  */
static int timeval_subtract(struct timeval *result, struct timeval *x, struct timeval *y)
{
    /* Perform the carry for the later subtraction by updating Y. */
    if (x->tv_usec < y->tv_usec) {
        int nsec = (y->tv_usec - x->tv_usec) / 1000000 + 1;
        y->tv_usec -= 1000000 * nsec;
        y->tv_sec += nsec;
    }
    if (x->tv_usec - y->tv_usec > 1000000) {
        int nsec = (x->tv_usec - y->tv_usec) / 1000000;
        y->tv_usec += 1000000 * nsec;
        y->tv_sec -= nsec;
    }

    /* Compute the time remaining to wait.
     * `tv_usec' is certainly positive. */
    result->tv_sec = x->tv_sec - y->tv_sec;
    result->tv_usec = x->tv_usec - y->tv_usec;

    /* Return 1 if result is negative. */
    return x->tv_sec < y->tv_sec;
}
