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

#ifndef __OPTIONS_H__
#define __OPTIONS_H__

#include <stdbool.h>
#include <stdint.h>

#ifndef ARRAY_SIZE
#   define ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))
#endif


enum xmr_algo_variant {
    XMR_AV0_AUTO,
    XMR_AV1_AESNI,
    XMR_AV2_AESNI_DOUBLE,
    XMR_AV3_SOFT_AES,
    XMR_AV4_SOFT_AES_DOUBLE,
    XMR_AV_MAX
};


extern bool opt_colors;
extern bool opt_keepalive;
extern bool opt_background;
extern bool opt_double_hash;
extern bool opt_safe;
extern char *opt_url;
extern char *opt_backup_url;
extern char *opt_userpass;
extern char *opt_user;
extern char *opt_pass;
extern int opt_n_threads;
extern int opt_algo_variant;
extern int opt_retry_pause;
extern int opt_retries;
extern int opt_donate_level;
extern int opt_max_cpu_usage;
extern int64_t opt_affinity;

void parse_cmdline(int argc, char *argv[]);
void show_usage_and_exit(int status);
void show_version_and_exit(void);

extern void proper_exit(int reason);


#endif /* __OPTIONS_H__ */
