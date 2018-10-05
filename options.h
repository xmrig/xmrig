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

#ifndef XMRIG_OPTIONS_H
#define XMRIG_OPTIONS_H

#include <stdbool.h>
#include <stdint.h>


#ifndef ARRAY_SIZE
#   define ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))
#endif


enum Algo {
    ALGO_CRYPTONIGHT,      /* CryptoNight (Monero) */
    ALGO_CRYPTONIGHT_LITE, /* CryptoNight-Lite (AEON) */
};


enum Variant {
    VARIANT_AUTO = -1,
    VARIANT_0    = 0,
    VARIANT_1    = 1,
    VARIANT_2    = 2,
    VARIANT_MAX
};


enum AlgoVariant {
    AV_AUTO,        // --av=0 Automatic mode.
    AV_SINGLE,      // --av=1  Single hash mode
    AV_DOUBLE,      // --av=2  Double hash mode
    AV_SINGLE_SOFT, // --av=3  Single hash mode (Software AES)
    AV_DOUBLE_SOFT, // --av=4  Double hash mode (Software AES)
    AV_MAX
};


enum Assembly {
    ASM_NONE,
    ASM_AUTO,
    ASM_INTEL,
    ASM_RYZEN,
    ASM_MAX
};


extern bool opt_colors;
extern bool opt_keepalive;
extern bool opt_background;
extern bool opt_double_hash;
extern bool opt_safe;
extern bool opt_nicehash;
extern char *opt_url;
extern char *opt_backup_url;
extern char *opt_userpass;
extern char *opt_user;
extern char *opt_pass;
extern int opt_n_threads;
extern int opt_retry_pause;
extern int opt_retries;
extern int opt_donate_level;
extern int opt_max_cpu_usage;
extern int64_t opt_affinity;

extern enum Algo opt_algo;
extern enum Variant opt_variant;
extern enum AlgoVariant opt_av;
extern enum Assembly opt_assembly;

void parse_cmdline(int argc, char *argv[]);
void show_usage_and_exit(int status);
void show_version_and_exit(void);
const char *get_current_algo_name(void);
const char *get_current_variant_name(void);

extern void proper_exit(int reason);


#endif /* XMRIG_OPTIONS_H */
