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


#include <uv.h>
#include <jansson.h>
#include <getopt.h>


#include "Console.h"
#include "Options.h"
#include "version.h"
#include "donate.h"
#include "net/Url.h"


#ifndef ARRAY_SIZE
#   define ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))
#endif


Options *Options::m_self = nullptr;


static char const usage[] = "\
Usage: " APP_ID " [OPTIONS]\n\
Options:\n\
  -a, --algo=ALGO       cryptonight (default) or cryptonight-lite\n\
  -o, --url=URL         URL of mining server\n\
  -b, --backup-url=URL  URL of backup mining server\n\
  -O, --userpass=U:P    username:password pair for mining server\n\
  -u, --user=USERNAME   username for mining server\n\
  -p, --pass=PASSWORD   password for mining server\n\
  -t, --threads=N       number of miner threads\n\
  -v, --av=N            algorithm variation, 0 auto select\n\
  -k, --keepalive       send keepalived for prevent timeout (need pool support)\n\
  -r, --retries=N       number of times to retry before switch to backup server (default: 5)\n\
  -R, --retry-pause=N   time to pause between retries (default: 5)\n\
      --cpu-affinity    set process affinity to CPU core(s), mask 0x3 for cores 0 and 1\n\
      --no-color        disable colored output\n\
      --donate-level=N  donate level, default 5%% (5 minutes in 100 minutes)\n\
  -B, --background      run the miner in the background\n\
  -c, --config=FILE     load a JSON-format configuration file\n\
      --max-cpu-usage=N maximum CPU usage for automatic threads mode (default 75)\n\
      --safe            safe adjust threads and av settings for current CPU\n\
      --nicehash        enable nicehash support\n\
  -h, --help            display this help and exit\n\
  -V, --version         output version information and exit\n\
";


static char const short_options[] = "a:c:khBp:Px:r:R:s:t:T:o:u:O:v:Vb:";


static struct option const options[] = {
    { "algo",          1, nullptr, 'a'  },
    { "av",            1, nullptr, 'v'  },
    { "background",    0, nullptr, 'B'  },
    { "backup-url",    1, nullptr, 'b'  },
    { "config",        1, nullptr, 'c'  },
    { "cpu-affinity",  1, nullptr, 1020 },
    { "donate-level",  1, nullptr, 1003 },
    { "help",          0, nullptr, 'h'  },
    { "keepalive",     0, nullptr ,'k'  },
    { "max-cpu-usage", 1, nullptr, 1004 },
    { "nicehash",      0, nullptr, 1006 },
    { "no-color",      0, nullptr, 1002 },
    { "pass",          1, nullptr, 'p'  },
    { "retries",       1, nullptr, 'r'  },
    { "retry-pause",   1, nullptr, 'R'  },
    { "safe",          0, nullptr, 1005 },
    { "threads",       1, nullptr, 't'  },
    { "url",           1, nullptr, 'o'  },
    { "user",          1, nullptr, 'u'  },
    { "userpass",      1, nullptr, 'O'  },
    { "version",       0, nullptr, 'V'  },
    { 0, 0, 0, 0 }
};


static const char *algo_names[] = {
    "cryptonight",
#   ifndef XMRIG_NO_AEON
    "cryptonight-lite"
#   endif
};


Options *Options::parse(int argc, char **argv)
{
    if (!m_self) {
        m_self = new Options(argc, argv);
    }

    return m_self;
}


Options::Options(int argc, char **argv) :
    m_background(false),
    m_colors(true),
    m_doubleHash(false),
    m_keepAlive(false),
    m_nicehash(false),
    m_ready(false),
    m_safe(false),
    m_pass(nullptr),
    m_user(nullptr),
    m_algo(0),
    m_algoVariant(0),
    m_donateLevel(kDonateLevel),
    m_maxCpuUsage(75),
    m_retries(5),
    m_retryPause(5),
    m_threads(0),
    m_affinity(-1L),
    m_backupUrl(nullptr),
    m_url(nullptr)
{
    int key;

    while (1) {
        key = getopt_long(argc, argv, short_options, options, NULL);
        if (key < 0) {
            break;
        }

        if (!parseArg(key, optarg)) {
            return;
        }
    }

    if (optind < argc) {
        fprintf(stderr, "%s: unsupported non-option argument '%s'\n", argv[0], argv[optind]);
        return;
    }

    if (!m_url) {
        LOG_ERR("No pool URL supplied. Exiting.", argv[0]);
        return;
    }

    if (!m_nicehash && m_url->isNicehash()) {
        m_nicehash = true;
    }

    if (!m_user) {
        m_user = strdup("x");
    }

    if (!m_pass) {
        m_pass = strdup("x");
    }

    m_ready = true;
}


Options::~Options()
{
    delete m_url;
    delete m_backupUrl;

    free(m_user);
    free(m_pass);
}


bool Options::parseArg(int key, char *arg)
{
//    char *p;
    int v;
//    uint64_t ul;
    Url *url;

    switch (key) {
    case 'a': /* --algo */
        if (!setAlgo(arg)) {
            return false;
        }
        break;

    case 'O': /* --userpass */
        if (!setUserpass(arg)) {
            return false;
        }
        break;

    case 'o': /* --url */
        url = parseUrl(arg);
        if (url) {
            free(m_url);
            m_url = url;
        }
        break;

    case 'b': /* --backup-url */
        url = parseUrl(arg);
        if (url) {
            free(m_backupUrl);
            m_backupUrl = url;
        }
        break;

    case 'u': /* --user */
        free(m_user);
        m_user = strdup(arg);
        break;

    case 'p': /* --pass */
        free(m_pass);
        m_pass = strdup(arg);
        break;

    case 'r': /* --retries */
        v = atoi(arg);
        if (v < 1 || v > 1000) {
            showUsage(1);
            return false;
        }

        m_retries = v;
        break;

    case 'R': /* --retry-pause */
        v = atoi(arg);
        if (v < 1 || v > 3600) {
            showUsage(1);
            return false;
        }

        m_retryPause = v;
        break;

    case 't': /* --threads */
        v = atoi(arg);
        if (v < 1 || v > 1024) {
            showUsage(1);
            return false;
        }

        m_threads = v;
        break;

    case 1004: /* --max-cpu-usage */
        v = atoi(arg);
        if (v < 1 || v > 100) {
            showUsage(1);
            return false;
        }

        m_maxCpuUsage = v;
        break;

    case 1005: /* --safe */
        m_safe = true;
        break;

    case 'k': /* --keepalive */
        m_keepAlive = true;
        break;

    case 'V': /* --version */
        showVersion();
        return false;

    case 'h': /* --help */
        showUsage(0);
        return false;

    case 'B': /* --background */
        m_background = true;
        m_colors = false;
        break;

    case 'v': /* --av */
        v = atoi(arg);
        if (v < 0 || v > 1000) {
            showUsage(1);
            return false;
        }

        m_algoVariant = v;
        break;

    case 1020: /* --cpu-affinity */
//        p  = strstr(arg, "0x");
//        ul = p ? strtoul(p, NULL, 16) : atol(arg);
//        if (ul > (1UL << cpu_info.total_logical_cpus) -1) {
//            ul = -1;
//        }

//        opt_affinity = ul;
        break;

    case 1002: /* --no-color */
        m_colors = false;
        break;

    case 1003: /* --donate-level */
        v = atoi(arg);
        if (v < 1 || v > 99) {
            showUsage(1);
            return false;
        }

        m_donateLevel = v;
        break;

    case 1006: /* --nicehash */
        m_nicehash = true;
        break;

    default:
        showUsage(1);
        return false;
    }

    return true;
}


Url *Options::parseUrl(const char *arg) const
{
    auto url = new Url(arg);
    if (!url->isValid()) {
        delete url;
        return nullptr;
    }

    return url;
}


void Options::showUsage(int status) const
{
    if (status) {
        fprintf(stderr, "Try \"" APP_ID "\" --help' for more information.\n");
    }
    else {
        printf(usage);
    }
}


void Options::showVersion()
{
    printf(APP_NAME " " APP_VERSION "\n built on " __DATE__

    #ifdef __GNUC__
    " with GCC");
    printf(" %d.%d.%d", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
    #endif

    printf("\n features:"
    #ifdef __i386__
    " i386"
    #endif
    #ifdef __x86_64__
    " x86_64"
    #endif
    #ifdef __AES__
    " AES-NI"
    #endif
    "\n");

    printf("\nlibuv/%s\n", uv_version_string());
    printf("libjansson/%s\n", JANSSON_VERSION);
}


bool Options::setAlgo(const char *algo)
{
    for (size_t i = 0; i < ARRAY_SIZE(algo_names); i++) {
        if (algo_names[i] && !strcmp(algo, algo_names[i])) {
            m_algo = i;
            break;
        }

#       ifndef XMRIG_NO_AEON
        if (i == ARRAY_SIZE(algo_names) - 1 && !strcmp(algo, "cryptonight-light")) {
            m_algo = ALGO_CRYPTONIGHT_LITE;
            break;
        }
#       endif

        if (i == ARRAY_SIZE(algo_names) - 1) {
            showUsage(1);
            return false;
        }
    }

    return true;
}


bool Options::setUserpass(const char *userpass)
{
    char *p = strchr(userpass, ':');
    if (!p) {
        showUsage(1);
        return false;
    }

    free(m_user);
    free(m_pass);

    m_user = static_cast<char*>(calloc(p - userpass + 1, 1));
    strncpy(m_user, userpass, p - userpass);
    m_pass = strdup(p + 1);

    return true;
}
