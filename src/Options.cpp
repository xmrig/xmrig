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


#include <jansson.h>
#include <string.h>
#include <uv.h>


#ifdef _MSC_VER
#   include "getopt/getopt.h"
#else
#   include <getopt.h>
#endif


#include "Cpu.h"
#include "donate.h"
#include "net/Url.h"
#include "Options.h"
#include "version.h"


#ifndef ARRAY_SIZE
#   define ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))
#endif


Options *Options::m_self = nullptr;


static char const usage[] = "\
Usage: " APP_ID " [OPTIONS]\n\
Options:\n\
  -a, --algo=ALGO       cryptonight (default) or cryptonight-lite\n\
  -o, --url=URL         URL of mining server\n\
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
  -l, --log-file=FILE   log all output to a file\n"
# ifdef HAVE_SYSLOG_H
"\
  -S, --syslog          use system log for output messages\n"
# endif
"\
      --max-cpu-usage=N maximum CPU usage for automatic threads mode (default 75)\n\
      --safe            safe adjust threads and av settings for current CPU\n\
      --nicehash        enable nicehash support\n\
      --print-time=N    print hashrate report every N seconds\n\
  -h, --help            display this help and exit\n\
  -V, --version         output version information and exit\n\
";


static char const short_options[] = "a:c:khBp:Px:r:R:s:t:T:o:u:O:v:Vl:S";


static struct option const options[] = {
    { "algo",          1, nullptr, 'a'  },
    { "av",            1, nullptr, 'v'  },
    { "background",    0, nullptr, 'B'  },
    { "config",        1, nullptr, 'c'  },
    { "cpu-affinity",  1, nullptr, 1020 },
    { "donate-level",  1, nullptr, 1003 },
    { "help",          0, nullptr, 'h'  },
    { "keepalive",     0, nullptr ,'k'  },
    { "log-file",      1, nullptr, 'l'  },
    { "max-cpu-usage", 1, nullptr, 1004 },
    { "nicehash",      0, nullptr, 1006 },
    { "no-color",      0, nullptr, 1002 },
    { "pass",          1, nullptr, 'p'  },
    { "print-time",    1, nullptr, 1007 },
    { "retries",       1, nullptr, 'r'  },
    { "retry-pause",   1, nullptr, 'R'  },
    { "safe",          0, nullptr, 1005 },
    { "syslog",        0, nullptr, 'S'  },
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


const char *Options::algoName() const
{
    return algo_names[m_algo];
}


Options::Options(int argc, char **argv) :
    m_background(false),
    m_colors(true),
    m_doubleHash(false),
    m_ready(false),
    m_safe(false),
    m_syslog(false),
    m_logFile(nullptr),
    m_algo(0),
    m_algoVariant(0),
    m_donateLevel(kDonateLevel),
    m_maxCpuUsage(75),
    m_printTime(60),
    m_retries(5),
    m_retryPause(5),
    m_threads(0),
    m_affinity(-1L)
{
    m_pools.push_back(new Url());

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

    if (!m_pools[0]->isValid()) {
        fprintf(stderr, "No pool URL supplied. Exiting.");
        return;
    }

    m_algoVariant = getAlgoVariant();
    if (m_algoVariant == AV2_AESNI_DOUBLE || m_algoVariant == AV4_SOFT_AES_DOUBLE) {
        m_doubleHash = true;
    }

    if (!m_threads) {
        m_threads = Cpu::optimalThreadsCount(m_algo, m_doubleHash, m_maxCpuUsage);
    }
    else if (m_safe) {
        const int count = Cpu::optimalThreadsCount(m_algo, m_doubleHash, m_maxCpuUsage);
        if (m_threads > count) {
            m_threads = count;
        }
    }

    m_ready = true;
}


Options::~Options()
{
}


bool Options::parseArg(int key, char *arg)
{
    char *p;
    int v;
    uint64_t ul;

    switch (key) {
    case 'a': /* --algo */
        if (!setAlgo(arg)) {
            return false;
        }
        break;

    case 'O': /* --userpass */
        if (!m_pools.back()->setUserpass(arg)) {
            return false;
        }

        break;

    case 'o': /* --url */
        if (m_pools.size() > 1 || m_pools[0]->isValid()) {
            Url *url = new Url(arg);
            if (url->isValid()) {
                m_pools.push_back(url);
            }
            else {
                delete url;
            }
        }
        else {
            m_pools[0]->parse(arg);
        }

        if (!m_pools.back()->isValid()) {
            return false;
        }

        break;

    case 'u': /* --user */
        m_pools.back()->setUser(arg);
        break;

    case 'p': /* --pass */
        m_pools.back()->setPassword(arg);
        break;

    case 'l': /* --log-file */
        free(m_logFile);
        m_logFile = strdup(arg);
        m_colors = false;
        break;

    case 'r': /* --retries */
        v = strtol(arg, nullptr, 10);
        if (v < 1 || v > 1000) {
            showUsage(1);
            return false;
        }

        m_retries = v;
        break;

    case 'R': /* --retry-pause */
        v = strtol(arg, nullptr, 10);
        if (v < 1 || v > 3600) {
            showUsage(1);
            return false;
        }

        m_retryPause = v;
        break;

    case 't': /* --threads */
        v = strtol(arg, nullptr, 10);
        if (v < 1 || v > 1024) {
            showUsage(1);
            return false;
        }

        m_threads = v;
        break;

    case 1004: /* --max-cpu-usage */
        v = strtol(arg, nullptr, 10);
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
        m_pools.back()->setKeepAlive(true);
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

    case 'S': /* --syslog */
        m_syslog = true;
        m_colors = false;
        break;

    case 'v': /* --av */
        v = strtol(arg, nullptr, 10);
        if (v < 0 || v > 1000) {
            showUsage(1);
            return false;
        }

        m_algoVariant = v;
        break;

    case 1020: /* --cpu-affinity */
        p  = strstr(arg, "0x");
        ul = p ? strtoul(p, NULL, 16) : atol(arg);
        if (ul > (1UL << Cpu::threads()) -1) {
            ul = -1;
        }

        m_affinity = ul;
        break;

    case 1002: /* --no-color */
        m_colors = false;
        break;

    case 1003: /* --donate-level */
        v = strtol(arg, nullptr, 10);
        if (v < 1 || v > 99) {
            showUsage(1);
            return false;
        }

        m_donateLevel = v;
        break;

    case 1006: /* --nicehash */
        m_pools.back()->setNicehash(true);
        break;

    case 1007: /* --print-time */
        v = strtol(arg, nullptr, 10);
        if (v < 0 || v > 1000) {
            showUsage(1);
            return false;
        }

        m_printTime = v;
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

#   if defined(__clang__)
    " with clang " __clang_version__);
#   elif defined(__GNUC__)
    " with GCC");
    printf(" %d.%d.%d", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#   elif defined(_MSC_VER)
    " with MSVC");
    printf(" %d", MSVC_VERSION);
#   else
    );
#   endif

    printf("\n features:"
#   if defined(__i386__) || defined(_M_IX86)
    " i386"
#   elif defined(__x86_64__) || defined(_M_AMD64)
    " x86_64"
#   endif

#   if defined(__AES__) || defined(_MSC_VER)
    " AES-NI"
#   endif
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


int Options::getAlgoVariant() const
{
#   ifndef XMRIG_NO_AEON
    if (m_algo == ALGO_CRYPTONIGHT_LITE) {
        return getAlgoVariantLite();
    }
#   endif

    if (m_algoVariant <= AV0_AUTO || m_algoVariant >= AV_MAX) {
        return Cpu::hasAES() ? AV1_AESNI : AV3_SOFT_AES;
    }

    if (m_safe && !Cpu::hasAES() && m_algoVariant <= AV2_AESNI_DOUBLE) {
        return m_algoVariant + 2;
    }

    return m_algoVariant;
}


#ifndef XMRIG_NO_AEON
int Options::getAlgoVariantLite() const
{
    if (m_algoVariant <= AV0_AUTO || m_algoVariant >= AV_MAX) {
        return Cpu::hasAES() ? AV2_AESNI_DOUBLE : AV4_SOFT_AES_DOUBLE;
    }

    if (m_safe && !Cpu::hasAES() && m_algoVariant <= AV2_AESNI_DOUBLE) {
        return m_algoVariant + 2;
    }

    return m_algoVariant;
}
#endif
