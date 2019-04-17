/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018-2019 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2019 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#ifndef XMRIG_USAGE_H
#define XMRIG_USAGE_H


#include "version.h"


namespace xmrig {


static char const usage[] = "\
Usage: " APP_ID " [OPTIONS]\n\
Options:\n\
  -a, --algo=ALGO               specify the algorithm to use\n\
                                  cryptonight\n"
#ifndef XMRIG_NO_AEON
"\
                                  cryptonight-lite\n"
#endif
#ifndef XMRIG_NO_SUMO
"\
                                  cryptonight-heavy\n"
#endif
#ifndef XMRIG_NO_CN_PICO
"\
                                  cryptonight-pico\n"
#endif
"\
  -o, --url=URL                 URL of mining server\n\
  -O, --userpass=U:P            username:password pair for mining server\n\
  -u, --user=USERNAME           username for mining server\n\
  -p, --pass=PASSWORD           password for mining server\n\
      --rig-id=ID               rig identifier for pool-side statistics (needs pool support)\n\
  -t, --threads=N               number of miner threads\n\
  -v, --av=N                    algorithm variation, 0 auto select\n\
  -k, --keepalive               send keepalived packet for prevent timeout (needs pool support)\n\
      --nicehash                enable nicehash.com support\n"
#ifdef XMRIG_FEATURE_TLS
"\
      --tls                     enable SSL/TLS support (needs pool support)\n\
      --tls-fingerprint=F       pool TLS certificate fingerprint, if set enable strict certificate pinning\n"
#endif
#ifdef XMRIG_FEATURE_HTTP
"\
      --daemon                  use daemon RPC instead of pool for solo mining\n\
      --daemon-poll-interval=N  daemon poll interval in milliseconds (default: 1000)\n"
#endif
"\
  -r, --retries=N               number of times to retry before switch to backup server (default: 5)\n\
  -R, --retry-pause=N           time to pause between retries (default: 5)\n\
      --cpu-affinity            set process affinity to CPU core(s), mask 0x3 for cores 0 and 1\n\
      --cpu-priority            set process priority (0 idle, 2 normal to 5 highest)\n\
      --no-huge-pages           disable huge pages support\n\
      --no-color                disable colored output\n\
      --variant                 algorithm PoW variant\n\
      --donate-level=N          donate level, default 5%% (5 minutes in 100 minutes)\n\
      --user-agent              set custom user-agent string for pool\n\
  -B, --background              run the miner in the background\n\
  -c, --config=FILE             load a JSON-format configuration file\n\
  -l, --log-file=FILE           log all output to a file\n"
# ifdef HAVE_SYSLOG_H
"\
  -S, --syslog                  use system log for output messages\n"
# endif
"\
      --max-cpu-usage=N         maximum CPU usage for automatic threads mode (default: 100)\n\
      --safe                    safe adjust threads and av settings for current CPU\n\
      --asm=ASM                 ASM optimizations, possible values: auto, none, intel, ryzen, bulldozer.\n\
      --print-time=N            print hashrate report every N seconds\n"
#ifdef XMRIG_FEATURE_HTTP
"\
      --api-worker-id=ID        custom worker-id for API\n\
      --api-id=ID               custom instance ID for API\n\
      --http-enabled            enable HTTP API\n\
      --http-host=HOST          bind host for HTTP API (default: 127.0.0.1)\n\
      --http-port=N             bind port for HTTP API\n\
      --http-access-token=T     access token for HTTP API\n\
      --http-no-restricted      enable full remote access to HTTP API (only if access token set)\n"
#endif
"\
      --dry-run                 test configuration and exit\n\
  -h, --help                    display this help and exit\n\
  -V, --version                 output version information and exit\n\
";


} /* namespace xmrig */

#endif /* XMRIG_USAGE_H */
