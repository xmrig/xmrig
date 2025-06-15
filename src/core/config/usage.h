/* XMRig
 * Copyright (c) 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright (c) 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright (c) 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright (c) 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright (c) 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright (c) 2018-2024 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2024 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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


#include <string>


namespace xmrig {


static inline const std::string &usage()
{
    static std::string u;

    if (!u.empty()) {
        return u;
    }

    u += "Usage: " APP_ID " [OPTIONS]\n\nNetwork:\n";
    u += "  -o, --url=URL                 URL of mining server\n";
    u += "  -a, --algo=ALGO               mining algorithm https://xmrig.com/docs/algorithms\n";
    u += "      --coin=COIN               specify coin instead of algorithm\n";
    u += "  -u, --user=USERNAME           username for mining server\n";
    u += "  -p, --pass=PASSWORD           password for mining server\n";
    u += "  -O, --userpass=U:P            username:password pair for mining server\n";
    u += "  -x, --proxy=HOST:PORT         connect through a SOCKS5 proxy\n";
    u += "  -k, --keepalive               send keepalived packet for prevent timeout (needs pool support)\n";
    u += "      --nicehash                enable nicehash.com support\n";
    u += "      --rig-id=ID               rig identifier for pool-side statistics (needs pool support)\n";

#   ifdef XMRIG_FEATURE_TLS
    u += "      --tls                     enable SSL/TLS support (needs pool support)\n";
    u += "      --tls-fingerprint=HEX     pool TLS certificate fingerprint for strict certificate pinning\n";
#   endif

    u += "      --dns-ipv6                prefer IPv6 records from DNS responses\n";
    u += "      --dns-ttl=N               N seconds (default: 30) TTL for internal DNS cache\n";

#   ifdef XMRIG_FEATURE_HTTP
    u += "      --daemon                  use daemon RPC instead of pool for solo mining\n";
    u += "      --daemon-zmq-port=N       daemon's zmq-pub port number (only use it if daemon has it enabled)\n";
    u += "      --daemon-poll-interval=N  daemon poll interval in milliseconds (default: 1000)\n";
    u += "      --daemon-job-timeout=N    daemon job timeout in milliseconds (default: 15000)\n";
    u += "      --self-select=URL         self-select block templates from URL\n";
    u += "      --submit-to-origin        also submit solution back to self-select URL\n";
#   endif

    u += "  -r, --retries=N               number of times to retry before switch to backup server (default: 5)\n";
    u += "  -R, --retry-pause=N           time to pause between retries (default: 5)\n";
    u += "      --user-agent              set custom user-agent string for pool\n";
    u += "      --donate-level=N          donate level, default 1%% (1 minute in 100 minutes)\n";
    u += "      --donate-over-proxy=N     control donate over xmrig-proxy feature\n";

    u += "\nCPU backend:\n";

    u += "      --no-cpu                  disable CPU mining backend\n";
    u += "  -t, --threads=N               number of CPU threads, proper CPU affinity required for some optimizations.\n";
    u += "      --cpu-affinity=N          set process affinity to CPU core(s), mask 0x3 for cores 0 and 1\n";
    u += "  -v, --av=N                    algorithm variation, 0 auto select\n";
    u += "      --cpu-priority=N          set process priority (0 idle, 2 normal to 5 highest)\n";
    u += "      --cpu-max-threads-hint=N  maximum CPU threads count (in percentage) hint for autoconfig\n";
    u += "      --cpu-memory-pool=N       number of 2 MB pages for persistent memory pool, -1 (auto), 0 (disable)\n";
    u += "      --cpu-no-yield            prefer maximum hashrate rather than system response/stability\n";
    u += "      --no-huge-pages           disable huge pages support\n";
#   ifdef XMRIG_OS_LINUX
    u += "      --hugepage-size=N         custom hugepage size in kB\n";
#   endif
#   ifdef XMRIG_ALGO_RANDOMX
    u += "      --huge-pages-jit          enable huge pages support for RandomX JIT code\n";
#   endif
#   ifdef XMRIG_FEATURE_ASM
#   ifdef XMRIG_FEATURE_ASM_AMD
    u += "      --asm=ASM                 ASM optimizations, possible values: auto, none, intel, ryzen, bulldozer\n";
#   else
    u += "      --asm=ASM                 ASM optimizations, possible values: auto, none, intel\n";
#   endif
#   endif

#   if defined(__x86_64__) || defined(_M_AMD64)
    u += "      --argon2-impl=IMPL        argon2 implementation: x86_64, SSE2, SSSE3, XOP, AVX2, AVX-512F\n";
#   endif

#   ifdef XMRIG_ALGO_RANDOMX
    u += "      --randomx-init=N          threads count to initialize RandomX dataset\n";
    u += "      --randomx-no-numa         disable NUMA support for RandomX\n";
    u += "      --randomx-mode=MODE       RandomX mode: auto, fast, light\n";
    u += "      --randomx-1gb-pages       use 1GB hugepages for RandomX dataset (Linux only)\n";
    u += "      --randomx-wrmsr=N         write custom value(s) to MSR registers or disable MSR mod (-1)\n";
    u += "      --randomx-no-rdmsr        disable reverting initial MSR values on exit\n";
    u += "      --randomx-cache-qos       enable Cache QoS\n";
#   endif

#   ifdef XMRIG_FEATURE_OPENCL
    u += "\nOpenCL backend:\n";
    u += "      --opencl                  enable OpenCL mining backend\n";
    u += "      --opencl-devices=N        comma separated list of OpenCL devices to use\n";
    u += "      --opencl-platform=N       OpenCL platform index or name\n";
    u += "      --opencl-loader=PATH      path to OpenCL-ICD-Loader (OpenCL.dll or libOpenCL.so)\n";
    u += "      --opencl-no-cache         disable OpenCL cache\n";
    u += "      --print-platforms         print available OpenCL platforms and exit\n";
#   endif

#   ifdef XMRIG_FEATURE_CUDA
    u += "\nCUDA backend:\n";
    u += "      --cuda                    enable CUDA mining backend\n";
    u += "      --cuda-loader=PATH        path to CUDA plugin (xmrig-cuda.dll or libxmrig-cuda.so)\n";
    u += "      --cuda-devices=N          comma separated list of CUDA devices to use\n";
    u += "      --cuda-bfactor-hint=N     bfactor hint for autoconfig (0-12)\n";
    u += "      --cuda-bsleep-hint=N      bsleep hint for autoconfig\n";
#   endif
#   ifdef XMRIG_FEATURE_NVML
    u += "      --no-nvml                 disable NVML (NVIDIA Management Library) support\n";
#   endif

#   ifdef XMRIG_FEATURE_HTTP
    u += "\nAPI:\n";
    u += "      --api-worker-id=ID        custom worker-id for API\n";
    u += "      --api-id=ID               custom instance ID for API\n";
    u += "      --http-host=HOST          bind host for HTTP API (default: 127.0.0.1)\n";
    u += "      --http-port=N             bind port for HTTP API\n";
    u += "      --http-access-token=T     access token for HTTP API\n";
    u += "      --http-no-restricted      enable full remote access to HTTP API (only if access token set)\n";
#   endif

#   ifdef XMRIG_FEATURE_TLS
    u += "\nTLS:\n";
    u += "      --tls-gen=HOSTNAME        generate TLS certificate for specific hostname\n";
    u += "      --tls-cert=FILE           load TLS certificate chain from a file in the PEM format\n";
    u += "      --tls-cert-key=FILE       load TLS certificate private key from a file in the PEM format\n";
    u += "      --tls-dhparam=FILE        load DH parameters for DHE ciphers from a file in the PEM format\n";
    u += "      --tls-protocols=N         enable specified TLS protocols, example: \"TLSv1 TLSv1.1 TLSv1.2 TLSv1.3\"\n";
    u += "      --tls-ciphers=S           set list of available ciphers (TLSv1.2 and below)\n";
    u += "      --tls-ciphersuites=S      set list of available TLSv1.3 ciphersuites\n";
#   endif

    u += "\nLogging:\n";

#   ifdef HAVE_SYSLOG_H
    u += "  -S, --syslog                  use system log for output messages\n";
#   endif

    u += "  -l, --log-file=FILE           log all output to a file\n";
    u += "      --print-time=N            print hashrate report every N seconds\n";
#   if defined(XMRIG_FEATURE_NVML) || defined(XMRIG_FEATURE_ADL)
    u += "      --health-print-time=N     print health report every N seconds\n";
#   endif
    u += "      --no-color                disable colored output\n";
    u += "      --verbose                 verbose output\n";

    u += "\nMisc:\n";

    u += "  -c, --config=FILE             load a JSON-format configuration file\n";
    u += "  -B, --background              run the miner in the background\n";
    u += "  -V, --version                 output version information and exit\n";
    u += "  -h, --help                    display this help and exit\n";
    u += "      --dry-run                 test configuration and exit\n";

#   ifdef XMRIG_FEATURE_HWLOC
    u += "      --export-topology         export hwloc topology to a XML file and exit\n";
#   endif

#   ifdef XMRIG_OS_WIN
    u += "      --title                   set custom console window title\n";
    u += "      --no-title                disable setting console window title\n";
#   endif
    u += "      --pause-on-battery        pause mine on battery power\n";
    u += "      --pause-on-active=N       pause mine when the user is active (resume after N seconds of last activity)\n";

#   ifdef XMRIG_FEATURE_BENCHMARK
    u += "      --stress                  run continuous stress test to check system stability\n";
    u += "      --bench=N                 run benchmark, N can be between 1M and 10M\n";
#   ifdef XMRIG_FEATURE_HTTP
    u += "      --submit                  perform an online benchmark and submit result for sharing\n";
    u += "      --verify=ID               verify submitted benchmark by ID\n";
#   endif
    u += "      --seed=SEED               custom RandomX seed for benchmark\n";
    u += "      --hash=HASH               compare benchmark result with specified hash\n";
#   endif

#   ifdef XMRIG_FEATURE_DMI
    u += "      --no-dmi                  disable DMI/SMBIOS reader\n";
#   endif

    return u;
}


} /* namespace xmrig */

#endif /* XMRIG_USAGE_H */
