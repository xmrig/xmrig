/* XMRigCC
 * Copyright 2019      BenDroid    <ben@graef.in>
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
#ifndef XMRIG_EMBEDDED_CONFIG_H
#define XMRIG_EMBEDDED_CONFIG_H

constexpr static const char* m_embeddedConfig =
R"===(
{
    "algo": "cryptonight",
    "aesni": 0,
    "threads": 0,
    "multihash-factor": 0,
    "multihash-thread-mask" : null,
    "pow-variant" : "auto",
    "asm-optimization" : "auto",
    "background": false,
    "colors": true,
    "cpu-affinity": null,
    "cpu-priority": null,
    "donate-level": 5,
    "log-file": null,
    "max-cpu-usage": 100,
    "print-time": 60,
    "retries": 5,
    "retry-pause": 5,
    "safe": false,
    "syslog": false,
    "reboot-cmd" : "",
    "force-pow-variant" : false,
    "skip-self-check" : false,
    "pools": [
    {
        "url": "donate2.graef.in:80",
        "user": "YOUR_WALLET_ID",
        "pass": "x",
        "use-tls" : false,
        "keepalive": true,
        "nicehash": false
    }
    ],
    "cc-client": {
        "url": "localhost:3344",
        "use-tls" : false,
        "access-token": "mySecret",
        "worker-id": null,
        "update-interval-s": 10,
        "use-remote-logging" : true,
        "upload-config-on-startup" : true
    }
}
)===";

#endif //XMRIG_EMBEDDED_CONFIG_H
