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


#include <stdio.h>
#include <uv.h>


#ifdef XMRIG_FEATURE_TLS
#   include <openssl/opensslv.h>
#endif


#include "base/kernel/Entry.h"
#include "base/kernel/Process.h"
#include "core/config/usage.h"
#include "version.h"


static int showVersion()
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
    " 32-bit"
#   elif defined(__x86_64__) || defined(_M_AMD64)
    " 64-bit"
#   endif

#   if defined(__AES__) || defined(_MSC_VER)
    " AES"
#   endif
    "\n");

    printf("\nlibuv/%s\n", uv_version_string());

#   if defined(XMRIG_FEATURE_TLS) && defined(OPENSSL_VERSION_TEXT)
    {
        constexpr const char *v = OPENSSL_VERSION_TEXT + 8;
        printf("OpenSSL/%.*s\n", static_cast<int>(strchr(v, ' ') - v), v);
    }
#   endif

    return 0;
}


xmrig::Entry::Id xmrig::Entry::get(const Process &process)
{
    const Arguments &args = process.arguments();
    if (args.hasArg("-h") || args.hasArg("--help")) {
         return Usage;
    }

    if (args.hasArg("-V") || args.hasArg("--version")) {
         return Version;
    }

    return Default;
}


int xmrig::Entry::exec(const Process &, Id id)
{
    switch (id) {
    case Usage:
        printf(usage);
        return 0;

    case Version:
        return showVersion();

    default:
        break;
    }

    return 1;
}
