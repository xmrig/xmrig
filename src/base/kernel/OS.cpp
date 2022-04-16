/* XMRig
 * Copyright (c) 2018-2021 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2021 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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


#include "base/kernel/OS.h"
#include "3rdparty/fmt/core.h"
#include "base/kernel/Process.h"
#include "base/kernel/Versions.h"
#include "version.h"


#ifndef XMRIG_OS_WIN
#   include <unistd.h>
#endif


#ifndef UV_MAXHOSTNAMESIZE
#   ifdef MAXHOSTNAMELEN
#       define UV_MAXHOSTNAMESIZE (MAXHOSTNAMELEN + 1)
#   else
#       define UV_MAXHOSTNAMESIZE 256
#   endif
#endif


namespace xmrig {


#if (XMRIG_ARM == 8 || defined(__arm64__) || defined(__aarch64__) || defined(_M_ARM64))
const char *OS::arch = "arm64";
#elif (XMRIG_ARM == 7 || defined(__arm__) || defined(_M_ARM))
const char *OS::arch = "arm";
#elif (defined(__x86_64__) || defined(_M_AMD64))
const char *OS::arch = "x86_64";
#elif (defined(_X86_) || defined(_M_IX86))
const char *OS::arch = "x86";
#else
static_assert (false, "Unsupported CPU or compiler");
#endif


} // namespace xmrig


std::string xmrig::OS::userAgent()
{
    return fmt::format("{}/{} ({}; {}) uv/{} {}/{}",
                       APP_NAME,
                       APP_VERSION,
                       name(),
                       arch,
                       Process::versions()[Versions::kUv],
                       Versions::kCompiler,
                       Process::versions()[Versions::kCompiler]
            );
}


#ifndef XMRIG_OS_WIN
xmrig::String xmrig::OS::hostname()
{
    char buf[UV_MAXHOSTNAMESIZE]{};

    if (gethostname(buf, sizeof(buf)) == 0) {
        return static_cast<const char *>(buf);
    }

    return {};
}
#endif


uint64_t xmrig::OS::freemem()
{
    return uv_get_free_memory();
}


uint64_t xmrig::OS::totalmem()
{
    return uv_get_total_memory();
}
