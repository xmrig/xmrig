/* XMRig
 * Copyright (c) 2016-2022 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2022 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#include "base/kernel/Entry.h"
#include "base/io/log/Log.h"
#include "base/kernel/OS.h"
#include "base/kernel/Process.h"
#include "base/kernel/Versions.h"
#include "base/tools/Arguments.h"
#include "version.h"


#include <iostream>


#ifdef XMRIG_FEATURE_OPENCL
#   include "backend/opencl/wrappers/OclLib.h"
#   include "backend/opencl/wrappers/OclPlatform.h"
#endif

#ifdef XMRIG_FEATURE_HWLOC
#   include <hwloc.h>
#endif


namespace xmrig {


static bool showVersion(int &/*rc*/)
{
    if (!Process::arguments().contains("-V", "--version")) {
        return false;
    }

    std::cout << APP_NAME " v" << Process::version() << std::endl
              << " built on " __DATE__ " with " << Versions::kCompiler << "/" << Process::versions().get(Versions::kCompiler)
              << " (" << OS::arch << ")" << std::endl;

#   ifdef XMRIG_LEGACY
    std::cout << std::endl << "uv/" << Process::versions().get(Versions::kUv) << std::endl;

#   ifdef XMRIG_FEATURE_TLS
    std::cout << Versions::kTls << "/" << Process::versions().get(Versions::kTls) << std::endl;
#   endif

#   ifdef XMRIG_FEATURE_HWLOC
    std::cout << "hwloc/" << Process::versions().get(Versions::kHwloc) << std::endl;
#   endif
#   endif

    return true;
}


static bool showVersions(int &/*rc*/)
{
    if (Process::arguments().contains("--versions")) {
        for (const auto &kv : Process::versions().get()) {
            std::cout << kv.first << "/" << kv.second << std::endl;;
        }

        return true;
    }

    return false;
}


static bool userAgent(int &/*rc*/)
{
    Process::setUserAgent(Process::arguments().value("--user-agent"));

    if (Process::arguments().contains("--print-user-agent")) {
        std::cout << Process::userAgent() << std::endl;

        return true;
    }

    return false;
}


#ifdef XMRIG_FEATURE_HWLOC
static bool exportTopology(int &rc)
{
    if (!Process::arguments().contains("--export-topology")) {
        return false;
    }

    const auto path = Process::locate(Process::DataLocation, "topology.xml");

    hwloc_topology_t topology = nullptr;
    hwloc_topology_init(&topology);
    hwloc_topology_load(topology);

#   if HWLOC_API_VERSION >= 0x20000
    if (hwloc_topology_export_xml(topology, path, 0) == -1) {
#   else
    if (hwloc_topology_export_xml(topology, path) == -1) {
#   endif
        rc = 1;
        std::cout << "failed to export hwloc topology" << std::endl;
    }
    else {
        std::cout << "hwloc topology successfully exported to \"" << path << '"' << std::endl;
    }

    hwloc_topology_destroy(topology);

    return true;
}
#endif


} // namespace xmrig


xmrig::Entry::Entry(const Usage &usage)
{
    add(showVersion);
    add(showVersions);
    add(userAgent);

#   ifdef XMRIG_FEATURE_HWLOC
    add(exportTopology);
#   endif

    add([usage](int &/*rc*/) {
        if (!Process::arguments().contains("-h", "--help")) {
            return false;
        }

        std::cout << "Usage: " APP_ID " [OPTIONS]\n";
        std::cout << usage();

#       ifndef XMRIG_LEGACY
        std::cout << "\nBase:\n";
        std::cout << "  -h, --help                    print this help and exit\n";
        std::cout << "  -V, --version                 print " APP_ID " version and exit\n";
        std::cout << "      --versions                print versions and exit\n";
        std::cout << "  -d, --data-dir=<PATH>         alternative working directory\n";
        std::cout << "  -c, --config=<FILE>           load a JSON-format configuration file\n";
        std::cout << "  -B, --background              run " APP_ID " in the background\n";
        std::cout << "      --no-color                disable colored output\n";
        std::cout << "      --verbose=[LEVEL]         verbose level (0-5)\n";
        std::cout << "      --print-time=<N>          print report every N seconds\n";
        std::cout << "  -l, --log-file=<FILE>         log all output to a file\n";

#       ifdef HAVE_SYSLOG_H
        std::cout << "  -S, --syslog                  use system log for output messages\n";
#       endif

#       ifdef XMRIG_OS_WIN
        std::cout << "      --title=[TITLE]           set custom console window title\n";
#       endif

        std::cout << "      --user-agent=<UA>         set custom user agent string\n";
        std::cout << "      --print-user-agent        print current user agent and exit\n";
        std::cout << "      --dns-ipv6                prefer IPv6 records from DNS responses\n";
        std::cout << "      --dns-ttl=<N>             N seconds (default: 30) TTL for internal DNS cache\n";

#       ifdef XMRIG_FEATURE_HWLOC
        std::cout << "      --export-topology         export hwloc topology to a XML file and exit\n";
#       endif
#       endif

        return true;
    });

#   ifdef XMRIG_FEATURE_OPENCL
    add([](int &/*rc*/) {
        if (Process::arguments().contains("--print-platforms")) {
            if (OclLib::init()) {
                OclPlatform::print();
            }

            return true;
        }

        return false;
    });
#   endif

    add([](int &rc) {
        if (Process::arguments().contains("-B", "--background")) {
            Log::setBackground(true);

            return background(rc);
        }

        return false;
    });
}


bool xmrig::Entry::exec(int &rc) const
{
    for (const auto &fn : m_entries) {
        if (fn(rc)) {
            return true;
        }
    }

    return false;
}


void xmrig::Entry::add(Fn &&fn)
{
    m_entries.emplace_back(std::move(fn));
}
