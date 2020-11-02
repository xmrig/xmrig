/* XMRig
 * Copyright (c) 2018-2020 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2020 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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


#include <ctime>
#include <string>
#include <uv.h>


#include "base/kernel/Process.h"
#include "3rdparty/fmt/core.h"
#include "base/tools/Chrono.h"
#include "version.h"


#ifdef XMRIG_OS_WIN
#   ifdef _MSC_VER
#       include <direct.h>
#       define MKDIR(path) _mkdir(path.c_str());
#   else
#       define MKDIR(path) mkdir((path).c_str());
#   endif
#else
#   define MKDIR(path) mkdir(path.c_str(), 0700);
#endif


namespace xmrig {


static char pathBuf[520];
static std::string dataDir;


static std::string getPath(Process::Location location)
{
    size_t size = sizeof(pathBuf);

    if (location == Process::DataLocation) {
        if (!dataDir.empty()) {
            return dataDir;
        }

        location = Process::ExeLocation;
    }

    if (location == Process::HomeLocation) {
#       if UV_VERSION_HEX >= 0x010600
        return uv_os_homedir(pathBuf, &size) < 0 ? "" : std::string(pathBuf, size);
#       else
        location = Process::ExeLocation;
#       endif
    }

    if (location == Process::TempLocation) {
#       if UV_VERSION_HEX >= 0x010900
        return uv_os_tmpdir(pathBuf, &size) < 0 ? "" : std::string(pathBuf, size);
#       else
        location = Process::ExeLocation;
#       endif
    }

    if (location == Process::ExeLocation) {
        if (uv_exepath(pathBuf, &size) < 0) {
            return {};
        }

        const auto path = std::string(pathBuf, size);
        const auto pos  = path.rfind(*XMRIG_DIR_SEPARATOR);

        if (pos != std::string::npos) {
            return path.substr(0, pos);
        }

        return path;
    }

    if (location == Process::CwdLocation) {
        return uv_cwd(pathBuf, &size) < 0 ? "" : std::string(pathBuf, size);
    }

    return {};
}


static void setDataDir(const char *path)
{
    if (path == nullptr) {
        return;
    }

    std::string dir = path;
    if (!dir.empty() && (dir.back() == '/' || dir.back() == '\\')) {
        dir.pop_back();
    }

    if (!dir.empty() && uv_chdir(dir.c_str()) == 0) {
        dataDir = dir;
    }
}


} // namespace xmrig


xmrig::Process::Process(int argc, char **argv) :
    m_arguments(argc, argv)
{
    srand(static_cast<unsigned int>(Chrono::currentMSecsSinceEpoch() ^ reinterpret_cast<uintptr_t>(this)));

    setDataDir(m_arguments.value("--data-dir", "-d"));

#   ifdef XMRIG_SHARED_DATADIR
    if (dataDir.empty()) {
        dataDir = fmt::format("{}" XMRIG_DIR_SEPARATOR ".xmrig" XMRIG_DIR_SEPARATOR, location(HomeLocation));
        MKDIR(dataDir);

        dataDir += APP_KIND;
        MKDIR(dataDir);

        uv_chdir(dataDir.c_str());
    }
#   endif
}


int xmrig::Process::ppid()
{
#   if UV_VERSION_HEX >= 0x011000
    return uv_os_getppid();
#   else
    return 0;
#   endif
}


xmrig::String xmrig::Process::exepath()
{
    size_t size = sizeof(pathBuf);

    return uv_exepath(pathBuf, &size) < 0 ? "" : String(pathBuf, size);
}


xmrig::String xmrig::Process::location(Location location, const char *fileName)
{
    auto path = getPath(location);
    if (path.empty() || fileName == nullptr) {
        return path.c_str();
    }

    return fmt::format("{}" XMRIG_DIR_SEPARATOR "{}", path, fileName).c_str();
}
