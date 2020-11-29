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


#include "base/io/Env.h"
#include "base/kernel/Process.h"
#include "version.h"


#include <regex>
#include <uv.h>
#include <map>


#ifndef _WIN32
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


#ifdef XMRIG_FEATURE_ENV
static std::map<String, String> variables;


static void createVariables()
{
    variables.insert({ "XMRIG_VERSION",  APP_VERSION });
    variables.insert({ "XMRIG_KIND",     APP_KIND });
    variables.insert({ "XMRIG_HOSTNAME", Env::hostname() });
    variables.insert({ "XMRIG_EXE",      Process::exepath() });
    variables.insert({ "XMRIG_EXE_DIR",  Process::location(Process::ExeLocation) });
    variables.insert({ "XMRIG_CWD",      Process::location(Process::CwdLocation) });
    variables.insert({ "XMRIG_HOME_DIR", Process::location(Process::HomeLocation) });
    variables.insert({ "XMRIG_TEMP_DIR", Process::location(Process::TempLocation) });
    variables.insert({ "XMRIG_DATA_DIR", Process::location(Process::DataLocation) });
}
#endif


} // namespace xmrig


xmrig::String xmrig::Env::expand(const char *in, const std::map<String, String> &extra)
{
#   ifdef XMRIG_FEATURE_ENV
    if (in == nullptr) {
        return {};
    }

    std::string text(in);
    if (text.size() < 4) {
        return text.c_str();
    }

    static const std::regex env_re{R"--(\$\{([^}]+)\})--"};

    std::map<std::string, String> vars;

    for (std::sregex_iterator i = std::sregex_iterator(text.begin(), text.end(), env_re); i != std::sregex_iterator(); ++i) {
        std::smatch m = *i;
        const auto var = m.str();

        if (vars.count(var)) {
            continue;
        }

        vars.insert({ var, get(m[1].str().c_str(), extra) });
    }

    for (const auto &kv : vars) {
        if (kv.second.isNull()) {
            continue;
        }

        size_t pos = 0;
        while ((pos = text.find(kv.first, pos)) != std::string::npos) {
            text.replace(pos, kv.first.size(), kv.second);
            pos += kv.second.size();
        }
    }

    return text.c_str();
#   else
    return in;
#   endif
}


xmrig::String xmrig::Env::get(const String &name, const std::map<String, String> &extra)
{
#   ifdef XMRIG_FEATURE_ENV
    if (variables.empty()) {
        createVariables();
    }

    const auto it = variables.find(name);
    if (it != variables.end()) {
        return it->second;
    }

    if (!extra.empty()) {
        const auto it = extra.find(name);
        if (it != extra.end()) {
            return it->second;
        }
    }
#   endif

    return static_cast<const char *>(getenv(name));
}


xmrig::String xmrig::Env::hostname()
{
    char buf[UV_MAXHOSTNAMESIZE]{};
    size_t size = sizeof(buf);

#   if UV_VERSION_HEX >= 0x010c00
    if (uv_os_gethostname(buf, &size) == 0) {
        return static_cast<const char *>(buf);
    }
#   else
    if (gethostname(buf, size) == 0) {
        return static_cast<const char *>(buf);
    }
#   endif

    return {};
}
