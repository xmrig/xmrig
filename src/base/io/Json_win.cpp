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


#include <windows.h>


#ifdef __GNUC__
#   include <fcntl.h>
#   include <ext/stdio_filebuf.h>
#endif


#include <fstream>


#include "base/io/Json.h"
#include "rapidjson/document.h"
#include "rapidjson/istreamwrapper.h"
#include "rapidjson/ostreamwrapper.h"
#include "rapidjson/prettywriter.h"


#if defined(_MSC_VER) || defined (__GNUC__)
static std::wstring toUtf16(const char *str)
{
    const int size = static_cast<int>(strlen(str));
    std::wstring ret;

    int len = MultiByteToWideChar(CP_UTF8, 0, str, size, nullptr, 0);
    if (len > 0) {
        ret.resize(static_cast<size_t>(len));
        MultiByteToWideChar(CP_UTF8, 0, str, size, &ret[0], len);
    }

    return ret;
}
#endif


bool xmrig::Json::get(const char *fileName, rapidjson::Document &doc)
{
    using namespace rapidjson;
    constexpr const std::ios_base::openmode mode = std::ios_base::in | std::ios_base::binary;

#   if defined(_MSC_VER)
    std::ifstream ifs(toUtf16(fileName), mode);
    if (!ifs.is_open()) {
        return false;
    }
#   elif defined(__GNUC__)
    const int fd = _wopen(toUtf16(fileName).c_str(), _O_RDONLY | _O_BINARY);
    if (fd == -1) {
        return false;
    }

    __gnu_cxx::stdio_filebuf<char> buf(fd, mode);
    std::istream ifs(&buf);
#   else
    std::ifstream ifs(fileName, mode);
    if (!ifs.is_open()) {
        return false;
    }
#   endif

    IStreamWrapper isw(ifs);
    doc.ParseStream<kParseCommentsFlag | kParseTrailingCommasFlag>(isw);

    return !doc.HasParseError() && doc.IsObject();
}


bool xmrig::Json::save(const char *fileName, const rapidjson::Document &doc)
{
    using namespace rapidjson;
    constexpr const std::ios_base::openmode mode = std::ios_base::out | std::ios_base::binary | std::ios_base::trunc;

#   if defined(_MSC_VER)
    std::ofstream ofs(toUtf16(fileName), mode);
    if (!ofs.is_open()) {
        return false;
    }
#   elif defined(__GNUC__)
    const int fd = _wopen(toUtf16(fileName).c_str(), _O_WRONLY | _O_BINARY | _O_TRUNC);
    if (fd == -1) {
        return false;
    }

    __gnu_cxx::stdio_filebuf<char> buf(fd, mode);
    std::ostream ofs(&buf);
#   else
    std::ofstream ofs(fileName, mode);
    if (!ofs.is_open()) {
        return false;
    }
#   endif

    OStreamWrapper osw(ofs);
    PrettyWriter<OStreamWrapper> writer(osw);
    doc.Accept(writer);

    return true;
}
