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

#include <fstream>


#include "base/io/json/Json.h"
#include "3rdparty/rapidjson/document.h"
#include "3rdparty/rapidjson/istreamwrapper.h"
#include "3rdparty/rapidjson/ostreamwrapper.h"
#include "3rdparty/rapidjson/prettywriter.h"


bool xmrig::Json::get(const char *fileName, rapidjson::Document &doc)
{
    std::ifstream ifs(fileName, std::ios_base::in | std::ios_base::binary);
    if (!ifs.is_open()) {
        return false;
    }

    rapidjson::IStreamWrapper isw(ifs);
    doc.ParseStream<rapidjson::kParseCommentsFlag | rapidjson::kParseTrailingCommasFlag>(isw);

    return !doc.HasParseError() && (doc.IsObject() || doc.IsArray());
}


bool xmrig::Json::save(const char *fileName, const rapidjson::Document &doc)
{
    std::ofstream ofs(fileName, std::ios_base::out | std::ios_base::binary | std::ios_base::trunc);
    if (!ofs.is_open()) {
        return false;
    }

    rapidjson::OStreamWrapper osw(ofs);
    rapidjson::PrettyWriter<rapidjson::OStreamWrapper> writer(osw);

#   ifdef XMRIG_JSON_SINGLE_LINE_ARRAY
    writer.SetFormatOptions(rapidjson::kFormatSingleLineArray);
#   endif

    doc.Accept(writer);

    return true;
}


bool xmrig::Json::convertOffset(const char* fileName, size_t offset, size_t& line, size_t& pos, std::vector<std::string>& s)
{
    std::ifstream ifs(fileName, std::ios_base::in | std::ios_base::binary);
    if (!ifs.is_open()) {
        return false;
    }

    return convertOffset(ifs, offset, line, pos, s);
}
