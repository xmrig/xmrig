/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018-2020 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2020 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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


#include <algorithm>


#include "backend/opencl/OclThreads.h"
#include "3rdparty/rapidjson/document.h"
#include "base/io/json/Json.h"


xmrig::OclThreads::OclThreads(const rapidjson::Value &value)
{
    if (value.IsArray()) {
        for (auto &v : value.GetArray()) {
            OclThread thread(v);
            if (thread.isValid()) {
                add(std::move(thread));
            }
        }
    }
}


xmrig::OclThreads::OclThreads(const std::vector<OclDevice> &devices, const Algorithm &algorithm)
{
    for (const auto &device : devices) {
        device.generate(algorithm, *this);
    }
}


bool xmrig::OclThreads::isEqual(const OclThreads &other) const
{
    if (isEmpty() && other.isEmpty()) {
        return true;
    }

    return count() == other.count() && std::equal(m_data.begin(), m_data.end(), other.m_data.begin());
}


rapidjson::Value xmrig::OclThreads::toJSON(rapidjson::Document &doc) const
{
    using namespace rapidjson;
    auto &allocator = doc.GetAllocator();

    Value out(kArrayType);

    out.SetArray();

    for (const OclThread &thread : m_data) {
        out.PushBack(thread.toJSON(doc), allocator);
    }

    return out;
}
