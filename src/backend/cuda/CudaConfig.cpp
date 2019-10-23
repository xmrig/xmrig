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


#include "backend/cuda/CudaConfig.h"
#include "backend/cuda/CudaConfig_gen.h"
#include "base/io/json/Json.h"
#include "base/io/log/Log.h"
#include "rapidjson/document.h"


namespace xmrig {


static const char *kDevicesHint = "devices-hint";
static const char *kEnabled     = "enabled";


extern template class Threads<CudaThreads>;


}


rapidjson::Value xmrig::CudaConfig::toJSON(rapidjson::Document &doc) const
{
    using namespace rapidjson;
    auto &allocator = doc.GetAllocator();

    Value obj(kObjectType);

    obj.AddMember(StringRef(kEnabled),  m_enabled, allocator);

    m_threads.toJSON(obj, doc);

    return obj;
}


void xmrig::CudaConfig::read(const rapidjson::Value &value)
{
    if (value.IsObject()) {
        m_enabled   = Json::getBool(value, kEnabled, m_enabled);

        setDevicesHint(Json::getString(value, kDevicesHint));

        m_threads.read(value);

        generate();
    }
    else if (value.IsBool()) {
        m_enabled = value.GetBool();

        generate();
    }
    else {
        m_shouldSave = true;

        generate();
    }
}


void xmrig::CudaConfig::generate()
{
    if (!isEnabled() || m_threads.has("*")) {
        return;
    }

    size_t count = 0;

//    count += xmrig::generate<Algorithm::CN>(m_threads, devices);
//    count += xmrig::generate<Algorithm::CN_LITE>(m_threads, devices);
//    count += xmrig::generate<Algorithm::CN_HEAVY>(m_threads, devices);
//    count += xmrig::generate<Algorithm::CN_PICO>(m_threads, devices);
//    count += xmrig::generate<Algorithm::RANDOM_X>(m_threads, devices);

    m_shouldSave = count > 0;
}


void xmrig::CudaConfig::setDevicesHint(const char *devicesHint)
{
    if (devicesHint == nullptr) {
        return;
    }

    const auto indexes = String(devicesHint).split(',');
    m_devicesHint.reserve(indexes.size());

    for (const auto &index : indexes) {
        m_devicesHint.push_back(strtoul(index, nullptr, 10));
    }
}
