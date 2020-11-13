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

#include "base/net/stratum/benchmark/BenchConfig.h"
#include "3rdparty/fmt/core.h"
#include "3rdparty/rapidjson/document.h"
#include "base/io/json/Json.h"


#include <string>


#ifdef _MSC_VER
#   define strcasecmp  _stricmp
#endif


namespace xmrig {


const char *BenchConfig::kAlgo      = "algo";
const char *BenchConfig::kBenchmark = "benchmark";
const char *BenchConfig::kHash      = "hash";
const char *BenchConfig::kId        = "id";
const char *BenchConfig::kSeed      = "seed";
const char *BenchConfig::kSize      = "size";
const char *BenchConfig::kSubmit    = "submit";
const char *BenchConfig::kToken     = "token";
const char *BenchConfig::kVerify    = "verify";

#ifndef XMRIG_DEBUG_BENCHMARK_API
const char *BenchConfig::kApiHost   = "api.xmrig.com";
#else
const char *BenchConfig::kApiHost   = "127.0.0.1";
#endif

} // namespace xmrig


xmrig::BenchConfig::BenchConfig(uint32_t size, const String &id, const rapidjson::Value &object) :
    m_algorithm(Json::getString(object, kAlgo)),
    m_submit(Json::getBool(object, kSubmit)),
    m_id(id),
    m_seed(Json::getString(object, kSeed)),
    m_token(Json::getString(object, kToken)),
    m_size(size),
    m_hash(0)
{
    if (!m_algorithm.isValid() || m_algorithm.family() != Algorithm::RANDOM_X) {
        m_algorithm = Algorithm::RX_0;
    }

    const char *hash = Json::getString(object, kHash);
    if (hash) {
        m_hash = strtoull(hash, nullptr, 16);
    }
}


xmrig::BenchConfig *xmrig::BenchConfig::create(const rapidjson::Value &object)
{
    if (!object.IsObject() || object.ObjectEmpty()) {
        return nullptr;
    }

    const uint32_t size = getSize(Json::getString(object, kSize));
    const String id     = Json::getString(object, kVerify);

    if (size == 0 && id.isEmpty()) {
        return nullptr;
    }

    return new BenchConfig(size, id, object);
}


rapidjson::Value xmrig::BenchConfig::toJSON(rapidjson::Document &doc) const
{
    using namespace rapidjson;
    Value out(kObjectType);
    auto &allocator = doc.GetAllocator();

    if (m_size == 0) {
        out.AddMember(StringRef(kSize), 0U, allocator);
    }
    else if (m_size < 1000000) {
        out.AddMember(StringRef(kSize), Value(fmt::format("{}K", m_size / 1000).c_str(), allocator), allocator);
    }
    else {
        out.AddMember(StringRef(kSize), Value(fmt::format("{}M", m_size / 1000000).c_str(), allocator), allocator);
    }

    out.AddMember(StringRef(kAlgo),     m_algorithm.toJSON(), allocator);
    out.AddMember(StringRef(kSubmit),   m_submit, allocator);
    out.AddMember(StringRef(kVerify),   m_id.toJSON(), allocator);
    out.AddMember(StringRef(kToken),    m_token.toJSON(), allocator);
    out.AddMember(StringRef(kSeed),     m_seed.toJSON(), allocator);

    if (m_hash) {
        out.AddMember(StringRef(kHash), Value(fmt::format("{:016X}", m_hash).c_str(), allocator), allocator);
    }
    else {
        out.AddMember(StringRef(kHash), kNullType, allocator);
    }

    return out;
}


uint32_t xmrig::BenchConfig::getSize(const char *benchmark)
{
    if (!benchmark) {
        return 0;
    }

    const auto size = strtoul(benchmark, nullptr, 10);
    if (size >= 1 && size <= 10) {
        return strcasecmp(benchmark, fmt::format("{}M", size).c_str()) == 0 ? size * 1000000 : 0;
    }

    if (size == 250 || size == 500) {
        return strcasecmp(benchmark, fmt::format("{}K", size).c_str()) == 0 ? size * 1000 : 0;
    }

    return 0;
}
