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

#ifndef XMRIG_BENCHCONFIG_H
#define XMRIG_BENCHCONFIG_H


#include "base/crypto/Algorithm.h"
#include "base/tools/String.h"


namespace xmrig {


class BenchConfig
{
public:
    static const char *kAlgo;
    static const char *kApiHost;
    static const char *kBenchmark;
    static const char *kHash;
    static const char *kId;
    static const char *kSeed;
    static const char *kSize;
    static const char *kSubmit;
    static const char *kToken;
    static const char *kVerify;

#   ifndef XMRIG_DEBUG_BENCHMARK_API
    static constexpr bool kApiTLS               = true;
    static constexpr const uint16_t kApiPort    = 443;
#   else
    static constexpr bool kApiTLS               = false;
    static constexpr const uint16_t kApiPort    = 18805;
#   endif

    BenchConfig(uint32_t size, const String &id, const rapidjson::Value &object);

    static BenchConfig *create(const rapidjson::Value &object);

    inline bool isSubmit() const                { return m_submit; }
    inline const Algorithm &algorithm() const   { return m_algorithm; }
    inline const String &id() const             { return m_id; }
    inline const String &seed() const           { return m_seed; }
    inline uint32_t size() const                { return m_size; }
    inline uint64_t hash() const                { return m_hash; }

private:
    static uint32_t getSize(const char *benchmark);

    Algorithm m_algorithm;
    bool m_submit;
    String m_id;
    String m_seed;
    uint32_t m_size;
    uint64_t m_hash;
};


} /* namespace xmrig */


#endif /* XMRIG_BENCHCONFIG_H */
