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


#include <type_traits>

#include "backend/common/Threads.h"
#include "3rdparty/rapidjson/document.h"
#include "backend/cpu/CpuThreads.h"
#include "crypto/cn/CnAlgo.h"


#ifdef XMRIG_FEATURE_OPENCL
#   include "backend/opencl/OclThreads.h"
#endif


#ifdef XMRIG_FEATURE_CUDA
#   include "backend/cuda/CudaThreads.h"
#endif


namespace xmrig {


static const char *kAsterisk = "*";
static const char *kCn2      = "cn/2";


} // namespace xmrig


template <class T>
const T &xmrig::Threads<T>::get(const String &profileName) const
{
    static T empty;
    if (profileName.isNull() || !has(profileName)) {
        return empty;
    }

    return m_profiles.at(profileName);
}


template <class T>
size_t xmrig::Threads<T>::read(const rapidjson::Value &value)
{
    using namespace rapidjson;

    for (auto &member : value.GetObject()) {
        if (member.value.IsArray() || member.value.IsObject()) {
            T threads(member.value);

            if (!threads.isEmpty()) {
                move(member.name.GetString(), std::move(threads));
            }
        }
    }

    for (auto &member : value.GetObject()) {
        if (member.value.IsArray() || member.value.IsObject()) {
            continue;
        }

        const Algorithm algo(member.name.GetString());
        if (!algo.isValid()) {
            continue;
        }

        if (member.value.IsBool() && member.value.IsFalse()) {
            disable(algo);
            continue;
        }

        if (member.value.IsString()) {
            if (has(member.value.GetString())) {
                m_aliases.insert({ algo, member.value.GetString() });
            }
            else {
                m_disabled.insert(algo);
            }
        }
    }

    return m_profiles.size();
}


template <class T>
xmrig::String xmrig::Threads<T>::profileName(const Algorithm &algorithm, bool strict) const
{
    if (isDisabled(algorithm)) {
        return String();
    }

    const String name = algorithm.shortName();
    if (has(name)) {
        return name;
    }

    if (m_aliases.count(algorithm) > 0) {
        return m_aliases.at(algorithm);
    }

    if (strict) {
        return String();
    }

    if (algorithm.family() == Algorithm::CN && CnAlgo<>::base(algorithm) == Algorithm::CN_2 && has(kCn2)) {
        return kCn2;
    }

    if (name.contains("/")) {
        const String base = name.split('/').at(0);
        if (has(base)) {
            return base;
        }
    }

    if (std::is_same<T, CpuThreads>::value && name == "defyx" && has("rx")) return "rx";

    if (has(kAsterisk)) {
        return kAsterisk;
    }

    return String();
}


template <class T>
void xmrig::Threads<T>::toJSON(rapidjson::Value &out, rapidjson::Document &doc) const
{
    using namespace rapidjson;
    auto &allocator = doc.GetAllocator();

    for (const auto &kv : m_profiles) {
        out.AddMember(kv.first.toJSON(), kv.second.toJSON(doc), allocator);
    }

    for (const Algorithm &algo : m_disabled) {
        out.AddMember(StringRef(algo.shortName()), false, allocator);
    }

    for (const auto &kv : m_aliases) {
        out.AddMember(StringRef(kv.first.shortName()), kv.second.toJSON(), allocator);
    }
}


namespace xmrig {

template class Threads<CpuThreads>;

#ifdef XMRIG_FEATURE_OPENCL
template class Threads<OclThreads>;
#endif

#ifdef XMRIG_FEATURE_CUDA
template class Threads<CudaThreads>;
#endif

} // namespace xmrig
