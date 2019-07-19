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


#include "backend/common/Threads.h"
#include "backend/cpu/CpuThread.h"
#include "rapidjson/document.h"


namespace xmrig {


static const char *kAsterisk = "*";


} // namespace xmrig


template <class T>
const std::vector<T> &xmrig::Threads<T>::get(const String &profileName) const
{
    static std::vector<T> empty;
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
        if (member.value.IsArray()) {
            std::vector<T> threads;

            for (auto &v : member.value.GetArray()) {
                T thread(v);
                if (thread.isValid()) {
                    threads.push_back(std::move(thread));
                }
            }

            if (!threads.empty()) {
                move(member.name.GetString(), std::move(threads));
            }

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

    if (name.contains("/")) {
        const String base = name.split('/').at(0);
        if (has(base)) {
            return base;
        }
    }

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
        Value arr(kArrayType);

        for (const T &thread : kv.second) {
            arr.PushBack(thread.toJSON(doc), allocator);
        }

        out.AddMember(kv.first.toJSON(), arr, allocator);
    }

    for (const Algorithm &algo : m_disabled) {
        out.AddMember(StringRef(algo.shortName()), false, allocator);
    }

    for (const auto &kv : m_aliases) {
        out.AddMember(StringRef(kv.first.shortName()), kv.second.toJSON(), allocator);
    }
}


namespace xmrig {

template class Threads<CpuThread>;

} // namespace xmrig
