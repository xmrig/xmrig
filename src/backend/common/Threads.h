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

#ifndef XMRIG_THREADS_H
#define XMRIG_THREADS_H


#include <map>
#include <set>


#include "3rdparty/rapidjson/fwd.h"
#include "base/crypto/Algorithm.h"
#include "base/tools/String.h"


namespace xmrig {


template <class T>
class Threads
{
public:
    inline bool has(const char *profile) const                                         { return m_profiles.count(profile) > 0; }
    inline bool isDisabled(const Algorithm &algo) const                                { return m_disabled.count(algo) > 0; }
    inline bool isEmpty() const                                                        { return m_profiles.empty(); }
    inline bool isExist(const Algorithm &algo) const                                   { return isDisabled(algo) || m_aliases.count(algo) > 0 || has(algo.shortName()); }
    inline const T &get(const Algorithm &algo, bool strict = false) const              { return get(profileName(algo, strict)); }
    inline void disable(const Algorithm &algo)                                         { m_disabled.insert(algo); }
    inline void setAlias(const Algorithm &algo, const char *profile)                   { m_aliases[algo] = profile; }

    inline size_t move(const char *profile, T &&threads)
    {
        if (has(profile)) {
            return 0;
        }

        const size_t count = threads.count();

        if (!threads.isEmpty()) {
            m_profiles.insert({ profile, std::move(threads) });
        }

        return count;
    }

    const T &get(const String &profileName) const;
    size_t read(const rapidjson::Value &value);
    String profileName(const Algorithm &algorithm, bool strict = false) const;
    void toJSON(rapidjson::Value &out, rapidjson::Document &doc) const;

private:
    std::map<Algorithm, String> m_aliases;
    std::map<String, T> m_profiles;
    std::set<Algorithm> m_disabled;
};


} /* namespace xmrig */


#endif /* XMRIG_THREADS_H */
