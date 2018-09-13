/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018      Lee Clagett <https://github.com/vtnerd>
 * Copyright 2018      SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2018 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#ifndef XMRIG_ALGORITHM_H
#define XMRIG_ALGORITHM_H


#include <vector>


#include "common/xmrig.h"


namespace xmrig {


class Algorithm
{
public:
    inline Algorithm() :
        m_algo(INVALID_ALGO),
        m_variant(VARIANT_AUTO)
    {}

    inline Algorithm(Algo algo, Variant variant) :
        m_variant(variant)
    {
        setAlgo(algo);
    }

    inline Algorithm(const char *algo)
    {
        parseAlgorithm(algo);
    }

    bool isEqual(const Algorithm &other) const { return m_algo == other.m_algo && m_variant == other.m_variant; }
    inline Algo algo() const                   { return m_algo; }
    inline const char *name() const            { return name(false); }
    inline const char *shortName() const       { return name(true); }
    inline Variant variant() const             { return m_variant; }
    inline void setVariant(Variant variant)    { m_variant = variant; }

    inline bool operator!=(const Algorithm &other) const  { return !isEqual(other); }
    inline bool operator==(const Algorithm &other) const  { return isEqual(other); }

    bool isValid() const;
    const char *variantName() const;
    void parseAlgorithm(const char *algo);
    void parseVariant(const char *variant);
    void parseVariant(int variant);
    void setAlgo(Algo algo);

#   ifdef XMRIG_PROXY_PROJECT
    void parseXmrStakAlgorithm(const char *algo);
#   endif

private:
    const char *name(bool shortName) const;

    Algo m_algo;
    Variant m_variant;
};


typedef std::vector<xmrig::Algorithm> Algorithms;


} /* namespace xmrig */

#endif /* __ALGORITHM_H__ */
