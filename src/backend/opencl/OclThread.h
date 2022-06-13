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

#ifndef XMRIG_OCLTHREAD_H
#define XMRIG_OCLTHREAD_H


#include "3rdparty/rapidjson/fwd.h"


#include <bitset>
#include <vector>


namespace xmrig {


class OclThread
{
public:
    OclThread() = delete;
    OclThread(uint32_t index, uint32_t intensity, uint32_t worksize, uint32_t stridedIndex, uint32_t memChunk, uint32_t threads, uint32_t unrollFactor) :
        m_threads(threads, -1),
        m_index(index),
        m_memChunk(memChunk),
        m_stridedIndex(stridedIndex),
        m_unrollFactor(unrollFactor),
        m_worksize(worksize)
    {
        setIntensity(intensity);
    }

#   ifdef XMRIG_ALGO_RANDOMX
    OclThread(uint32_t index, uint32_t intensity, uint32_t worksize, uint32_t threads, bool gcnAsm, bool datasetHost, uint32_t bfactor) :
        m_datasetHost(datasetHost),
        m_gcnAsm(gcnAsm),
        m_fields(2),
        m_threads(threads, -1),
        m_bfactor(bfactor),
        m_index(index),
        m_memChunk(0),
        m_stridedIndex(0),
        m_worksize(worksize)
    {
        setIntensity(intensity);
    }
#   endif

#   ifdef XMRIG_ALGO_KAWPOW
    OclThread(uint32_t index, uint32_t intensity, uint32_t worksize, uint32_t threads) :
        m_fields(8),
        m_threads(threads, -1),
        m_index(index),
        m_memChunk(0),
        m_stridedIndex(0),
        m_unrollFactor(1),
        m_worksize(worksize)
    {
        setIntensity(intensity);
    }
#   endif

    OclThread(const rapidjson::Value &value);

    inline bool isAsm() const                               { return m_gcnAsm; }
    inline bool isDatasetHost() const                       { return m_datasetHost; }
    inline bool isValid() const                             { return m_intensity > 0; }
    inline const std::vector<int64_t> &threads() const      { return m_threads; }
    inline uint32_t bfactor() const                         { return m_bfactor; }
    inline uint32_t index() const                           { return m_index; }
    inline uint32_t intensity() const                       { return m_intensity; }
    inline uint32_t memChunk() const                        { return m_memChunk; }
    inline uint32_t stridedIndex() const                    { return m_stridedIndex; }
    inline uint32_t unrollFactor() const                    { return m_unrollFactor; }
    inline uint32_t worksize() const                        { return m_worksize; }

    inline bool operator!=(const OclThread &other) const    { return !isEqual(other); }
    inline bool operator==(const OclThread &other) const    { return isEqual(other); }

    bool isEqual(const OclThread &other) const;
    rapidjson::Value toJSON(rapidjson::Document &doc) const;

private:
    enum Fields {
        STRIDED_INDEX_FIELD,
        RANDOMX_FIELDS,
        KAWPOW_FIELDS,
        FIELD_MAX
    };

    inline void setIntensity(uint32_t intensity)            { m_intensity = intensity / m_worksize * m_worksize; }

    bool m_datasetHost              = false;
    bool m_gcnAsm                   = true;
    std::bitset<FIELD_MAX> m_fields = 1;
    std::vector<int64_t> m_threads;
    uint32_t m_bfactor              = 6;
    uint32_t m_index                = 0;
    uint32_t m_intensity            = 0;
    uint32_t m_memChunk             = 2;
    uint32_t m_stridedIndex         = 2;
    uint32_t m_unrollFactor         = 8;
    uint32_t m_worksize             = 0;
};


} /* namespace xmrig */


#endif /* XMRIG_OCLTHREAD_H */
