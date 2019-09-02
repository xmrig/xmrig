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


#include <algorithm>


#include "backend/opencl/OclThread.h"
#include "base/io/json/Json.h"
#include "rapidjson/document.h"


namespace xmrig {

static const char *kAffinity     = "affinity";
static const char *kCompMode     = "comp_mode";
static const char *kIndex        = "index";
static const char *kIntensity    = "intensity";
static const char *kMemChunk     = "mem_chunk";
static const char *kStridedIndex = "strided_index";
static const char *kUnroll       = "unroll";
static const char *kWorksize     = "worksize";

#ifdef XMRIG_ALGO_RANDOMX
static const char *kBFactor      = "bfactor";
static const char *kGCNAsm       = "gcn_asm";
static const char* kDatasetHost  = "dataset_host";
#endif

} // namespace xmrig


xmrig::OclThread::OclThread(const rapidjson::Value &value)
{
    m_index     = Json::getUint(value, kIndex);
    m_intensity = Json::getUint(value, kIntensity);
    m_worksize  = Json::getUint(value, kWorksize);
    m_affinity  = Json::getInt64(value, kAffinity, -1);
    m_memChunk  = std::min(Json::getUint(value, kMemChunk, m_memChunk), 18u);
    m_compMode  = Json::getBool(value, kCompMode, m_compMode);

    setUnrollFactor(Json::getUint(value, kUnroll, m_unrollFactor));

#   ifdef XMRIG_ALGO_RANDOMX
    m_bfactor     = Json::getUint(value, kBFactor, 6);
    m_gcnAsm      = Json::getUint(value, kGCNAsm, m_gcnAsm);
    m_datasetHost = Json::getInt(value, kDatasetHost, m_datasetHost);
#   endif

    const rapidjson::Value &stridedIndex = Json::getValue(value, kStridedIndex);
    if (stridedIndex.IsBool()) {
        m_stridedIndex = stridedIndex.GetBool() ? 1 : 0;
    }
    else if (stridedIndex.IsUint()) {
        m_stridedIndex = std::min(stridedIndex.GetUint(), 2u);
    }
}


bool xmrig::OclThread::isEqual(const OclThread &other) const
{
    return other.m_compMode     == m_compMode &&
           other.m_affinity     == m_affinity &&
           other.m_bfactor      == m_bfactor &&
           other.m_datasetHost  == m_datasetHost &&
           other.m_gcnAsm       == m_gcnAsm &&
           other.m_index        == m_index &&
           other.m_intensity    == m_intensity &&
           other.m_memChunk     == m_memChunk &&
           other.m_stridedIndex == m_stridedIndex &&
           other.m_unrollFactor == m_unrollFactor &&
           other.m_worksize     == m_worksize;
}


rapidjson::Value xmrig::OclThread::toJSON(rapidjson::Document &doc) const
{
    using namespace rapidjson;
    auto &allocator = doc.GetAllocator();

    Value out(kObjectType);

    out.AddMember(StringRef(kIndex),        index(), allocator);
    out.AddMember(StringRef(kIntensity),    intensity(), allocator);
    out.AddMember(StringRef(kWorksize),     worksize(), allocator);
    out.AddMember(StringRef(kStridedIndex), stridedIndex(), allocator);

    if (stridedIndex() == 2) {
        out.AddMember(StringRef(kMemChunk), memChunk(), allocator);
    }

    out.AddMember(StringRef(kUnroll),       unrollFactor(), allocator);
    out.AddMember(StringRef(kAffinity),     affinity(), allocator);

    if (isCompMode()) {
        out.AddMember(StringRef(kCompMode), true, allocator);
    }

#   ifdef XMRIG_ALGO_RANDOMX
    if (m_datasetHost != -1) {
        out.AddMember(StringRef(kBFactor),      bfactor(), allocator);
        out.AddMember(StringRef(kGCNAsm),       gcnAsm(), allocator);
        out.AddMember(StringRef(kDatasetHost),  datasetHost(), allocator);
    }
#   endif

    return out;
}


void xmrig::OclThread::setUnrollFactor(uint32_t unrollFactor)
{
    m_unrollFactor = unrollFactor == 0 ? 1 : std::min(unrollFactor, 128u);
}
