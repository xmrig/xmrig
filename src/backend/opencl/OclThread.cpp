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


#include "backend/opencl/OclThread.h"
#include "3rdparty/rapidjson/document.h"
#include "base/io/json/Json.h"


#include <algorithm>


namespace xmrig {

static const char *kIndex        = "index";
static const char *kIntensity    = "intensity";
static const char *kStridedIndex = "strided_index";
static const char *kThreads      = "threads";
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
    if (!value.IsObject()) {
        return;
    }

    m_index         = Json::getUint(value, kIndex);
    m_worksize      = std::max(std::min(Json::getUint(value, kWorksize), 512u), 1u);
    m_unrollFactor  = std::max(std::min(Json::getUint(value, kUnroll, m_unrollFactor), 128u), 1u);

    setIntensity(Json::getUint(value, kIntensity));

    const auto &si = Json::getArray(value, kStridedIndex);
    if (si.IsArray() && si.Size() >= 2) {
        m_stridedIndex = std::min(si[0].GetUint(), 2u);
        m_memChunk     = std::min(si[1].GetUint(), 18u);
    }
    else {
        m_stridedIndex = 0;
        m_memChunk     = 0;
        m_fields.set(STRIDED_INDEX_FIELD, false);
    }

    const auto &threads = Json::getArray(value, kThreads);
    if (threads.IsArray()) {
        m_threads.reserve(threads.Size());

        for (const auto &affinity : threads.GetArray()) {
            m_threads.emplace_back(affinity.GetInt64());
        }
    }

    if (m_threads.empty()) {
        m_threads.emplace_back(-1);
    }

#   ifdef XMRIG_ALGO_RANDOMX
    const auto &gcnAsm = Json::getValue(value, kGCNAsm);
    if (gcnAsm.IsBool()) {
        m_fields.set(RANDOMX_FIELDS, true);

        m_gcnAsm      = gcnAsm.GetBool();
        m_bfactor     = Json::getUint(value, kBFactor, m_bfactor);
        m_datasetHost = Json::getBool(value, kDatasetHost, m_datasetHost);
    }
#   endif
}


bool xmrig::OclThread::isEqual(const OclThread &other) const
{
    return other.m_threads.size() == m_threads.size() &&
           std::equal(m_threads.begin(), m_threads.end(), other.m_threads.begin()) &&
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
    if (!m_fields.test(ASTROBWT_FIELDS)) {
        out.AddMember(StringRef(kWorksize), worksize(), allocator);
    }

    if (m_fields.test(STRIDED_INDEX_FIELD)) {
        Value si(kArrayType);
        si.Reserve(2, allocator);
        si.PushBack(stridedIndex(), allocator);
        si.PushBack(memChunk(), allocator);
        out.AddMember(StringRef(kStridedIndex), si, allocator);
    }

    Value threads(kArrayType);
    threads.Reserve(m_threads.size(), allocator);

    for (auto thread : m_threads) {
        threads.PushBack(thread, allocator);
    }

    out.AddMember(StringRef(kThreads), threads, allocator);

    if (m_fields.test(RANDOMX_FIELDS)) {
#       ifdef XMRIG_ALGO_RANDOMX
        out.AddMember(StringRef(kBFactor),      bfactor(), allocator);
        out.AddMember(StringRef(kGCNAsm),       isAsm(), allocator);
        out.AddMember(StringRef(kDatasetHost),  isDatasetHost(), allocator);
#       endif
    }
    else if (!m_fields.test(ASTROBWT_FIELDS) && !m_fields.test(KAWPOW_FIELDS)) {
        out.AddMember(StringRef(kUnroll), unrollFactor(), allocator);
    }

    return out;
}
