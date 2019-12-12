/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2019 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018-2019 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2019 XMRig       <support@xmrig.com>
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


#ifdef XMRIG_HWLOC_DEBUG
#   include <uv.h>
#endif


#include <algorithm>
#include <cmath>
#include <hwloc.h>


#if HWLOC_API_VERSION < 0x00010b00
#   define HWLOC_OBJ_PACKAGE HWLOC_OBJ_SOCKET
#   define HWLOC_OBJ_NUMANODE HWLOC_OBJ_NODE
#endif


#include "backend/cpu/platform/HwlocCpuInfo.h"
#include "base/io/log/Log.h"


namespace xmrig {


uint32_t HwlocCpuInfo::m_features = 0;


static inline bool isCacheObject(hwloc_obj_t obj)
{
#   if HWLOC_API_VERSION >= 0x20000
    return hwloc_obj_type_is_cache(obj->type);
#   else
    return obj->type == HWLOC_OBJ_CACHE;
#   endif
}


template <typename func>
static inline void findCache(hwloc_obj_t obj, unsigned min, unsigned max, func lambda)
{
    for (size_t i = 0; i < obj->arity; i++) {
        if (isCacheObject(obj->children[i])) {
            const unsigned depth = obj->children[i]->attr->cache.depth;
            if (depth < min || depth > max) {
                continue;
            }

            lambda(obj->children[i]);
        }

        findCache(obj->children[i], min, max, lambda);
    }
}


template <typename func>
static inline void findByType(hwloc_obj_t obj, hwloc_obj_type_t type, func lambda)
{
    for (size_t i = 0; i < obj->arity; i++) {
        if (obj->children[i]->type == type) {
            lambda(obj->children[i]);
        }
        else {
            findByType(obj->children[i], type, lambda);
        }
    }
}


static inline std::vector<hwloc_obj_t> findByType(hwloc_obj_t obj, hwloc_obj_type_t type)
{
    std::vector<hwloc_obj_t> out;
    findByType(obj, type, [&out](hwloc_obj_t found) { out.emplace_back(found); });

    return out;
}


static inline size_t countByType(hwloc_topology_t topology, hwloc_obj_type_t type)
{
    const int count = hwloc_get_nbobjs_by_type(topology, type);

    return count > 0 ? static_cast<size_t>(count) : 0;
}


static inline size_t countByType(hwloc_obj_t obj, hwloc_obj_type_t type)
{
    size_t count = 0;
    findByType(obj, type, [&count](hwloc_obj_t) { count++; });

    return count;
}


static inline bool isCacheExclusive(hwloc_obj_t obj)
{
    const char *value = hwloc_obj_get_info_by_name(obj, "Inclusive");
    return value == nullptr || value[0] != '1';
}


} // namespace xmrig


xmrig::HwlocCpuInfo::HwlocCpuInfo()
{
    m_threads = 0;

    hwloc_topology_init(&m_topology);
    hwloc_topology_load(m_topology);

#   ifdef XMRIG_HWLOC_DEBUG
#   if defined(UV_VERSION_HEX) && UV_VERSION_HEX >= 0x010c00
    {
        char env[520] = { 0 };
        size_t size   = sizeof(env);

        if (uv_os_getenv("HWLOC_XMLFILE", env, &size) == 0) {
            printf("use HWLOC XML file: \"%s\"\n", env);
        }
    }
#   endif

    const std::vector<hwloc_obj_t> packages = findByType(hwloc_get_root_obj(m_topology), HWLOC_OBJ_PACKAGE);
    if (!packages.empty()) {
        const char *value = hwloc_obj_get_info_by_name(packages[0], "CPUModel");
        if (value) {
            strncpy(m_brand, value, 64);
        }
    }
#   endif

    hwloc_obj_t root = hwloc_get_root_obj(m_topology);

#   if HWLOC_API_VERSION >= 0x00010b00
    const char *version = hwloc_obj_get_info_by_name(root, "hwlocVersion");
    if (version) {
        snprintf(m_backend, sizeof m_backend, "hwloc/%s", version);
    }
    else
#   endif
    {
        snprintf(m_backend, sizeof m_backend, "hwloc/%d.%d.%d",
                       (HWLOC_API_VERSION>>16)&0x000000ff,
                       (HWLOC_API_VERSION>>8 )&0x000000ff,
                       (HWLOC_API_VERSION    )&0x000000ff
               );
    }

    findCache(root, 2, 3, [this](hwloc_obj_t found) { this->m_cache[found->attr->cache.depth] += found->attr->cache.size; });

    m_threads   = countByType(m_topology, HWLOC_OBJ_PU);
    m_cores     = countByType(m_topology, HWLOC_OBJ_CORE);
    m_nodes     = std::max(hwloc_bitmap_weight(hwloc_topology_get_complete_nodeset(m_topology)), 1);
    m_packages  = countByType(m_topology, HWLOC_OBJ_PACKAGE);

    if (m_nodes > 1) {
        if (hwloc_topology_get_support(m_topology)->membind->set_thisthread_membind) {
            m_features |= SET_THISTHREAD_MEMBIND;
        }

        m_nodeset.reserve(m_nodes);
        hwloc_obj_t node = nullptr;

        while ((node = hwloc_get_next_obj_by_type(m_topology, HWLOC_OBJ_NUMANODE, node)) != nullptr) {
            m_nodeset.emplace_back(node->os_index);
        }
    }
}


xmrig::HwlocCpuInfo::~HwlocCpuInfo()
{
    hwloc_topology_destroy(m_topology);
}


bool xmrig::HwlocCpuInfo::membind(hwloc_const_bitmap_t nodeset)
{
    if (!hwloc_topology_get_support(m_topology)->membind->set_thisthread_membind) {
        return false;
    }

#   if HWLOC_API_VERSION >= 0x20000
    return hwloc_set_membind(m_topology, nodeset, HWLOC_MEMBIND_BIND, HWLOC_MEMBIND_THREAD | HWLOC_MEMBIND_BYNODESET) >= 0;
#   else
    return hwloc_set_membind_nodeset(m_topology, nodeset, HWLOC_MEMBIND_BIND, HWLOC_MEMBIND_THREAD) >= 0;
#   endif
}


xmrig::CpuThreads xmrig::HwlocCpuInfo::threads(const Algorithm &algorithm, uint32_t limit) const
{
    if (L2() == 0 && L3() == 0) {
        return BasicCpuInfo::threads(algorithm, limit);
    }

    const unsigned depth = L3() > 0 ? 3 : 2;

    CpuThreads threads;
    threads.reserve(m_threads);

    std::vector<hwloc_obj_t> caches;
    caches.reserve(16);

    findCache(hwloc_get_root_obj(m_topology), depth, depth, [&caches](hwloc_obj_t found) { caches.emplace_back(found); });

    if (limit > 0 && limit < 100 && !caches.empty()) {
        const double maxTotalThreads = round(m_threads * (limit / 100.0));
        const auto maxPerCache       = std::max(static_cast<int>(round(maxTotalThreads / caches.size())), 1);
        int remaining                = std::max(static_cast<int>(maxTotalThreads), 1);

        for (hwloc_obj_t cache : caches) {
            processTopLevelCache(cache, algorithm, threads, std::min(maxPerCache, remaining));

            remaining -= maxPerCache;
            if (remaining <= 0) {
                break;
            }
        }
    }
    else {
        for (hwloc_obj_t cache : caches) {
            processTopLevelCache(cache, algorithm, threads, 0);
        }
    }

    if (threads.isEmpty()) {
        LOG_WARN("hwloc auto configuration for algorithm \"%s\" failed.", algorithm.shortName());

        return BasicCpuInfo::threads(algorithm, limit);
    }

    return threads;
}


void xmrig::HwlocCpuInfo::processTopLevelCache(hwloc_obj_t cache, const Algorithm &algorithm, CpuThreads &threads, size_t limit) const
{
    constexpr size_t oneMiB = 1024U * 1024U;

    size_t PUs = countByType(cache, HWLOC_OBJ_PU);
    if (PUs == 0) {
        return;
    }

    std::vector<hwloc_obj_t> cores;
    cores.reserve(m_cores);
    findByType(cache, HWLOC_OBJ_CORE, [&cores](hwloc_obj_t found) { cores.emplace_back(found); });

    size_t L3               = cache->attr->cache.size;
    const bool L3_exclusive = isCacheExclusive(cache);
    size_t L2               = 0;
    int L2_associativity    = 0;
    size_t extra            = 0;
    const size_t scratchpad = algorithm.l3();
    uint32_t intensity      = algorithm.maxIntensity() == 1 ? 0 : 1;

    if (cache->attr->cache.depth == 3) {
        for (size_t i = 0; i < cache->arity; ++i) {
            hwloc_obj_t l2 = cache->children[i];
            if (!isCacheObject(l2) || l2->attr == nullptr) {
                continue;
            }

            L2 += l2->attr->cache.size;
            L2_associativity = l2->attr->cache.associativity;

            if (L3_exclusive && l2->attr->cache.size >= scratchpad) {
                extra += scratchpad;
            }
        }
    }

    if (scratchpad == 2 * oneMiB) {
        if (L2 && (cores.size() * oneMiB) == L2 && L2_associativity == 16 && L3 >= L2) {
            L3    = L2;
            extra = L2;
        }
    }

    size_t cacheHashes = ((L3 + extra) + (scratchpad / 2)) / scratchpad;

#   ifdef XMRIG_ALGO_CN_PICO
    if (algorithm == Algorithm::CN_PICO_0 && (cacheHashes / PUs) >= 2) {
        intensity = 2;
    }
#   endif

#   ifdef XMRIG_ALGO_CN_GPU
    if (algorithm == Algorithm::CN_GPU) {
        cacheHashes = PUs;
    }
#   endif

#   ifdef XMRIG_ALGO_RANDOMX
    if (extra == 0 && algorithm.l2() > 0) {
        cacheHashes = std::min<size_t>(std::max<size_t>(L2 / algorithm.l2(), cores.size()), cacheHashes);
    }
#   endif

    if (limit > 0) {
        cacheHashes = std::min(cacheHashes, limit);
    }

    if (cacheHashes >= PUs) {
        for (hwloc_obj_t core : cores) {
            const std::vector<hwloc_obj_t> units = findByType(core, HWLOC_OBJ_PU);
            for (hwloc_obj_t pu : units) {
                threads.add(pu->os_index, intensity);
            }
        }

        return;
    }

    size_t pu_id = 0;
    while (cacheHashes > 0 && PUs > 0) {
        bool allocated_pu = false;

        for (hwloc_obj_t core : cores) {
            const std::vector<hwloc_obj_t> units = findByType(core, HWLOC_OBJ_PU);
            if (units.size() <= pu_id) {
                continue;
            }

            cacheHashes--;
            PUs--;

            allocated_pu = true;
            threads.add(units[pu_id]->os_index, intensity);

            if (cacheHashes == 0) {
                break;
            }
        }

        if (!allocated_pu) {
            break;
        }

        pu_id++;
    }
}
