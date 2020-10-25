/* XMRig
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

#ifndef XMRIG_TAGS_H
#define XMRIG_TAGS_H


#include <cstddef>
#include <cstdint>


namespace xmrig {


class Tags
{
public:
    static const char *config();
    static const char *network();
    static const char *signal();

#   ifdef XMRIG_MINER_PROJECT
    static const char *cpu();
    static const char *miner();
#   ifdef XMRIG_ALGO_RANDOMX
    static const char *randomx();
#   endif
#   ifdef XMRIG_FEATURE_BENCHMARK
    static const char *bench();
#   endif
#   endif

#   ifdef XMRIG_PROXY_PROJECT
    static const char *proxy();
#   endif

#   ifdef XMRIG_FEATURE_CUDA
    static const char *nvidia();
#   endif

#   ifdef XMRIG_FEATURE_OPENCL
    static const char *opencl();
#   endif

#   ifdef XMRIG_FEATURE_MO_BENCHMARK
    static const char *benchmark();
#   endif

#   ifdef XMRIG_FEATURE_PROFILING
    static const char* profiler();
#   endif
};


} /* namespace xmrig */


#endif /* XMRIG_TAGS_H */
