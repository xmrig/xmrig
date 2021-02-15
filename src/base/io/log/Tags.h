/* xmlcore
 * Copyright 2018-2020 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2020 xmlcore       <https://github.com/xmlcore>, <support@xmlcore.com>
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

#ifndef xmlcore_TAGS_H
#define xmlcore_TAGS_H


#include <cstddef>
#include <cstdint>


namespace xmlcore {


class Tags
{
public:
    static const char *config();
    static const char *network();
    static const char *signal();

#   ifdef xmlcore_MINER_PROJECT
    static const char *cpu();
    static const char *miner();
#   ifdef xmlcore_ALGO_RANDOMX
    static const char *randomx();
#   endif
#   ifdef xmlcore_FEATURE_BENCHMARK
    static const char *bench();
#   endif
#   endif

#   ifdef xmlcore_PROXY_PROJECT
    static const char *proxy();
#   endif

#   ifdef xmlcore_FEATURE_CUDA
    static const char *nvidia();
#   endif

#   ifdef xmlcore_FEATURE_OPENCL
    static const char *opencl();
#   endif

#   ifdef xmlcore_FEATURE_PROFILING
    static const char* profiler();
#   endif
};


} /* namespace xmlcore */


#endif /* xmlcore_TAGS_H */
