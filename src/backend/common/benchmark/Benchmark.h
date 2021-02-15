/* xmlcore
 * Copyright (c) 2018-2020 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2020 xmlcore       <https://github.com/xmlcore>, <support@xmlcore.com>
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

#ifndef xmlcore_BENCHMARK_H
#define xmlcore_BENCHMARK_H


#include "base/tools/Object.h"


namespace xmlcore {


class IBackend;


class Benchmark
{
public:
    xmlcore_DISABLE_COPY_MOVE_DEFAULT(Benchmark)

    Benchmark(size_t workers, const IBackend *backend);
    ~Benchmark() = default;

    bool finish(uint64_t totalHashCount);
    void printProgress() const;
    void start();

private:
    const IBackend *m_backend;
    const size_t m_workers;
    uint64_t m_current          = 0;
    uint64_t m_startTime        = 0;
};


} // namespace xmlcore


#endif /* xmlcore_BENCHMARK_H */
