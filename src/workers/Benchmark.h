/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2016-2018 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
 * Copyright 2018 MoneroOcean      <https://github.com/MoneroOcean>, <support@moneroocean.stream>
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

#pragma once

#include <stdint.h>

#include "common/xmrig.h"
#include "interfaces/IJobResultListener.h"
#include "core/Controller.h"

class Benchmark : public IJobResultListener {
    bool m_shouldSaveConfig; // should save config after all benchmark rounds
    xmrig::PerfAlgo m_pa;    // current perf algo we benchmark
    uint64_t m_hash_count;   // number of hashes calculated for current perf algo
    uint64_t m_time_start;   // time of measurements start for current perf algo (in ms)
    xmrig::Controller* m_controller; // to get access to config and network

    uint64_t get_now() const; // get current time in ms

    void onJobResult(const JobResult&) override; // onJobResult is called after each computed benchmark hash

    public:
        Benchmark() : m_shouldSaveConfig(false) {}
        virtual ~Benchmark() {}

        void set_controller(xmrig::Controller* controller) { m_controller = controller; }
        void should_save_config() { m_shouldSaveConfig = true; }
        void start_perf_bench(const xmrig::PerfAlgo); // start benchmark for specified perf algo
};
