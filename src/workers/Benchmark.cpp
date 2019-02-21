/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2016-2018 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
 * Copyright 2018-2019 MoneroOcean <https://github.com/MoneroOcean>, <support@moneroocean.stream>
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

#include "workers/Benchmark.h"
#include "workers/Workers.h"
#include "core/Config.h"
#include "net/Network.h"
#include "common/log/Log.h"
#include <chrono>

// start performance measurements for specified perf algo
void Benchmark::start_perf_bench(const xmrig::PerfAlgo pa) {
    Workers::switch_algo(xmrig::Algorithm(pa)); // switch workers to new algo (Algo part)

    // prepare test job for benchmark runs
    Job job;
    job.setPoolId(-100); // to make sure we can detect benchmark jobs
    job.setId(xmrig::Algorithm::perfAlgoName(pa)); // need to set different id so that workers will see job change
    const static uint8_t test_input[76] = {
        0x99, // 0x99 here to trigger all future algo versions for auto veriant detection based on block version
        0x05, 0xA0, 0xDB, 0xD6, 0xBF, 0x05, 0xCF, 0x16, 0xE5, 0x03, 0xF3, 0xA6, 0x6F, 0x78, 0x00,
        0x7C, 0xBF, 0x34, 0x14, 0x43, 0x32, 0xEC, 0xBF, 0xC2, 0x2E, 0xD9, 0x5C, 0x87, 0x00, 0x38, 0x3B,
        0x30, 0x9A, 0xCE, 0x19, 0x23, 0xA0, 0x96, 0x4B, 0x00, 0x00, 0x00, 0x08, 0xBA, 0x93, 0x9A, 0x62,
        0x72, 0x4C, 0x0D, 0x75, 0x81, 0xFC, 0xE5, 0x76, 0x1E, 0x9D, 0x8A, 0x0E, 0x6A, 0x1C, 0x3F, 0x92,
        0x4F, 0xDD, 0x84, 0x93, 0xD1, 0x11, 0x56, 0x49, 0xC0, 0x5E, 0xB6, 0x01,
    };
    job.setRawBlob(test_input, 76);
    job.setTarget("FFFFFFFFFFFFFF20"); // set difficulty to 8 cause onJobResult after every 8-th computed hash
    job.setAlgorithm(xmrig::Algorithm(pa)); // set job algo (for Variant part)

    m_pa = pa; // current perf algo
    m_hash_count = 0; // number of hashes calculated for current perf algo
    m_time_start = 0; // init time of measurements start (in ms) during the first onJobResult
    Workers::setJob(job, false); // set job for workers to compute
}

void Benchmark::onJobResult(const xmrig::JobResult& result) {
    if (result.poolId != -100) { // switch to network pool jobs
        Workers::setListener(m_controller->network());
        static_cast<xmrig::IJobResultListener*>(m_controller->network())->onJobResult(result);
        return;
    }
    // ignore benchmark results for other perf algo
    if (m_pa == xmrig::PA_INVALID || result.jobId != xmrig::Id(xmrig::Algorithm::perfAlgoName(m_pa))) return;
    ++ m_hash_count;
    const uint64_t now = get_now();
    if (!m_time_start) m_time_start = now; // time of measurements start (in ms)
    else if (now - m_time_start > static_cast<unsigned>(m_controller->config()->calibrateAlgoTime())*1000) { // end of becnhmark round for m_pa
        const float hashrate = static_cast<float>(m_hash_count) * result.diff / (now - m_time_start) * 1000.0f;
        m_controller->config()->set_algo_perf(m_pa, hashrate); // store hashrate result
        Log::i()->text(m_controller->config()->isColors()
            ? GREEN_BOLD(" ===> ") CYAN_BOLD("%s") WHITE_BOLD(" hashrate: ") CYAN_BOLD("%f")
            : " ===> %s hasrate: %f",
            xmrig::Algorithm::perfAlgoName(m_pa),
            hashrate
        );
        const xmrig::PerfAlgo next_pa = static_cast<xmrig::PerfAlgo>(m_pa + 1); // compute next perf algo to benchmark
        if (next_pa != xmrig::PerfAlgo::PA_MAX) {
            start_perf_bench(next_pa);
        } else { // end of benchmarks and switching to jobs from the pool (network)
            m_pa = xmrig::PA_INVALID;
            if (m_shouldSaveConfig) m_controller->config()->save(); // save config with measured algo-perf
            Workers::pause(); // do not compute anything before job from the pool
            m_controller->network()->connect();
        }
    }
}

uint64_t Benchmark::get_now() const { // get current time in ms
    using namespace std::chrono;
    return time_point_cast<milliseconds>(high_resolution_clock::now()).time_since_epoch().count();
}
