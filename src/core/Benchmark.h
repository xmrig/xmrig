/* XMRig
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

#pragma once

#include "net/interfaces/IJobResultListener.h"
#include "crypto/common/Algorithm.h"
#include "rapidjson/fwd.h"

namespace xmrig {

class Controller;
class Miner;
class Job;

class Benchmark : public IJobResultListener {

        enum BenchAlgo : int {
            RX_0,          // "rx/0"             RandomX (reference configuration).
            RX_WOW,        // "rx/wow"           RandomWOW (Wownero).
            CN_R,          // "cn/r"             CryptoNightR (Monero's variant 4).
            CN_GPU,        // "cn/gpu"           CryptoNight-GPU (Ryo).
            CN_LITE_1,     // "cn-lite/1"        CryptoNight-Lite variant 1.
            CN_HEAVY_TUBE, // "cn-heavy/tube"    CryptoNight-Heavy (modified, TUBE only).
            CN_PICO_0,     // "cn-pico"          CryptoNight Turtle (TRTL)
            MAX,
            MIN = 0,
            INVALID = -1,
        };

        const Algorithm::Id ba2a[BenchAlgo::MAX] = {
            Algorithm::RX_0,
            Algorithm::RX_WOW,
            Algorithm::CN_R,
            Algorithm::CN_GPU,
            Algorithm::CN_LITE_1,
            Algorithm::CN_HEAVY_TUBE,
            Algorithm::CN_PICO_0,
        };

        Job* m_bench_job[BenchAlgo::MAX];
        float m_bench_algo_perf[BenchAlgo::MAX];

        Controller* m_controller;          // to get access to config and network
        bool m_isNewBenchRun;
        Benchmark::BenchAlgo m_bench_algo; // current perf algo we benchmark
        uint64_t m_hash_count;             // number of hashes calculated for current perf algo
        uint64_t m_time_start;             // time of measurements start for current perf algo (in ms)

        uint64_t get_now() const;                      // get current time in ms
        float get_algo_perf(Algorithm::Id algo) const; // get algo perf based on m_bench_algo_perf
        void start(const Benchmark::BenchAlgo);        // start benchmark for specified perf algo
        void finish();                                 // end of benchmarks, switch to jobs from the pool (network), fill algo_perf
        void onJobResult(const JobResult&) override;   // onJobResult is called after each computed benchmark hash

    public:
        Benchmark();
        virtual ~Benchmark();

        void set_controller(Controller* controller) { m_controller = controller; }

        void start(); // start benchmarks

        bool isNewBenchRun() const { return m_isNewBenchRun; }
        float algo_perf[Algorithm::MAX];

        rapidjson::Value toJSON(rapidjson::Document &doc) const;
        void read(const rapidjson::Value &value);
};

} // namespace xmrig
