/* XMRig
 * Copyright 2018-2020 MoneroOcean <https://github.com/MoneroOcean>, <support@moneroocean.stream>
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

#include <set>
#include <map>
#include "net/interfaces/IJobResultListener.h"
#include "base/crypto/Algorithm.h"
#include "base/net/stratum/Job.h"
#include "rapidjson/fwd.h"

#include <memory>

namespace xmrig {

class Controller;
class Miner;
class Job;

class MoBenchmark : public IJobResultListener {

        const Algorithm::Id bench_algos[15] = {
#           ifdef XMRIG_ALGO_GHOSTRIDER
            Algorithm::FLEX_KCN,
            Algorithm::GHOSTRIDER_RTM,
#           endif
            Algorithm::CN_R,
#           ifdef XMRIG_ALGO_CN_LITE
            Algorithm::CN_LITE_1,
#           endif
#           ifdef XMRIG_ALGO_CN_HEAVY
            Algorithm::CN_HEAVY_XHV,
#           endif
#           ifdef XMRIG_ALGO_CN_PICO
            Algorithm::CN_PICO_0,
#           endif
            Algorithm::CN_CCX,
#           ifdef XMRIG_ALGO_CN_GPU
            Algorithm::CN_GPU,
#           endif
#           ifdef XMRIG_ALGO_ARGON2
            Algorithm::AR2_CHUKWA_V2,
#           endif
#           ifdef XMRIG_ALGO_KAWPOW
            Algorithm::KAWPOW_RVN,
#           endif
#           ifdef XMRIG_ALGO_RANDOMX
            Algorithm::RX_0,
            Algorithm::RX_GRAFT,
            Algorithm::RX_ARQ,
            Algorithm::RX_XLA,
#           endif
            Algorithm::INVALID
        };

        Job m_bench_job;

        Controller *m_controller;          // to get access to config and network
        bool m_isNewBenchRun;              // true if benchmark is need to be executed or was executed
        uint64_t m_bench_algo;             // current perf algo number we benchmark (in bench_algos array)
        uint64_t m_hash_count;             // number of hashes calculated for current perf algo
        uint64_t m_time_start;             // time of the first resultt for current perf algo (in ms)
        uint64_t m_bench_start;            // time of measurements start for current perf algo (in ms) after all backends are started
        unsigned m_enabled_backend_count;  // number of active miner backends
        std::set<uint32_t> m_backends_started; // id of backend started for benchmark

        uint64_t get_now() const;                       // get current time in ms
        double get_algo_perf(Algorithm::Id algo) const; // get algo perf based on algo_perf known perf numbers
        void start();                                   // start benchmark for m_bench_algo number
        void finish();                                  // end of benchmarks, switch to jobs from the pool (network), fill algo_perf
        void onJobResult(const JobResult&) override;    // onJobResult is called after each computed benchmark hash
        void run_next_bench_algo();                     // run next bench algo or finish benchmark for the last one

    public:
        MoBenchmark();
        virtual ~MoBenchmark();

        void set_controller(std::shared_ptr<Controller> controller) { m_controller = controller.get(); }

        void start_perf(); // start benchmarks
        void flush_perf();

        bool isNewBenchRun() const { return m_isNewBenchRun; }
        mutable std::map<Algorithm::Id, double> algo_perf;

        rapidjson::Value toJSON(rapidjson::Document &doc) const;
        void read(const rapidjson::Value &value);
};

} // namespace xmrig
