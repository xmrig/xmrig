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

#include "core/MoBenchmark.h"
#include "3rdparty/rapidjson/document.h"
#include "backend/common/Hashrate.h"
#include "backend/common/interfaces/IBackend.h"
#include "backend/common/Tags.h"
#include "base/io/log/Log.h"
#include "base/io/log/Tags.h"
#include "core/config/Config.h"
#include "core/Controller.h"
#include "core/Miner.h"
#include "net/JobResult.h"
#include "net/JobResults.h"
#include "net/Network.h"

#include <chrono>

namespace xmrig {

MoBenchmark::MoBenchmark() : m_controller(nullptr), m_isNewBenchRun(true) {}

MoBenchmark::~MoBenchmark() {}

// start performance measurements from the first bench_algo
void MoBenchmark::start_perf() {
    JobResults::setListener(this, m_controller->config()->cpu().isHwAES()); // register benchmark as job result listener to compute hashrates there
    // write text before first benchmark round
    LOG_INFO("%s " BRIGHT_BLACK_BG(CYAN_BOLD_S " STARTING ALGO PERFORMANCE CALIBRATION (with " MAGENTA_BOLD_S "%i" CYAN_BOLD_S " seconds round) "), Tags::benchmark(), m_controller->config()->benchAlgoTime());
    // start benchmarking from first PerfAlgo in the list
    m_bench_algo = 0;
    start();
    m_isNewBenchRun = true; // need to save it to true to save config after benchmark
}

// end of benchmarks, switch to jobs from the pool (network), fill algo_perf
void MoBenchmark::finish() {
    for (const Algorithm::Id algo : Algorithm::all([this](const Algorithm &algo) { return true; })) {
        if (algo_perf[algo] == 0.0f) algo_perf[algo] = get_algo_perf(algo);
    }
    LOG_INFO("%s " BRIGHT_BLACK_BG(CYAN_BOLD_S " ALGO PERFORMANCE CALIBRATION COMPLETE "), Tags::benchmark());
    m_controller->miner()->pause(); // do not compute anything before job from the pool
    JobResults::stop();
    JobResults::setListener(m_controller->network(), m_controller->config()->cpu().isHwAES());
    m_controller->start();
}

rapidjson::Value MoBenchmark::toJSON(rapidjson::Document &doc) const
{
    using namespace rapidjson;
    auto &allocator = doc.GetAllocator();

    Value obj(kObjectType);

    for (const Algorithm a : Algorithm::all()) {
        if (algo_perf[a.id()] == 0.0f) continue;
        obj.AddMember(StringRef(a.name()), algo_perf[a.id()], allocator);
    }

    return obj;
}

void MoBenchmark::flush_perf() {
   for (const Algorithm::Id algo : Algorithm::all()) algo_perf[algo] = 0.0f;
}

void MoBenchmark::read(const rapidjson::Value &value)
{
    flush_perf();
    if (value.IsObject()) {
        for (auto &member : value.GetObject()) {
            const Algorithm algo(member.name.GetString());
            if (!algo.isValid()) {
                LOG_INFO("%s " BRIGHT_BLACK_BG(MAGENTA_BOLD_S " Ignoring wrong name for algo-perf[%s] "), Tags::benchmark(), member.name.GetString());
                continue;
            }
            if (member.value.IsDouble()) {
                algo_perf[algo.id()] = member.value.GetDouble();
                continue;
            }
            if (member.value.IsInt()) {
                algo_perf[algo.id()] = member.value.GetInt();
                continue;
            }
            LOG_INFO("%s " BRIGHT_BLACK_BG(MAGENTA_BOLD_S " Ignoring wrong value for algo-perf[%s] "), Tags::benchmark(), member.name.GetString());
        }
    }
    m_isNewBenchRun = false;
    for (int i = 0; bench_algos[i] != Algorithm::INVALID; ++ i)
        if (algo_perf[bench_algos[i]] == 0.0f) {
            m_isNewBenchRun = true;
            return;
        }
}

double MoBenchmark::get_algo_perf(Algorithm::Id algo) const {
    switch (algo) {
        case Algorithm::CN_0:            return algo_perf[Algorithm::CN_CCX] / 2;
        case Algorithm::CN_1:            return algo_perf[Algorithm::CN_R];
        case Algorithm::CN_2:            return algo_perf[Algorithm::CN_R];
        case Algorithm::CN_RTO:          return algo_perf[Algorithm::CN_R];
        case Algorithm::CN_XAO:          return algo_perf[Algorithm::CN_R];
        case Algorithm::CN_FAST:         return algo_perf[Algorithm::CN_R] * 2;
        case Algorithm::CN_HALF:         return algo_perf[Algorithm::CN_R] * 2;
        case Algorithm::CN_RWZ:          return algo_perf[Algorithm::CN_R] / 3 * 4;
        case Algorithm::CN_ZLS:          return algo_perf[Algorithm::CN_R] / 3 * 4;
        case Algorithm::CN_DOUBLE:       return algo_perf[Algorithm::CN_R] / 2;
#       ifdef XMRIG_ALGO_CN_LITE
        case Algorithm::CN_LITE_0:       return algo_perf[Algorithm::CN_LITE_1];
#       endif
#       ifdef XMRIG_ALGO_CN_PICO
        case Algorithm::CN_PICO_TLO:     return algo_perf[Algorithm::CN_PICO_0];
#       endif
#       ifdef XMRIG_ALGO_RANDOMX
        case Algorithm::RX_SFX:          return algo_perf[Algorithm::RX_0];
        case Algorithm::RX_XEQ:          return algo_perf[Algorithm::RX_ARQ];
#       endif
        default:                         return algo_perf[algo];
    }
}

// start performance measurements for bench_algos[m_bench_algo]
void MoBenchmark::start() {
    const Algorithm algo(bench_algos[m_bench_algo]);
    if (algo_perf[algo.id()] > 0.0f) {
        run_next_bench_algo();
        return;
    }
    // calculate number of active miner backends in m_enabled_backend_count
    m_enabled_backend_count = 0;
    for (auto backend : m_controller->miner()->backends()) if (backend->isEnabled() && backend->isEnabled(algo)) ++ m_enabled_backend_count;
    if (m_enabled_backend_count == 0) {
        LOG_INFO("%s " BRIGHT_BLACK_BG(WHITE_BOLD_S " Algo " MAGENTA_BOLD_S "%s" WHITE_BOLD_S " is skipped due to a disabled backend"), Tags::benchmark(), algo.name());
        algo_perf[algo.id()] = -1.0f; // to avoid re-running benchmark next time
        run_next_bench_algo();
        return;
    }
    LOG_INFO("%s " BRIGHT_BLACK_BG(WHITE_BOLD_S " Algo " MAGENTA_BOLD_S "%s" WHITE_BOLD_S " Preparation "), Tags::benchmark(), algo.name());
    // prepare test job for benchmark runs ("benchmark" client id is to make sure we can detect benchmark jobs)
    m_bench_job = Job(false, Algorithm(bench_algos[m_bench_algo]), "benchmark");
    m_bench_job.setId(algo.name()); // need to set different id so that workers will see job change
    switch (algo.id()) {
#     ifdef XMRIG_ALGO_KAWPOW
      case Algorithm::KAWPOW_RVN:
          m_bench_job.setBlob("4c38e8a5f7b2944d1e4274635d828519b97bc64a1f1c7896ecdbb139989aa0e80000000000000000000000000000000000000000000000000000000000000000000000000000000000000000");
          m_bench_job.setDiff(Job::toDiff(strtoull("000000639c000000", nullptr, 16)));
          m_bench_job.setHeight(1500000);
          break;
#     endif

#     ifdef XMRIG_ALGO_GHOSTRIDER
      case Algorithm::GHOSTRIDER_RTM:
      case Algorithm::FLEX_KCN:
          m_bench_job.setBlob("000000208c246d0b90c3b389c4086e8b672ee040d64db5b9648527133e217fbfa48da64c0f3c0a0b0e8350800568b40fbb323ac3ccdf2965de51b9aaeb939b4f11ff81c49b74a16156ff251c00000000");
          m_bench_job.setDiff(1000);
          break;
#     endif

      default:
          // 99 here to trigger all future bench_algo versions for auto veriant detection based on block version
          m_bench_job.setBlob("9905A0DBD6BF05CF16E503F3A66F78007CBF34144332ECBFC22ED95C8700383B309ACE1923A0964B00000008BA939A62724C0D7581FCE5761E9D8A0E6A1C3F924FDD8493D1115649C05EB601");
          m_bench_job.setTarget("FFFFFFFFFFFFFF20"); // set difficulty to 8 cause onJobResult after every 8-th computed hash
          m_bench_job.setHeight(1000);
          m_bench_job.setSeedHash("0000000000000000000000000000000000000000000000000000000000000001");
    }
    m_hash_count  = 0;          // number of hashes calculated for current perf bench_algo
    m_time_start  = 0;          // init time of the first result (in ms) during the first onJobResult
    m_bench_start = 0;          // init time of measurements start (in ms) during the first onJobResult
    m_backends_started.clear();
    m_controller->miner()->setJob(m_bench_job, false); // set job for workers to compute
}

// run next bench algo or finish benchmark for the last one
void MoBenchmark::run_next_bench_algo() {
    ++ m_bench_algo;
    if (bench_algos[m_bench_algo] != Algorithm::INVALID) {
        start();
    } else {
        finish();
    }
}

void MoBenchmark::onJobResult(const JobResult& result) {
    if (result.clientId != String("benchmark")) { // switch to network pool jobs
        JobResults::setListener(m_controller->network(), m_controller->config()->cpu().isHwAES());
        static_cast<IJobResultListener*>(m_controller->network())->onJobResult(result);
        return;
    }
    const Algorithm algo(bench_algos[m_bench_algo]);
    // ignore benchmark results for other perf bench_algo
    if (algo.id() == Algorithm::INVALID || result.jobId != String(algo.name())) return;
    const uint64_t now = get_now();
    if (!m_time_start) m_time_start = now; // time of the first result (in ms)
    m_backends_started.insert(result.backend);
    // waiting for all backends to start
    if (m_backends_started.size() < m_enabled_backend_count && (now - m_time_start < static_cast<unsigned>(3*60*1000))) return;
    ++ m_hash_count;
    if (!m_bench_start) {
       LOG_INFO("%s " BRIGHT_BLACK_BG(WHITE_BOLD_S " Algo " MAGENTA_BOLD_S "%s" WHITE_BOLD_S " Starting test "), Tags::benchmark(), algo.name());
       m_bench_start = now; // time of measurements start (in ms)
    } else if (now - m_bench_start > static_cast<unsigned>(m_controller->config()->benchAlgoTime()*1000)) { // end of benchmark round for m_bench_algo
        double t[3] = { 0.0 };
        for (auto backend : m_controller->miner()->backends()) {
            const Hashrate *hr = backend->hashrate();
            if (!hr) continue;
            auto hr_pair = hr->calc(Hashrate::ShortInterval);
            if (hr_pair.first) t[0] += hr_pair.second;
            hr_pair = hr->calc(Hashrate::MediumInterval);
            if (hr_pair.first) t[1] += hr_pair.second;
            hr_pair = hr->calc(Hashrate::LargeInterval);
            if (hr_pair.first) t[2] += hr_pair.second;
        }
        double hashrate = 0.0f;
        if (!(hashrate = t[2]))
            if (!(hashrate = t[1]))
                if (!(hashrate = t[0]))
                    hashrate = static_cast<double>(m_hash_count) * result.diff / (now - m_bench_start) * 1000.0f;
#       ifdef XMRIG_ALGO_KAWPOW
        if (algo.id() == Algorithm::KAWPOW_RVN) hashrate /= ((double)0xFFFFFFFFFFFFFFFF) / 0xFF000000;
#       endif
        algo_perf[algo.id()] = hashrate; // store hashrate result
        LOG_INFO("%s " BRIGHT_BLACK_BG(WHITE_BOLD_S " Algo " MAGENTA_BOLD_S "%s" WHITE_BOLD_S " hashrate: " CYAN_BOLD_S "%f "), Tags::benchmark(), algo.name(), hashrate);
        run_next_bench_algo();
    }
#   ifdef XMRIG_ALGO_GHOSTRIDER
    else switch (algo.id()) { // Update GhostRider algo job to produce more accurate perf results
        case Algorithm::GHOSTRIDER_RTM: {
            uint8_t* blob = m_bench_job.blob();
            ++ *reinterpret_cast<uint32_t*>(blob+4);
            m_controller->miner()->setJob(m_bench_job, false);
            break;
        }
        default:;
    }
#   endif
}

uint64_t MoBenchmark::get_now() const { // get current time in ms
    using namespace std::chrono;
    return time_point_cast<milliseconds>(high_resolution_clock::now()).time_since_epoch().count();
}

} // namespace xmrig

const char *xmrig::bm_tag()
{
    return Tags::benchmark();
}
