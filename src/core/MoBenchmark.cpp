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
#include "base/net/stratum/Job.h"
#include "core/config/Config.h"
#include "core/Controller.h"
#include "core/Miner.h"
#include "net/JobResult.h"
#include "net/JobResults.h"
#include "net/Network.h"

#include <chrono>

namespace xmrig {

MoBenchmark::MoBenchmark() : m_controller(nullptr), m_isNewBenchRun(true) {
  for (BenchAlgo bench_algo = BenchAlgo::MIN; bench_algo != BenchAlgo::MAX; bench_algo = static_cast<BenchAlgo>(bench_algo + 1)) {
    m_bench_job[bench_algo] = new Job(false, Algorithm(ba2a[bench_algo]), "benchmark");
  }
}

MoBenchmark::~MoBenchmark() {
  for (BenchAlgo bench_algo = BenchAlgo::MIN; bench_algo != BenchAlgo::MAX; bench_algo = static_cast<BenchAlgo>(bench_algo + 1)) {
    delete m_bench_job[bench_algo];
  }
}

// start performance measurements from the first bench_algo
void MoBenchmark::start() {
    JobResults::setListener(this, m_controller->config()->cpu().isHwAES()); // register benchmark as job result listener to compute hashrates there
    // write text before first benchmark round
    LOG_INFO("%s " BRIGHT_BLACK_BG(CYAN_BOLD_S " STARTING ALGO PERFORMANCE CALIBRATION (with " MAGENTA_BOLD_S "%i" CYAN_BOLD_S " seconds round) "), Tags::benchmark(), m_controller->config()->benchAlgoTime());
    // start benchmarking from first PerfAlgo in the list
    start(BenchAlgo::MIN);
    m_isNewBenchRun = true;
}

// end of benchmarks, switch to jobs from the pool (network), fill algo_perf
void MoBenchmark::finish() {
    for (const Algorithm::Id algo : Algorithm::all([this](const Algorithm &algo) { return true; })) {
        algo_perf[algo] = get_algo_perf(algo);
    }
    m_bench_algo = BenchAlgo::INVALID;
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

    for (const auto &a : m_controller->miner()->algorithms()) {
        if (algo_perf[a.id()] == 0.0f) continue;
        obj.AddMember(StringRef(a.name()), algo_perf[a.id()], allocator);
    }

    return obj;
}

void MoBenchmark::read(const rapidjson::Value &value)
{
    for (const Algorithm::Id algo : Algorithm::all([this](const Algorithm&) { return true; })) {
        algo_perf[algo] = 0.0f;
    }
    if (value.IsObject()) {
        for (auto &member : value.GetObject()) {
            const Algorithm algo(member.name.GetString());
            if (!algo.isValid()) {
                LOG_INFO("%s " BRIGHT_BLACK_BG(MAGENTA_BOLD_S " Ignoring wrong name for algo-perf[%s] "), Tags::benchmark(), member.name.GetString());
                continue;
            }
            if (member.value.IsFloat()) {
                algo_perf[algo.id()] = member.value.GetFloat();
                m_isNewBenchRun = false;
                continue;
            }
            if (member.value.IsInt()) {
                algo_perf[algo.id()] = member.value.GetInt();
                m_isNewBenchRun = false;
                continue;
            }
            LOG_INFO("%s " BRIGHT_BLACK_BG(MAGENTA_BOLD_S " Ignoring wrong value for algo-perf[%s] "), Tags::benchmark(), member.name.GetString());
        }
    }
}

double MoBenchmark::get_algo_perf(Algorithm::Id algo) const {
    switch (algo) {
        case Algorithm::CN_CCX:        return m_bench_algo_perf[BenchAlgo::CN_CCX];
        case Algorithm::CN_0:          return m_bench_algo_perf[BenchAlgo::CN_CCX] / 2;
        case Algorithm::CN_1:          return m_bench_algo_perf[BenchAlgo::CN_R];
        case Algorithm::CN_2:          return m_bench_algo_perf[BenchAlgo::CN_R];
        case Algorithm::CN_R:          return m_bench_algo_perf[BenchAlgo::CN_R];
        case Algorithm::CN_RTO:        return m_bench_algo_perf[BenchAlgo::CN_R];
        case Algorithm::CN_XAO:        return m_bench_algo_perf[BenchAlgo::CN_R];
        case Algorithm::CN_FAST:       return m_bench_algo_perf[BenchAlgo::CN_R] * 2;
        case Algorithm::CN_HALF:       return m_bench_algo_perf[BenchAlgo::CN_R] * 2;
        case Algorithm::CN_RWZ:        return m_bench_algo_perf[BenchAlgo::CN_R] / 3 * 4;
        case Algorithm::CN_ZLS:        return m_bench_algo_perf[BenchAlgo::CN_R] / 3 * 4;
        case Algorithm::CN_DOUBLE:     return m_bench_algo_perf[BenchAlgo::CN_R] / 2;
        case Algorithm::CN_LITE_0:     return m_bench_algo_perf[BenchAlgo::CN_LITE_1];
        case Algorithm::CN_LITE_1:     return m_bench_algo_perf[BenchAlgo::CN_LITE_1];
        case Algorithm::CN_HEAVY_XHV:  return m_bench_algo_perf[BenchAlgo::CN_HEAVY_XHV];
        case Algorithm::CN_PICO_0:     return m_bench_algo_perf[BenchAlgo::CN_PICO_0];
        case Algorithm::CN_PICO_TLO:   return m_bench_algo_perf[BenchAlgo::CN_PICO_0];
        case Algorithm::CN_GPU:        return m_bench_algo_perf[BenchAlgo::CN_GPU];
        case Algorithm::AR2_CHUKWA_V2: return m_bench_algo_perf[BenchAlgo::AR2_CHUKWA_V2];
        case Algorithm::ASTROBWT_DERO: return m_bench_algo_perf[BenchAlgo::ASTROBWT_DERO];
        case Algorithm::KAWPOW_RVN:    return m_bench_algo_perf[BenchAlgo::KAWPOW_RVN];
        case Algorithm::RX_0:          return m_bench_algo_perf[BenchAlgo::RX_0];
        case Algorithm::RX_SFX:        return m_bench_algo_perf[BenchAlgo::RX_0];
        case Algorithm::RX_WOW:        return m_bench_algo_perf[BenchAlgo::RX_WOW];
        case Algorithm::RX_ARQ:        return m_bench_algo_perf[BenchAlgo::RX_ARQ];
        case Algorithm::RX_XLA:        return m_bench_algo_perf[BenchAlgo::RX_XLA];
        default: return 0.0f;
    }
}

// start performance measurements for specified perf bench_algo
void MoBenchmark::start(const BenchAlgo bench_algo) {
    // calculate number of active miner backends in m_enabled_backend_count
    m_enabled_backend_count = 0;
    const Algorithm algo(ba2a[bench_algo]);
    for (auto backend : m_controller->miner()->backends()) if (backend->isEnabled() && backend->isEnabled(algo)) ++ m_enabled_backend_count;
    if (m_enabled_backend_count == 0) {
        run_next_bench_algo(bench_algo);
        return;
    }
    LOG_INFO("%s " BRIGHT_BLACK_BG(WHITE_BOLD_S " Algo " MAGENTA_BOLD_S "%s" WHITE_BOLD_S " Preparation "), Tags::benchmark(), algo.name());
    // prepare test job for benchmark runs ("benchmark" client id is to make sure we can detect benchmark jobs)
    Job& job = *m_bench_job[bench_algo];
    job.setId(algo.name()); // need to set different id so that workers will see job change
    if (bench_algo == BenchAlgo::KAWPOW_RVN) {
      job.setBlob("4c38e8a5f7b2944d1e4274635d828519b97bc64a1f1c7896ecdbb139988aa0e80000000000000000000000000000000000000000000000000000000000000000000000000000000000000000");
      job.setDiff(Job::toDiff(strtoull("000000639c000000", nullptr, 16)));
      job.setHeight(1500000);
    } else {
      // 99 here to trigger all future bench_algo versions for auto veriant detection based on block version
      job.setBlob("9905A0DBD6BF05CF16E503F3A66F78007CBF34144332ECBFC22ED95C8700383B309ACE1923A0964B00000008BA939A62724C0D7581FCE5761E9D8A0E6A1C3F924FDD8493D1115649C05EB601");
      job.setTarget("FFFFFFFFFFFFFF20"); // set difficulty to 8 cause onJobResult after every 8-th computed hash
      job.setHeight(1000);
      job.setSeedHash("0000000000000000000000000000000000000000000000000000000000000001");
    }
    m_bench_algo  = bench_algo; // current perf bench_algo
    m_hash_count  = 0;          // number of hashes calculated for current perf bench_algo
    m_time_start  = 0;          // init time of the first result (in ms) during the first onJobResult
    m_bench_start = 0;          // init time of measurements start (in ms) during the first onJobResult
    m_backends_started.clear();
    m_controller->miner()->setJob(job, false); // set job for workers to compute
}

// run next bench algo or finish benchmark for the last one
void MoBenchmark::run_next_bench_algo(const BenchAlgo bench_algo) {
    const BenchAlgo next_bench_algo = static_cast<BenchAlgo>(bench_algo + 1); // compute next perf bench_algo to benchmark
    if (next_bench_algo != BenchAlgo::MAX) {
        start(next_bench_algo);
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
    // ignore benchmark results for other perf bench_algo
    if (m_bench_algo == BenchAlgo::INVALID || result.jobId != String(Algorithm(ba2a[m_bench_algo]).name())) return;
    const uint64_t now = get_now();
    if (!m_time_start) m_time_start = now; // time of the first result (in ms)
    m_backends_started.insert(result.backend);
    // waiting for all backends to start
    if (m_backends_started.size() < m_enabled_backend_count && (now - m_time_start < static_cast<unsigned>(3*60*1000))) return;
    ++ m_hash_count;
    if (!m_bench_start) {
       LOG_INFO("%s " BRIGHT_BLACK_BG(WHITE_BOLD_S " Algo " MAGENTA_BOLD_S "%s" WHITE_BOLD_S " Starting test "), Tags::benchmark(), Algorithm(ba2a[m_bench_algo]).name());
       m_bench_start = now; // time of measurements start (in ms)
    } else if (now - m_bench_start > static_cast<unsigned>(m_controller->config()->benchAlgoTime()*1000)) { // end of benchmark round for m_bench_algo
        double t[3] = { 0.0 };
        for (auto backend : m_controller->miner()->backends()) {
            const Hashrate *hr = backend->hashrate();
            if (!hr) continue;
            t[0] += hr->calc(Hashrate::ShortInterval);
            t[1] += hr->calc(Hashrate::MediumInterval);
            t[2] += hr->calc(Hashrate::LargeInterval);
        }
        double hashrate = 0.0f;
        if (!(hashrate = t[2]))
            if (!(hashrate = t[1]))
                if (!(hashrate = t[0]))
                    hashrate = static_cast<double>(m_hash_count) * result.diff / (now - m_bench_start) * 1000.0f;
        if (m_bench_algo == KAWPOW_RVN) hashrate /= ((double)0xFFFFFFFFFFFFFFFF) / 0xFF000000;
        m_bench_algo_perf[m_bench_algo] = hashrate; // store hashrate result
        LOG_INFO("%s " BRIGHT_BLACK_BG(WHITE_BOLD_S " Algo " MAGENTA_BOLD_S "%s" WHITE_BOLD_S " hashrate: " CYAN_BOLD_S "%f "), Tags::benchmark(), Algorithm(ba2a[m_bench_algo]).name(), hashrate);
        run_next_bench_algo(m_bench_algo);
    }
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
