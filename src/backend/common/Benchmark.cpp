/* XMRig
 * Copyright (c) 2018-2020 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2020 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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


#include "backend/common/Benchmark.h"
#include "backend/common/interfaces/IWorker.h"
#include "base/io/log/Log.h"
#include "base/io/log/Tags.h"
#include "base/tools/Chrono.h"


#include <algorithm>


namespace xmrig {


static uint64_t hashCheck[2][10] = {
    { 0x898B6E0431C28A6BULL, 0xEE9468F8B40926BCULL, 0xC2BC5D11724813C0ULL, 0x3A2C7B285B87F941ULL, 0x3B5BD2C3A16B450EULL, 0x5CD0602F20C5C7C4ULL, 0x101DE939474B6812ULL, 0x52B765A1B156C6ECULL, 0x323935102AB6B45CULL, 0xB5231262E2792B26ULL },
    { 0x0F3E5400B39EA96AULL, 0x85944CCFA2752D1FULL, 0x64AFFCAE991811BAULL, 0x3E4D0B836D3B13BAULL, 0xEB7417D621271166ULL, 0x97FFE10C0949FFA5ULL, 0x84CAC0F8879A4BA1ULL, 0xA1B79F031DA2459FULL, 0x9B65226DA873E65DULL, 0x0F9E00C5A511C200ULL },
};


} // namespace xmrig


bool xmrig::Benchmark::finish(uint64_t totalHashCount)
{
    m_reset   = true;
    m_current = totalHashCount;

    if (m_done < m_workers) {
        return false;
    }

    const double dt    = (m_doneTime - m_startTime) / 1000.0;
    uint64_t checkData = 0;
    const uint32_t N   = (m_end / 1000000) - 1;

    if (((m_algo == Algorithm::RX_0) || (m_algo == Algorithm::RX_WOW)) && ((m_end % 1000000) == 0) && (N < 10)) {
        checkData = hashCheck[(m_algo == Algorithm::RX_0) ? 0 : 1][N];
    }

    const char *color = checkData ? ((m_data == checkData) ? GREEN_BOLD_S : RED_BOLD_S) : BLACK_BOLD_S;

    LOG_NOTICE("%s " WHITE_BOLD("benchmark finished in ") CYAN_BOLD("%.3f seconds") WHITE_BOLD_S " hash sum = " CLEAR "%s%016" PRIX64 CLEAR, Tags::bench(), dt, color, m_data);
    LOG_INFO("%s " WHITE_BOLD("press ") MAGENTA_BOLD("Ctrl+C") WHITE_BOLD(" to exit"), Tags::bench());

    return true;
}


void xmrig::Benchmark::start()
{
    m_startTime = Chrono::steadyMSecs();
}


void xmrig::Benchmark::printProgress() const
{
    if (!m_startTime || !m_current) {
        return;
    }

    const double dt = (Chrono::steadyMSecs() - m_startTime) / 1000.0;
    const double percent = static_cast<double>(m_current) / m_end * 100.0;

    LOG_NOTICE("%s " MAGENTA_BOLD("%5.2f%% ") CYAN_BOLD("%" PRIu64) CYAN("/%" PRIu64) BLACK_BOLD(" (%.3fs)"), Tags::bench(), percent, m_current, m_end, dt);
}


void xmrig::Benchmark::tick(IWorker *worker)
{
    if (m_reset) {
        m_data  = 0;
        m_done  = 0;
        m_reset = false;
    }

    const uint64_t doneTime = worker->benchDoneTime();
    if (!doneTime) {
        return;
    }

    ++m_done;
    m_data ^= worker->benchData();
    m_doneTime = std::max(doneTime, m_doneTime);
}
