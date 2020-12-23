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


#include "backend/common/HashrateInterpolator.h"


uint64_t xmrig::HashrateInterpolator::interpolate(uint64_t timeStamp) const
{
    timeStamp -= LagMS;

    std::lock_guard<std::mutex> l(m_lock);

    const size_t N = m_data.size();

    if (N < 2) {
        return 0;
    }

    for (size_t i = 0; i < N - 1; ++i) {
        const auto& a = m_data[i];
        const auto& b = m_data[i + 1];

        if (a.second <= timeStamp && timeStamp <= b.second) {
            return a.first + static_cast<int64_t>(b.first - a.first) * (timeStamp - a.second) / (b.second - a.second);
        }
    }

    return 0;
}

void xmrig::HashrateInterpolator::addDataPoint(uint64_t count, uint64_t timeStamp)
{
    std::lock_guard<std::mutex> l(m_lock);

    // Clean up old data
    while (!m_data.empty() && (timeStamp - m_data.front().second > LagMS * 2)) {
        m_data.pop_front();
    }

    m_data.emplace_back(count, timeStamp);
}
