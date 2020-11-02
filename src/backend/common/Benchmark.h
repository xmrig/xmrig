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

#ifndef XMRIG_BENCHMARK_H
#define XMRIG_BENCHMARK_H


#include "base/crypto/Algorithm.h"
#include "base/kernel/interfaces/IHttpListener.h"
#include "base/tools/Object.h"
#include "base/tools/String.h"


#include <memory>


namespace xmrig {


class IBackend;
class IWorker;
class Job;


class Benchmark : public IHttpListener
{
public:
    XMRIG_DISABLE_COPY_MOVE_DEFAULT(Benchmark)

    Benchmark(const Job &job, size_t workers, const IBackend *backend);
    ~Benchmark() override = default;

    bool finish(uint64_t totalHashCount);
    void printProgress() const;
    void start();
    void tick(IWorker *worker);

protected:
    void onHttpData(const HttpData &data) override;

private:
    uint64_t referenceHash() const;
    void printExit();
    void send(const rapidjson::Value &body);
    void setError(const char *message);

    bool m_reset                = false;
    const Algorithm m_algo;
    const IBackend *m_backend;
    const size_t m_workers;
    const String m_id;
    const String m_token;
    const uint64_t m_end;
    const uint64_t m_hash;
    std::shared_ptr<IHttpListener> m_httpListener;
    uint32_t m_done             = 0;
    uint64_t m_current          = 0;
    uint64_t m_data             = 0;
    uint64_t m_doneTime         = 0;
    uint64_t m_startTime        = 0;
};


} // namespace xmrig


#endif /* XMRIG_BENCHMARK_H */
