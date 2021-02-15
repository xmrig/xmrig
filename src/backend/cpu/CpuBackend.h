/* xmlcore
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018-2020 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2020 xmlcore       <https://github.com/xmlcore>, <support@xmlcore.com>
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

#ifndef xmlcore_CPUBACKEND_H
#define xmlcore_CPUBACKEND_H


#include "backend/common/interfaces/IBackend.h"
#include "base/tools/Object.h"


#include <utility>


namespace xmlcore {


class Controller;
class CpuBackendPrivate;
class Miner;


class CpuBackend : public IBackend
{
public:
    xmlcore_DISABLE_COPY_MOVE_DEFAULT(CpuBackend)

    CpuBackend(Controller *controller);
    ~CpuBackend() override;

protected:
    inline void execCommand(char) override {}

    bool isEnabled() const override;
    bool isEnabled(const Algorithm &algorithm) const override;
    bool tick(uint64_t ticks) override;
    const Hashrate *hashrate() const override;
    const String &profileName() const override;
    const String &type() const override;
    void prepare(const Job &nextJob) override;
    void printHashrate(bool details) override;
    void printHealth() override;
    void setJob(const Job &job) override;
    void start(IWorker *worker, bool ready) override;
    void stop() override;

#   ifdef xmlcore_FEATURE_API
    rapidjson::Value toJSON(rapidjson::Document &doc) const override;
    void handleRequest(IApiRequest &request) override;
#   endif

#   ifdef xmlcore_FEATURE_BENCHMARK
    Benchmark *benchmark() const override;
    void printBenchProgress() const override;
#   endif

private:
    CpuBackendPrivate *d_ptr;
};


} /* namespace xmlcore */


#endif /* xmlcore_CPUBACKEND_H */
