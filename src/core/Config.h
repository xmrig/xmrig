/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2016-2018 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#ifndef XMRIG_CONFIG_H
#define XMRIG_CONFIG_H


#include <stdint.h>
#include <vector>

#include "common/config/CommonConfig.h"
#include "common/xmrig.h"
#include "rapidjson/fwd.h"
#include "rapidjson/schema.h"
#include "HasherConfig.h"


namespace xmrig {


class IConfigListener;
class Process;


/**
 * @brief The Config class
 *
 * Options with dynamic reload:
 *   colors
 *   debug
 *   verbose
 *   custom-diff (only for new connections)
 *   api/worker-id
 *   pools/
 */
class Config : public CommonConfig
{
public:
    Config();

    bool reload(const char *json);

    void getJSON(rapidjson::Document &doc) const override;

    inline bool isShouldSave() const                     { return m_shouldSave && isAutoSave(); }
    inline const std::vector<HasherConfig *> &hasherConfigs() const { return m_hashers; }
    inline int priority() const                          { return m_priority; }
    inline int hashersCount() const                      { return m_hashers.size(); }
    inline int cpuThreads() const                        { return m_cpuThreads; }
    inline String cpuOptimization() const                { return m_cpuOptimization; }
    inline int64_t cpuAffinity() const                   { return m_mask; }
    inline std::vector<String> gpuEngine() const         { return m_gpuEngine; }
    inline std::vector<double> gpuIntensity() const      { return m_gpuIntensity; }
    inline std::vector<GPUFilter> gpuFilter() const      { return m_gpuFilter; }

    static Config *load(Process *process, IConfigListener *listener);

protected:
    bool finalize() override;
    bool parseBoolean(int key, bool enable) override;
    bool parseString(int key, const char *arg) override;
    bool parseUint64(int key, uint64_t arg) override;
    void parseJSON(const rapidjson::Document &doc) override;

private:
    bool parseInt(int key, int arg);

    static rapidjson::Value toGPUFilterConfig(const GPUFilter &filter, rapidjson::Document &doc) {
        using namespace rapidjson;
        Value obj(kObjectType);
        auto &allocator = doc.GetAllocator();
        if(!filter.engine.empty() && filter.engine != "*")
            obj.AddMember("engine", Value(filter.engine.data(), doc.GetAllocator()), allocator);
        obj.AddMember("filter", Value(filter.filter.data(), doc.GetAllocator()), allocator);

        return obj;
    }

    static GPUFilter parseGPUFilterConfig(const rapidjson::Value &object) {
        std::string engineInfo;
        std::string filterInfo;
        const auto &filter = object["filter"];
        if (filter.IsString()) {
            filterInfo = filter.GetString();
        }
        const auto &engine = object["engine"];
        if (engine.IsString()) {
            engineInfo = engine.GetString();
        }

        return GPUFilter(engineInfo, filterInfo);
    }
    bool m_shouldSave;
    int m_priority;
    int64_t m_mask;
    int m_cpuThreads;
    String m_cpuOptimization;
    std::vector<String> m_gpuEngine;
    std::vector<double> m_gpuIntensity;
    std::vector<GPUFilter> m_gpuFilter;
    std::vector<HasherConfig *> m_hashers;
};

} /* namespace xmrig */

#endif /* XMRIG_CONFIG_H */
