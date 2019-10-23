/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018-2019 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2019 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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


#include <mutex>
#include <string>


#include "backend/cuda/CudaBackend.h"
#include "backend/common/Hashrate.h"
#include "backend/common/interfaces/IWorker.h"
#include "backend/common/Tags.h"
#include "backend/common/Workers.h"
#include "base/io/log/Log.h"
#include "base/net/stratum/Job.h"
#include "base/tools/Chrono.h"
#include "base/tools/String.h"
#include "core/config/Config.h"
#include "core/Controller.h"
#include "rapidjson/document.h"


#ifdef XMRIG_FEATURE_API
#   include "base/api/interfaces/IApiRequest.h"
#endif


namespace xmrig {


extern template class Threads<OclThreads>;


constexpr const size_t oneMiB   = 1024u * 1024u;
static const char *tag          = MAGENTA_BG_BOLD(WHITE_BOLD_S " nv  ");
static const String kType       = "cuda";
static std::mutex mutex;



class CudaBackendPrivate
{
public:
    inline CudaBackendPrivate(Controller *controller) :
        controller(controller)
    {
    }


    void init(const OclConfig &cl)
    {
    }


    inline void start(const Job &job)
    {
    }


    Algorithm algo;
    Controller *controller;
    String profileName;
};


} // namespace xmrig


const char *xmrig::cuda_tag()
{
    return tag;
}


xmrig::CudaBackend::CudaBackend(Controller *controller) :
    d_ptr(new CudaBackendPrivate(controller))
{
}


xmrig::CudaBackend::~CudaBackend()
{
    delete d_ptr;
}


bool xmrig::CudaBackend::isEnabled() const
{
    return false;
}


bool xmrig::CudaBackend::isEnabled(const Algorithm &algorithm) const
{
    return false;
}


const xmrig::Hashrate *xmrig::CudaBackend::hashrate() const
{
    return nullptr;
//    return d_ptr->workers.hashrate();
}


const xmrig::String &xmrig::CudaBackend::profileName() const
{
    return d_ptr->profileName;
}


const xmrig::String &xmrig::CudaBackend::type() const
{
    return kType;
}


void xmrig::CudaBackend::prepare(const Job &)
{
}


void xmrig::CudaBackend::printHashrate(bool details)
{
    if (!details || !hashrate()) {
        return;
    }

    char num[8 * 3] = { 0 };

    Log::print(WHITE_BOLD_S "|   CUDA # | AFFINITY | 10s H/s | 60s H/s | 15m H/s |");

//    size_t i = 0;
//    for (const OclLaunchData &data : d_ptr->threads) {
//         Log::print("| %8zu | %8" PRId64 " | %7s | %7s | %7s |" CYAN_BOLD(" #%u") YELLOW(" %s") " %s",
//                    i,
//                    data.affinity,
//                    Hashrate::format(hashrate()->calc(i, Hashrate::ShortInterval),  num,         sizeof num / 3),
//                    Hashrate::format(hashrate()->calc(i, Hashrate::MediumInterval), num + 8,     sizeof num / 3),
//                    Hashrate::format(hashrate()->calc(i, Hashrate::LargeInterval),  num + 8 * 2, sizeof num / 3),
//                    data.device.index(),
//                    data.device.topology().toString().data(),
//                    data.device.printableName().data()
//                    );

//         i++;
//    }

    Log::print(WHITE_BOLD_S "|        - |        - | %7s | %7s | %7s |",
               Hashrate::format(hashrate()->calc(Hashrate::ShortInterval),  num,         sizeof num / 3),
               Hashrate::format(hashrate()->calc(Hashrate::MediumInterval), num + 8,     sizeof num / 3),
               Hashrate::format(hashrate()->calc(Hashrate::LargeInterval),  num + 8 * 2, sizeof num / 3)
               );
}


void xmrig::CudaBackend::setJob(const Job &job)
{
}


void xmrig::CudaBackend::start(IWorker *worker, bool ready)
{
}


void xmrig::CudaBackend::stop()
{
}


void xmrig::CudaBackend::tick(uint64_t ticks)
{
}


#ifdef XMRIG_FEATURE_API
rapidjson::Value xmrig::CudaBackend::toJSON(rapidjson::Document &doc) const
{
    using namespace rapidjson;
    auto &allocator = doc.GetAllocator();

    Value out(kObjectType);
    out.AddMember("type",       type().toJSON(), allocator);
    out.AddMember("enabled",    isEnabled(), allocator);
    out.AddMember("algo",       d_ptr->algo.toJSON(), allocator);
    out.AddMember("profile",    profileName().toJSON(), allocator);

    return out;
}


void xmrig::CudaBackend::handleRequest(IApiRequest &)
{
}
#endif
