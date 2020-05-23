/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
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

#ifndef XMRIG_MINER_H
#define XMRIG_MINER_H


#include <vector>


#include "backend/common/interfaces/IRxListener.h"
#include "base/api/interfaces/IApiListener.h"
#include "base/crypto/Algorithm.h"
#include "base/kernel/interfaces/IBaseListener.h"
#include "base/kernel/interfaces/ITimerListener.h"
#include "base/tools/Object.h"


namespace xmrig {


class Controller;
class Job;
class MinerPrivate;
class IBackend;


class Miner : public ITimerListener, public IBaseListener, public IApiListener, public IRxListener
{
public:
    XMRIG_DISABLE_COPY_MOVE_DEFAULT(Miner)

    Miner(Controller *controller);
    ~Miner() override;

    bool isEnabled() const;
    bool isEnabled(const Algorithm &algorithm) const;
    const Algorithms &algorithms() const;
    const std::vector<IBackend *> &backends() const;
    Job job() const;
    void execCommand(char command);
    void pause();
    void setEnabled(bool enabled);
    void setJob(const Job &job, bool donate);
    void stop();

protected:
    void onConfigChanged(Config *config, Config *previousConfig) override;
    void onTimer(const Timer *timer) override;

#   ifdef XMRIG_FEATURE_API
    void onRequest(IApiRequest &request) override;
#   endif

#   ifdef XMRIG_ALGO_RANDOMX
    void onDatasetReady() override;
#   endif

private:
    MinerPrivate *d_ptr;
};


} // namespace xmrig


#endif /* XMRIG_MINER_H */
