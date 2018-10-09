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

#ifndef __NETWORK_H__
#define __NETWORK_H__


#include <vector>
#include <uv.h>


#include "api/NetworkState.h"
#include "common/interfaces/IStrategyListener.h"
#include "interfaces/IJobResultListener.h"


class IStrategy;
class Url;


namespace xmrig {
    class Controller;
}


class Network : public IJobResultListener, public IStrategyListener
{
public:
  Network(xmrig::Controller *controller);
  ~Network();

  void connect();
  void stop();

protected:
  void onActive(IStrategy *strategy, Client *client) override;
  void onJob(IStrategy *strategy, Client *client, const Job &job) override;
  void onJobResult(const JobResult &result) override;
  void onPause(IStrategy *strategy) override;
  void onResultAccepted(IStrategy *strategy, Client *client, const SubmitResult &result, const char *error) override;

private:
  constexpr static int kTickInterval = 1 * 1000;

  bool isColors() const;
  void setJob(Client *client, const Job &job, bool donate);
  void tick();

  static void onTick(uv_timer_t *handle);

  IStrategy *m_donate;
  IStrategy *m_strategy;
  NetworkState m_state;
  uv_timer_t m_timer;
  xmrig::Controller *m_controller;
};


#endif /* __NETWORK_H__ */
