/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2016-2017 XMRig       <support@xmrig.com>
 *
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


#include "interfaces/IJobResultListener.h"
#include "interfaces/IStrategyListener.h"


class IStrategy;
class Options;
class Url;


class Network : public IJobResultListener, public IStrategyListener
{
public:
  Network(const Options *options);
  ~Network();

  void connect();

  static char *userAgent();

protected:
  void onActive(Client *client) override;
  void onJob(Client *client, const Job &job) override;
  void onJobResult(const JobResult &result) override;
  void onPause(IStrategy *strategy) override;
  void onResultAccepted(Client *client, int64_t seq, uint32_t diff, uint64_t ms, const char *error) override;

private:
  void setJob(Client *client, const Job &job);

  bool m_donateActive;
  char *m_agent;
  const Options *m_options;
  IStrategy *m_donate;
  IStrategy *m_strategy;
  uint64_t m_accepted;
  uint64_t m_rejected;
};


#endif /* __NETWORK_H__ */
