/* XMRigCC
 * Copyright 2019-     BenDr0id    <https://github.com/BenDr0id>, <ben@graef.in>
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

#ifndef __HTTPD_H__
#define __HTTPD_H__

#include <memory>
#include <string>
#include <sstream>

#include "3rdparty/cpp-httplib/httplib.h"

#include "CCServerConfig.h"
#include "Service.h"

class Httpd
{
public:
  explicit Httpd(const std::shared_ptr<CCServerConfig>& config);
  ~Httpd();

public:
  int start();
  void stop();

private:
  int basicAuth(const httplib::Request& req, httplib::Response& res);
  int bearerAuth(const httplib::Request& req, httplib::Response& res);

  const std::shared_ptr<CCServerConfig> m_config;
  std::shared_ptr<Service> m_service;
  std::shared_ptr<httplib::Server> m_srv;
};

#endif /* __HTTPD_H__ */
