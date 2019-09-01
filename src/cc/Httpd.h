/* XMRigCC
 * Copyright 2017-     BenDr0id    <https://github.com/BenDr0id>, <ben@graef.in>
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

#include <string>
#include <sstream>
#include <uv.h>

#include "Options.h"


struct MHD_Connection;
struct MHD_Daemon;
struct MHD_Response;

class Httpd
{
public:
    Httpd(const Options *options);
    bool start();

private:

    typedef struct PostContext
	{
		std::stringstream data;
    } ConnectionContext;

    static int sendResponse(MHD_Connection* connection, unsigned status, MHD_Response* rsp, const char* contentType);

    unsigned basicAuth(MHD_Connection* connection, const std::string& clientIp, std::string &resp);
    unsigned tokenAuth(MHD_Connection* connection, const std::string& clientIp);

    static int handler(void* httpd, MHD_Connection* connection, const char* url, const char* method, const char* version, const char* upload_data, size_t* upload_data_size, void**con_cls);
    static int handleGET(const Httpd* httpd, MHD_Connection* connection, const std::string& clientIp, const char* url);
    static int handlePOST(const Httpd* httpd, MHD_Connection* connection, const std::string& clientIp, const char* url, const char* upload_data, size_t* upload_data_size, void** con_cls);

	static std::string readFile(const std::string &fileName);

	const Options* m_options;
    MHD_Daemon* m_daemon;

	std::string m_keyPem;
	std::string m_certPem;
};

#endif /* __HTTPD_H__ */
