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


#include <string.h>
#include <uv.h>


#include "Platform.h"


std::string Platform::m_defaultConfigName = "";
std::string Platform::m_userAgent         = "";


const std::string & Platform::defaultConfigName()
{
	enum
	{
		C_SIZE = 520,
	};

	size_t size = C_SIZE ;
	char defaultConfigName[C_SIZE];
	if(uv_exepath(defaultConfigName, &size) < 0)
	{
		return m_defaultConfigName;
	}

	m_defaultConfigName = defaultConfigName;

	if(size < 500)
	{
#ifdef WIN32
		size_t p = m_defaultConfigName.find_last_of('\\');
#else
		size_t p = m_defaultConfigName.find_last_of('/');
#endif

		if(p != std::string::npos)
		{
			m_defaultConfigName.resize(p + 1);
			m_defaultConfigName.append("config.json");
		}
	}

	return m_defaultConfigName;
}
