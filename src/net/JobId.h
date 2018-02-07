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

#ifndef __JOBID_H__
#define __JOBID_H__


#include <string.h>


class JobId
{
public:
	inline JobId()
	{
	}

	inline JobId(const std::string & id, size_t sizeFix = 0)
	{
		setId(id, sizeFix);
	}

	inline bool operator==(const JobId & other) const
	{
		return m_data == other.m_data;
	}

	inline bool operator!=(const JobId & other) const
	{
		return !operator!=(other);
	}

	inline bool setId(const std::string & id, size_t sizeFix = 0)
	{
		m_data.clear();

		if(id.empty())
		{
			return false;
		}

		const size_t size = id.size();
		m_data = id.substr(0, size - sizeFix);
		return true;
	}

	inline const std::string & data() const
	{
		return m_data;
	}
	inline bool isValid() const
	{
		return 0 < m_data.size() && m_data[0] != '\0';
	}

private:
	std::string m_data;
};

#endif /* __JOBID_H__ */
