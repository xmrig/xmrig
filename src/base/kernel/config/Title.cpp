/* xmlcore
 * Copyright (c) 2018-2020 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2020 xmlcore       <https://github.com/xmlcore>, <support@xmlcore.com>
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


#include "base/kernel/config/Title.h"
#include "3rdparty/rapidjson/document.h"
#include "base/io/Env.h"
#include "version.h"


xmlcore::Title::Title(const rapidjson::Value &value)
{
    if (value.IsBool()) {
        m_enabled = value.GetBool();
    }
    else if (value.IsString()) {
        m_value = value.GetString();
    }
}


rapidjson::Value xmlcore::Title::toJSON() const
{
    if (isEnabled() && !m_value.isNull()) {
        return m_value.toJSON();
    }

    return rapidjson::Value(m_enabled);
}


xmlcore::String xmlcore::Title::value() const
{
    if (!isEnabled()) {
        return {};
    }

    if (m_value.isNull()) {
        return APP_NAME " " APP_VERSION;
    }

    return Env::expand(m_value);
}
