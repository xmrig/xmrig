/* XMRig
 * Copyright (c) 2018-2022 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2022 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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
 *
  * Additional permission under GNU GPL version 3 section 7
  *
  * If you modify this Program, or any covered work, by linking or combining
  * it with OpenSSL (or a modified version of that library), containing parts
  * covered by the terms of OpenSSL License and SSLeay License, the licensors
  * of this Program grant you additional permission to convey the resulting work.
 */

#include "base/kernel/private/Title.h"
#include "3rdparty/rapidjson/document.h"
#include "base/io/Env.h"
#include "base/kernel/interfaces/IJsonReader.h"
#include "base/tools/Arguments.h"
#include "version.h"


#if defined(APP_DEBUG) && defined(XMRIG_FEATURE_EVENTS)
#   include "base/io/log/Log.h"
#   include "base/kernel/Config.h"
#endif


const char *xmrig::Title::kField    = "title";


xmrig::Title::Title(const Arguments &arguments)
{
#   ifdef XMRIG_DEPRECATED
    if (arguments.contains("--no-title")) {
        m_enabled = false;

        return;
    }
#   endif

    const size_t pos = arguments.pos("--title");
    if (pos) {
        m_value   = arguments.value(pos + 1U);
        m_enabled = !m_value.isEmpty();
    }
}


xmrig::Title::Title(const IJsonReader &reader, const Title &current)
{
    if (!parse(reader.getValue(kField))) {
        m_value   = current.m_value;
        m_enabled = current.m_enabled;
    }
}


rapidjson::Value xmrig::Title::toJSON() const
{
    return isEnabled() ? m_value.toJSON() : rapidjson::Value(m_enabled);
}


xmrig::String xmrig::Title::value() const
{
    if (!isEnabled()) {
        return {};
    }

    if (m_value.isNull()) {
        return APP_NAME " " APP_VERSION;
    }

    return Env::expand(m_value);
}


void xmrig::Title::print() const
{
#   if defined(APP_DEBUG) && defined(XMRIG_FEATURE_EVENTS)
    LOG_DEBUG("%s " MAGENTA_BOLD("TITLE")
              MAGENTA("<enabled=") CYAN("%d")
              MAGENTA(", value=") "\"%s\""
              MAGENTA(">"),
              Config::tag(), m_enabled, m_value.data());
#   endif
}


void xmrig::Title::save(rapidjson::Document &doc) const
{
    doc.AddMember(rapidjson::StringRef(kField), toJSON(), doc.GetAllocator());
}


bool xmrig::Title::parse(const rapidjson::Value &value)
{
    if (value.IsBool()) {
        m_enabled = value.GetBool();

        return true;
    }

    if (value.IsString()) {
        m_value  = value.GetString();
        m_enabled = !m_value.isEmpty();

        return true;
    }

    return false;
}
