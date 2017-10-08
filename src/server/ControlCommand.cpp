/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2016-2017 XMRig       <support@xmrig.com>
 * Copyright 2017-     BenDr0id    <ben@graef.in>
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

#include <cstring>
#include <3rdparty/rapidjson/stringbuffer.h>
#include <3rdparty/rapidjson/prettywriter.h>

#include "log/Log.h"
#include "server/ControlCommand.h"

ControlCommand::ControlCommand()
    : m_command(Command::START)
{

}

ControlCommand::ControlCommand(ControlCommand::Command command)
    : m_command(command)
{

}

bool ControlCommand::parseFromJson(const std::string &json)
{
    bool result = false;

    rapidjson::Document document;
    if (!document.Parse(json.c_str()).HasParseError()) {
        if (document.HasMember("control_command"))
        {
            rapidjson::Value controlCommand = document["control_command"].GetObject();
            if (controlCommand.HasMember("command")) {
                m_command = static_cast<Command>(controlCommand["command"].GetUint());
                result = true;
            }
            else {
                LOG_ERR("Parse Error, JSON does not contain: command");
            }
        } else {
            LOG_ERR("Parse Error, JSON does not contain: control_command");
        }
    }
    else {
        LOG_ERR("Parse Error Occured: %d", document.GetParseError());
    }

    return result;
}

std::string ControlCommand::toJson()
{
    rapidjson::Document document;
    document.SetObject();

    rapidjson::Value controlCommand(rapidjson::kObjectType);
    controlCommand.AddMember("command", m_command, document.GetAllocator());

    document.AddMember("control_command", controlCommand, document.GetAllocator());

    rapidjson::StringBuffer buffer(0, 1024);
    rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);
    writer.SetMaxDecimalPlaces(10);
    document.Accept(writer);

    return strdup(buffer.GetString());;
}

void ControlCommand::setCommand(ControlCommand::Command command)
{
    m_command = command;
}

ControlCommand::Command ControlCommand::getCommand() const
{
    return m_command;
}
