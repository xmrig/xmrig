/* XMRigCC
 * Copyright 2017-     BenDr0id    <ben@graef.in>
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

#include "ControlCommand.h"

#include "version.h"

#ifdef TYPE_AMD_GPU
#include "common/log/Log.h"
#else
#include "log/Log.h"
#endif

ControlCommand::ControlCommand()
    : m_command(Command::START)
{

}

ControlCommand::ControlCommand(Command command)
    : m_command(command)
{

}

bool ControlCommand::parseFromJsonString(const std::string& json)
{
    bool result = false;

    rapidjson::Document document;
    if (!document.Parse(json.c_str()).HasParseError()) {
        result = parseFromJson(document);
    }

    return result;
}

bool ControlCommand::parseFromJson(const rapidjson::Document& document)
{
    bool result = false;

    if (document.HasMember("control_command")) {
        const rapidjson::Value& controlCommand = document["control_command"];
        if (controlCommand.HasMember("command")) {
            m_command = toCommand(controlCommand["command"].GetString());
            result = true;
        }
        else {
            LOG_ERR("Parse Error, JSON does not contain: command");
        }
    } else {
        LOG_ERR("Parse Error, JSON does not contain: control_command");
    }

    return result;
}

rapidjson::Value ControlCommand::toJson(rapidjson::MemoryPoolAllocator<rapidjson::CrtAllocator>& allocator)
{
    rapidjson::Value controlCommand(rapidjson::kObjectType);

    controlCommand.AddMember("command", rapidjson::StringRef(toString(m_command)), allocator);

    return controlCommand;
}

void ControlCommand::setCommand(const Command& command)
{
    m_command = command;
}

ControlCommand::Command ControlCommand::getCommand() const
{
    return m_command;
}

bool ControlCommand::isOneTimeCommand() const {

    return m_command == ControlCommand::UPDATE_CONFIG ||
           m_command == ControlCommand::PUBLISH_CONFIG ||
           m_command == ControlCommand::RESTART ||
           m_command == ControlCommand::SHUTDOWN;
}
