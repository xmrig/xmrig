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

#ifndef __CONTROL_COMMAND_H__
#define __CONTROL_COMMAND_H__

#include <string>
#include "rapidjson/document.h"

static const char* command_str[7] = {
        "START",
        "STOP",
        "UPDATE_CONFIG",
        "PUBLISH_CONFIG",
        "RESTART",
        "SHUTDOWN",
        "REBOOT"
};

class ControlCommand
{
public:
    enum Command {
        START,
        STOP,
        UPDATE_CONFIG,
        PUBLISH_CONFIG,
        RESTART,
        SHUTDOWN,
        REBOOT
    };

public:
    ControlCommand();
    explicit ControlCommand(Command command);

    static inline const char *toString (Command command)
    {
        return command_str[static_cast<int>(command)];
    }

    static inline Command toCommand (const char *command)
    {
        const int n = sizeof(command_str) / sizeof(command_str[0]);
        for (int i = 0; i < n; ++i)
        {
            if (strcmp(command_str[i], command) == 0)
                return (Command) i;
        }
        return Command::START;
    }

    rapidjson::Value toJson(rapidjson::MemoryPoolAllocator<rapidjson::CrtAllocator>& allocator);
    bool parseFromJsonString(const std::string& json);
    bool parseFromJson(const rapidjson::Document& document);

    Command getCommand() const;
    void setCommand(const Command& command);

    bool isOneTimeCommand() const;

private:
    Command m_command;
};

#endif /* __CONTROL_COMMAND_H__ */
