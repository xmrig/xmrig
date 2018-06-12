/* XMRigCC
 * Copyright 2018      Sebastian Stolzenberg <https://github.com/sebastianstolzenberg>
 * Copyright 2018-     BenDr0id <ben@graef.in>
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

#include <iostream>
#include <boost/exception/diagnostic_information.hpp>
#include <log/Log.h>

#include "Connection.h"
#include "BoostTcpConnection.h"

#ifndef XMRIG_NO_TLS
#include "BoostTlsConnection.h"
#endif

Connection::Connection(const ConnectionListener::Ptr &listener)
        : listener_(listener)
{
}

void Connection::notifyConnected()
{
    ConnectionListener::Ptr listener = listener_.lock();
    if (listener)
    {
        listener->scheduleOnConnected();
    }
}

void Connection::notifyRead(char* data, size_t size)
{
    ConnectionListener::Ptr listener = listener_.lock();
    if (listener)
    {
        listener->scheduleOnReceived(data, size);
    }
}

void Connection::notifyError(const std::string& error)
{
    ConnectionListener::Ptr listener = listener_.lock();
    if (listener)
    {
        listener->scheduleOnError(error);
    }
}


Connection::Ptr establishConnection(const ConnectionListener::Ptr& listener,
                                    ConnectionType type, const std::string& host, uint16_t port)
{
    Connection::Ptr connection;

    try {
        switch (type) {
            case CONNECTION_TYPE_TLS:
#ifndef XMRIG_NO_TLS
                connection = establishBoostTlsConnection(listener);
                break;
#endif
            case CONNECTION_TYPE_TCP:
                connection = establishBoostTcpConnection(listener);
                break;
        }

        connection->connect(host, port);
    }
    catch (...) {
        if (connection) {
            connection->disconnect();
        }

        connection->notifyError(std::string("[EstablishConnection] ") + boost::current_exception_diagnostic_information());
    }


    return connection;
}