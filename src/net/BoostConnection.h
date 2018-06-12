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

#ifndef __BOOSTCONNECTION_H__
#define __BOOSTCONNECTION_H__

#include "net/Connection.h"
#include "log/Log.h"

template <class SOCKET>
class BoostConnection : public Connection, public std::enable_shared_from_this<BoostConnection<SOCKET> >
{
public:
    BoostConnection(const ConnectionListener::Ptr& listener)
            : Connection(listener)
            , m_resolver(m_ioService)
            , m_socket(m_ioService)
    {
    }

    ~BoostConnection()
    {
        disconnect();
    }

    void connect(const std::string& server, uint16_t port) override
    {
        LOG_DEBUG("[%s:%d] Connecting", server.c_str(), port);

        boost::asio::ip::tcp::resolver::query query(server, std::to_string(port));

        m_resolver.async_resolve(query,
                                boost::bind(&BoostConnection::handleResolve, this->shared_from_this(),
                                            boost::asio::placeholders::error,
                                            boost::asio::placeholders::iterator));

        std::thread([this]() { m_ioService.run(); }).detach();
    }

    void handleResolve(const boost::system::error_code& error,
                       boost::asio::ip::tcp::resolver::iterator endpointIterator)
    {
        if (!error) {
            #ifdef APP_DEBUG
            boost::asio::ip::tcp::endpoint endpoint = *endpointIterator;
            #endif
            
            LOG_DEBUG("[%s:%d] DNS resolved ", endpoint.address().to_string().c_str(), endpoint.port());
            m_socket.connect(endpointIterator, boost::bind(&BoostConnection::handleConnect, this->shared_from_this(),
                            boost::asio::placeholders::error));
        } else {
            notifyError(std::string("[DNS resolve] ") + error.message());
        }
    }

    void handleConnect(const boost::system::error_code& error)
    {
        if (!error) {
            startReading();
            LOG_DEBUG("[%s:%d] Connected", getConnectedIp().c_str(), getConnectedPort());
            notifyConnected();
        } else {
            notifyError(std::string("[Connect] ") + error.message());
        }
    }

    void disconnect() override
    {
        if (isConnected()) {
            LOG_DEBUG("[%s:%d] Disconnecting", getConnectedIp().c_str(), getConnectedPort());

            boost::system::error_code ec;
            m_socket.get().lowest_layer().shutdown(boost::asio::ip::tcp::socket::shutdown_both, ec);
            m_socket.get().lowest_layer().close();
        }

        m_ioService.stop();
        m_ioService.reset();
    }

    bool isConnected() const override
    {
        boost::system::error_code ec;
        m_socket.get().lowest_layer().remote_endpoint(ec);
        return !ec && m_socket.get().lowest_layer().is_open();
    }

    std::string getConnectedIp() const override
    {
        return isConnected() ? m_socket.get().lowest_layer().remote_endpoint().address().to_string() : "";
    }

    uint16_t getConnectedPort() const override
    {
        return isConnected() ? m_socket.get().lowest_layer().remote_endpoint().port() : 0;
    }

    void send(const char* data, std::size_t size) override
    {
        LOG_DEBUG("[%s:%d] Sending: %.*s", getConnectedIp().c_str(), getConnectedPort(), size, data);

        boost::asio::async_write(m_socket.get(),
                                 boost::asio::buffer(data, size),
                                 boost::bind(&BoostConnection::handleWrite, this->shared_from_this(),
                                             boost::asio::placeholders::error,
                                             boost::asio::placeholders::bytes_transferred));
    }

    void handleWrite(const boost::system::error_code& error,
                     size_t bytes_transferred)
    {
        if (error) {
            LOG_DEBUG_ERR("[%s:%d] Sending failed: %s", getConnectedIp().c_str(), getConnectedPort(), error.message().c_str());
            notifyError(std::string("[Send] ") + error.message());
        }
    }

    void startReading()
    {
        boost::asio::async_read(m_socket.get(),
                                boost::asio::buffer(receiveBuffer_, sizeof(receiveBuffer_)),
                                boost::asio::transfer_at_least(1),
                                boost::bind(&BoostConnection::handleRead, this->shared_from_this(),
                                            boost::asio::placeholders::error,
                                            boost::asio::placeholders::bytes_transferred));
    }

    void handleRead(const boost::system::error_code& error,
                    size_t bytes_transferred)
    {
        if (!error) {
            LOG_DEBUG("[%s:%d] Read: %.*s", getConnectedIp().c_str(), getConnectedPort(), bytes_transferred, receiveBuffer_);
            notifyRead(receiveBuffer_, bytes_transferred);
            startReading();
        } else {
            LOG_DEBUG_ERR("[%s:%d] Read failed: %s", getConnectedIp().c_str(), getConnectedPort(), error.message().c_str());
            notifyError(std::string("[Read] ") + error.message());
        }
    }

private:
    boost::asio::io_service m_ioService;
    boost::asio::ip::tcp::resolver m_resolver;
    SOCKET m_socket;
    char receiveBuffer_[2048];
};

#endif /* __BOOSTCONNECTION_H__ */
