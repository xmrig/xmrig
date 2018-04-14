/* XMRigCC
 * Copyright 2018      Sebastian Stolzenberg <https://github.com/sebastianstolzenberg>
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
#include <thread>

#include <boost/bind.hpp>
#include <boost/asio.hpp>
#include <boost/asio/ssl.hpp>

#include "net/BoostConnection.h"
#include "net/BoostTlsConnection.h"

class BoostTlsSocket
{
public:
    typedef  boost::asio::ssl::stream<boost::asio::ip::tcp::socket> SocketType;
public:
    BoostTlsSocket(boost::asio::io_service& ioService)
            : sslContext_(boost::asio::ssl::context::sslv23_client)
            , socket_(ioService, sslContext_)

    {
        socket_.set_verify_mode(boost::asio::ssl::verify_none);
    }

    template <class ITERATOR, class HANDLER>
    void connect(ITERATOR& iterator, HANDLER handler)
    {
        boost::asio::async_connect(
            socket_.lowest_layer(), iterator,
            [this, handler](const boost::system::error_code& ec, const ITERATOR& iterator)
            {
              if (!ec) {
                  socket_.lowest_layer().set_option(boost::asio::ip::tcp::no_delay(true));
                  socket_.lowest_layer().set_option(boost::asio::socket_base::keep_alive(true));
                  socket_.async_handshake(
                      boost::asio::ssl::stream_base::client,
                      [handler](const boost::system::error_code& ec) {
                        handler(ec);
                      });
              } else {
                  handler(ec);
              }
            }
        );
    }

    SocketType& get()
    {
        return socket_;
    }

    const SocketType& get() const
    {
        return socket_;
    }


private:
    boost::asio::ssl::context sslContext_;
    SocketType socket_;
};

Connection::Ptr establishBoostTlsConnection(const ConnectionListener::Ptr& listener)
{
    return std::make_shared<BoostConnection<BoostTlsSocket> >(listener);
}