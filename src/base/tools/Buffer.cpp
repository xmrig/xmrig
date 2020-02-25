/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018-2020 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2020 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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


#include "base/tools/Buffer.h"


#include <random>


namespace xmrig {


static std::random_device randomDevice;
static std::mt19937 randomEngine(randomDevice());


} // namespace xmrig


static inline uint8_t hf_hex2bin(uint8_t c, bool &err)
{
    if (c >= '0' && c <= '9') {
        return c - '0';
    }

    if (c >= 'a' && c <= 'f') {
        return c - 'a' + 0xA;
    }

    if (c >= 'A' && c <= 'F') {
        return c - 'A' + 0xA;
    }

    err = true;
    return 0;
}


static inline uint8_t hf_bin2hex(uint8_t c)
{
    if (c <= 0x9) {
        return '0' + c;
    }

    return 'a' - 0xA + c;
}


xmrig::Buffer::Buffer(Buffer &&other) noexcept :
    m_data(other.m_data),
    m_size(other.m_size)
{
    other.m_data = nullptr;
    other.m_size = 0;
}


xmrig::Buffer::Buffer(const Buffer &other)
{
    copy(other.data(), other.size());
}


xmrig::Buffer::Buffer(const char *data, size_t size)
{
    copy(data, size);
}


xmrig::Buffer::Buffer(size_t size) :
    m_size(size)
{
    if (size > 0) {
        m_data = new char[size]();
    }
}


xmrig::Buffer::~Buffer()
{
    delete [] m_data;
}


void xmrig::Buffer::from(const char *data, size_t size)
{
    if (m_size > 0) {
        if (m_size == size) {
            memcpy(m_data, data, m_size);

            return;
        }

        delete [] m_data;
    }

    copy(data, size);
}


xmrig::Buffer xmrig::Buffer::allocUnsafe(size_t size)
{
    if (size == 0) {
        return {};
    }

    Buffer buf;
    buf.m_size = size;
    buf.m_data = new char[size];

    return buf;
}


xmrig::Buffer xmrig::Buffer::randomBytes(const size_t size)
{
    Buffer buf(size);
    std::uniform_int_distribution<> dis(0, 255);

    for (size_t i = 0; i < size; ++i) {
        buf.m_data[i] = static_cast<char>(dis(randomEngine));
    }

    return buf;
}


bool xmrig::Buffer::fromHex(const uint8_t *in, size_t size, uint8_t *out)
{
    bool error = false;
    for (size_t i = 0; i < size; i += 2) {
        out[i / 2] = static_cast<uint8_t>((hf_hex2bin(in[i], error) << 4) | hf_hex2bin(in[i + 1], error));

        if (error) {
            return false;
        }
    }

    return true;
}


xmrig::Buffer xmrig::Buffer::fromHex(const char *data, size_t size)
{
    if (data == nullptr || size % 2 != 0) {
        return {};
    }

    Buffer buf(size / 2);
    if (!fromHex(data, size, buf.data())) {
        return {};
    }

    return buf;
}


void xmrig::Buffer::toHex(const uint8_t *in, size_t size, uint8_t *out)
{
    for (size_t i = 0; i < size; i++) {
        out[i * 2]     = hf_bin2hex((in[i] & 0xF0) >> 4);
        out[i * 2 + 1] = hf_bin2hex(in[i] & 0x0F);
    }
}


xmrig::String xmrig::Buffer::toHex() const
{
    if (m_size == 0) {
        return String();
    }

    char *buf       = new char[m_size * 2 + 1];
    buf[m_size * 2] = '\0';

    toHex(m_data, m_size, buf);

    return String(buf);
}


void xmrig::Buffer::copy(const char *data, size_t size)
{
    if (size == 0) {
        m_data = nullptr;
        m_size = 0;

        return;
    }

    m_data = new char[size];
    m_size = size;

    memcpy(m_data, data, m_size);
}


void xmrig::Buffer::move(Buffer &&other)
{
    if (m_size > 0) {
        delete [] m_data;
    }

    m_data = other.m_data;
    m_size = other.m_size;

    other.m_data = nullptr;
    other.m_size = 0;
}
