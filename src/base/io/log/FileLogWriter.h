/* XMRig
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

#ifndef XMRIG_FILELOGWRITER_H
#define XMRIG_FILELOGWRITER_H


#include <cstddef>


namespace xmrig {


class FileLogWriter
{
public:
    FileLogWriter() = default;
    FileLogWriter(const char *fileName) { open(fileName); }

    inline bool isOpen() const  { return m_file >= 0; }

    bool open(const char *fileName);
    bool write(const char *data, size_t size);
    bool writeLine(const char *data, size_t size);

private:
    int m_file = -1;
};


} /* namespace xmrig */


#endif /* XMRIG_FILELOGWRITER_H */
