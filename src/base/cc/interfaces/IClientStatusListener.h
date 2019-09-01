/* XMRigCC
 * Copyright 2019-     BenDr0id    <https://github.com/BenDr0id>, <ben@graef.in>
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

#ifndef XMRIG_ICLIENTSTATUSLISTENER_H
#define XMRIG_ICLIENTSTATUSLISTENER_H

#include "cc/ClientStatus.h"

namespace xmrig {


class String;


class IClientStatusListener
{
public:
    virtual ~IClientStatusListener() = default;
#   ifdef XMRIG_FEATURE_CC_CLIENT
    virtual void onUpdateRequest(ClientStatus& clientStatus) = 0;
#   endif
};


} /* namespace xmrig */


#endif // XMRIG_ICLIENTSTATUSLISTENER_H
