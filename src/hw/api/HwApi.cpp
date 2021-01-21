/* XMRig
 * Copyright (c) 2018-2021 SChernykh    <https://github.com/SChernykh>
 * Copyright (c) 2016-2021 XMRig        <https://github.com/xmrig>, <support@xmrig.com>
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


#include "hw/api/HwApi.h"
#include "base/api/interfaces/IApiRequest.h"
#include "base/tools/String.h"


#ifdef XMRIG_FEATURE_DMI
#   include "hw/dmi/DmiReader.h"
#endif


void xmrig::HwApi::onRequest(IApiRequest &request)
{
    if (request.method() == IApiRequest::METHOD_GET) {
#       ifdef XMRIG_FEATURE_DMI
        if (request.url() == "/2/dmi") {
            if (!m_dmi) {
                m_dmi = std::make_shared<DmiReader>();
                m_dmi->read();
            }

            request.accept();
            m_dmi->toJSON(request.reply(), request.doc());
        }
#       endif
    }
}
