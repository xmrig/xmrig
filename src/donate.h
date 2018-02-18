/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2016-2017 XMRig       <support@xmrig.com>
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

#ifndef __DONATE_H__
#define __DONATE_H__


/*
 * Dev donation.
 *
 * Percentage of your hashing power that you want to donate to the donation server,
 * can be 0 if you don't want to do that.
 *
 * If you plan on changing this setting to 0 please consider making a one off donation to my wallet:
 * XMR: 433hhduFBtwVXtQiTTTeqyZsB36XaBLJB6bcQfnqqMs5RJitdpi8xBN21hWiEfuPp2hytmf1cshgK5Grgo6QUvLZCP2QSMi
 *
 * How it works:
 * Other connections switch to donation pool until the first 60 minutes, kDonateLevel minutes each hour
 * with overime compensation. In proxy no way to use precise donation time!
 * You can check actual donation via API.
 */
enum
{
	kDonateLevel = 1,
	kDonateKeepAlive = false,
	kDonateNiceHash = true,
};

static const char* kDonateUrl = "pool.minexmr.com:4444";
static const char* kDonateUser = "";
static const char* kDonatePass = "x";

#endif /* __DONATE_H__ */
