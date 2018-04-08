/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2016-2018 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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
 * Percentage of your hashing power that you want to donate to the developer, can be 0 if you don't want to do that.
 *
 * Example of how it works for the setting of 1%:
 * You miner will mine into your usual pool for random time (in range from 49.5 to 148.5 minutes),
 * then switch to the developer's pool for 1 minute, then switch again to your pool for 99 minutes
 * and then switch agaiin to developer's pool for 1 minute, these rounds will continue until miner working.
 *
 * Randomised only first round, to prevent waves on the donation pool.
 *
 * Switching is instant, and only happens after a successful connection, so you never loose any hashes.
 *
 * If you plan on changing this setting to 0 please consider making a one off donation to my wallet:
 * XMR: 48edfHu7V9Z84YzzMa6fUueoELZ9ZRXq9VetWzYGzKt52XU5xvqgzYnDK9URnRoJMk1j8nLwEVsaSWJ4fhdUyZijBGUicoD
 * BTC: 1P7ujsXeX7GxQwHNnJsRMgAdNkFZmNVqJT
 */
constexpr const int kDefaultDonateLevel = 5;
constexpr const int kMinimumDonateLevel = 1;


#endif /* __DONATE_H__ */
