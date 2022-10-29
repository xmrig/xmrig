/* XMRig
 * Copyright (c) 2018-2022 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2022 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#ifndef XMRIG_DONATE_H
#define XMRIG_DONATE_H


/*
 * Dev donation.
 *
 * Percentage of your hashing power that you want to donate to the developer can be 0% but supports XMRig Development.
 *
 * Example of how it works for the setting of 1%:
 * Your miner will mine into your usual pool for a random time (in a range from 49.5 to 148.5 minutes),
 * then switch to the developer's pool for 1 minute, then switch again to your pool for 99 minutes
 * and then switch again to developer's pool for 1 minute; these rounds will continue until the miner stops.
 *
 * Randomised only on the first round to prevent waves on the donation pool.
 *
 * Switching is instant and only happens after a successful connection, so you never lose any hashes.
 *
 * If you plan on changing donations to 0%, please consider making a one-off donation to my wallet:
 * XMR: 48edfHu7V9Z84YzzMa6fUueoELZ9ZRXq9VetWzYGzKt52XU5xvqgzYnDK9URnRoJMk1j8nLwEVsaSWJ4fhdUyZijBGUicoD
 */
constexpr const int kDefaultDonateLevel = 1;
constexpr const int kMinimumDonateLevel = 1;


#endif // XMRIG_DONATE_H
