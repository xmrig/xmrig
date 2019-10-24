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
#include <string>

#ifdef XMRIG_FEATURE_TLS
#   include <openssl/opensslv.h>
#   include <cstring>
#endif

#include "base/io/log/Log.h"
#include "version.h"
#include "Summary.h"

static void printVersions()
{
  char buf[256] = { 0 };

#   if defined(__clang__)
  snprintf(buf, sizeof buf, "clang/%d.%d.%d", __clang_major__, __clang_minor__, __clang_patchlevel__);
#   elif defined(__GNUC__)
  snprintf(buf, sizeof buf, "gcc/%d.%d.%d", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#   elif defined(_MSC_VER)
  snprintf(buf, sizeof buf, "MSVC/%d", MSVC_VERSION);
#   endif

  xmrig::Log::print(GREEN_BOLD(" * ") WHITE_BOLD("%-13s") CYAN_BOLD("%s/%s") WHITE_BOLD(" %s") BLUE_BOLD(" (%s)"), "ABOUT", APP_NAME, APP_VERSION, buf, BUILD_TYPE);

  std::string libs;

#if defined(XMRIG_FEATURE_TLS) && defined(OPENSSL_VERSION_TEXT)
{
  constexpr const char *v = OPENSSL_VERSION_TEXT + 8;
  snprintf(buf, sizeof buf, "OpenSSL/%.*s ", static_cast<int>(strchr(v, ' ') - v), v);
  libs += buf;
}
#endif
}

static void printCommands()
{
  xmrig::Log::print(GREEN_BOLD(" * ") WHITE_BOLD("COMMANDS     ") MAGENTA_BOLD("q") WHITE_BOLD("uit, "));
}

static void printPushinfo(const std::shared_ptr<CCServerConfig>& config)
{
  if (config->usePushover() || config->useTelegram())
  {
#ifdef XMRIG_FEATURE_TLS
    xmrig::Log::print(GREEN_BOLD(" * ") WHITE_BOLD("%-13s") CYAN_BOLD("%s%s%s"), "PUSHSERVICE",
                   config->usePushover() ? "Pushover" : "",
                   config->usePushover() && config->useTelegram() ? ", " : "",
                   config->useTelegram() ? "Telegram" : "");

#else
    xmrig::Log::print(GREEN_BOLD(" * ") WHITE_BOLD("%-13s") RED_BOLD("Unavailable requires TLS"), "PUSHSERVICE");
#endif
  }
  else
  {
    xmrig::Log::print(GREEN_BOLD(" * ") WHITE_BOLD("%-13s") RED_BOLD("Disabled"), "PUSHSERVICE");
  }
}

void Summary::print(const std::shared_ptr<CCServerConfig>& config)
{
  printVersions();
  printPushinfo(config);
  printCommands();
}
