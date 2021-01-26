/* XMRig
 * Copyright (c) 2018-2019 tevador     <tevador@gmail.com>
 * Copyright (c) 2018-2021 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2021 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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


#include "crypto/rx/RxFix.h"
#include "base/io/log/Log.h"


#include <csignal>
#include <cstdlib>
#include <ucontext.h>


namespace xmrig {


static thread_local std::pair<const void*, const void*> mainLoopBounds = { nullptr, nullptr };


static void MainLoopHandler(int sig, siginfo_t *info, void *ucontext)
{
    ucontext_t *ucp = (ucontext_t*) ucontext;

    LOG_VERBOSE(YELLOW_BOLD("%s at %p"), (sig == SIGSEGV) ? "SIGSEGV" : "SIGILL", ucp->uc_mcontext.gregs[REG_RIP]);

    void* p = reinterpret_cast<void*>(ucp->uc_mcontext.gregs[REG_RIP]);
    const std::pair<const void*, const void*>& loopBounds = mainLoopBounds;

    if ((loopBounds.first <= p) && (p < loopBounds.second)) {
        ucp->uc_mcontext.gregs[REG_RIP] = reinterpret_cast<size_t>(loopBounds.second);
    }
    else {
        abort();
    }
}


} // namespace xmrig



void xmrig::RxFix::setMainLoopBounds(const std::pair<const void *, const void *> &bounds)
{
    mainLoopBounds = bounds;
}


void xmrig::RxFix::setupMainLoopExceptionFrame()
{
    struct sigaction act = {};
    act.sa_sigaction = MainLoopHandler;
    act.sa_flags = SA_RESTART | SA_SIGINFO;
    sigaction(SIGSEGV, &act, nullptr);
    sigaction(SIGILL, &act, nullptr);
}
