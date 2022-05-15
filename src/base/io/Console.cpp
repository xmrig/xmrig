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
 *
  * Additional permission under GNU GPL version 3 section 7
  *
  * If you modify this Program, or any covered work, by linking or combining
  * it with OpenSSL (or a modified version of that library), containing parts
  * covered by the terms of OpenSSL License and SSLeay License, the licensors
  * of this Program grant you additional permission to convey the resulting work.
 */

#include "base/io/Console.h"
#include "base/io/log/Log.h"
#include "base/io/Signals.h"
#include "base/kernel/interfaces/IConsoleListener.h"
#include "base/kernel/private/Title.h"
#include "base/kernel/Process.h"
#include "base/tools/Cvt.h"
#include "base/tools/Handle.h"


#ifdef XMRIG_FEATURE_EVENTS
#   include "base/kernel/Events.h"
#   include "base/kernel/events/ConsoleEvent.h"
#endif


namespace xmrig {


class Console::Private : public IConsoleListener
{
public:
    XMRIG_DISABLE_COPY_MOVE(Private)

    Private();
    ~Private() override;

#   ifdef XMRIG_OS_WIN
    std::wstring title;
#   endif

    IConsoleListener *listener = nullptr;

protected:
    void onConsoleCommand(char command) override;

private:
    static bool isSupported();

    static void onAllocBuffer(uv_handle_t *handle, size_t suggested_size, uv_buf_t *buf);
    static void onRead(uv_stream_t *stream, ssize_t nread, const uv_buf_t *buf);

    char m_buf[1] = { 0 };
    uv_tty_t *m_tty = nullptr;
};


} // namespace xmrig


xmrig::Console::Console() :
    d(std::make_shared<Private>())
{
    d->listener = d.get();
}


xmrig::Console::Console(IConsoleListener *listener) :
    d(std::make_shared<Private>())
{
    d->listener = listener;
}


#ifdef XMRIG_OS_WIN
void xmrig::Console::setTitle(const Title &title) const
{
    SetConsoleTitleW(title.isEnabled() ? Cvt::toUtf16(title.value()).c_str() : d->title.c_str());
}
#endif


xmrig::Console::Private::Private()
{
#   ifdef XMRIG_OS_WIN
    {
        constexpr size_t kMaxTitleLength = 8192;

        wchar_t title_w[kMaxTitleLength]{};
        title = { title_w, GetConsoleTitleW(title_w, kMaxTitleLength) };
    }
#   endif

    if (!isSupported()) {
        return;
    }

    m_tty = new uv_tty_t;
    m_tty->data = this;
    uv_tty_init(uv_default_loop(), m_tty, 0, 1);

    if (!uv_is_readable(reinterpret_cast<uv_stream_t*>(m_tty))) {
        return;
    }

    uv_tty_set_mode(m_tty, UV_TTY_MODE_RAW);
    uv_read_start(reinterpret_cast<uv_stream_t*>(m_tty), onAllocBuffer, onRead);
}


xmrig::Console::Private::~Private()
{
    uv_tty_reset_mode();

    Handle::close(m_tty);
}


void xmrig::Console::Private::onConsoleCommand(char command)
{
#   ifdef XMRIG_FEATURE_EVENTS
    if (command == 3) {
        LOG_WARN("%s " YELLOW_BOLD("Ctrl+C ") YELLOW("received, exiting"), Signals::tag());

        Process::exit(0);
    }
    else {
        Process::events().send<ConsoleEvent>(command);
    }
#   else
    assert(!command);
#   endif            
}


bool xmrig::Console::Private::isSupported()
{
    const uv_handle_type type = uv_guess_handle(0);
    return type == UV_TTY || type == UV_NAMED_PIPE;
}


void xmrig::Console::Private::onAllocBuffer(uv_handle_t *handle, size_t, uv_buf_t *buf)
{
    auto console = static_cast<Private *>(handle->data);
    buf->len  = 1;
    buf->base = console->m_buf;
}


void xmrig::Console::Private::onRead(uv_stream_t *stream, ssize_t nread, const uv_buf_t *buf)
{
    if (nread < 0) {
        return uv_close(reinterpret_cast<uv_handle_t*>(stream), nullptr);
    }

    if (nread == 1) {
        static_cast<Private *>(stream->data)->listener->onConsoleCommand(*buf->base);
    }
}
