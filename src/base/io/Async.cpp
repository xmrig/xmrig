#include "base/io/Async.h"


#if defined(XMRIG_UV_PERFORMANCE_BUG)
#include <sys/eventfd.h>
#include <sys/poll.h>
#include <unistd.h>
#include <cstdlib>


namespace xmrig {


uv_async_t::~uv_async_t()
{
    close(m_fd);
}


static void on_schedule(uv_poll_t *handle, int status, int events)
{
    static uint64_t val;
    uv_async_t *async = reinterpret_cast<uv_async_t *>(handle);
    for (;;) {
        int r = read(async->m_fd, &val, sizeof(val));

        if (r == sizeof(val))
            continue;

        if (r != -1)
            break;

        if (errno == EAGAIN || errno == EWOULDBLOCK)
            break;

        if (errno == EINTR)
            continue;

        abort();
    }
    if (async->m_cb) {
        (*async->m_cb)(async);
    }
}


int uv_async_init(uv_loop_t *loop, uv_async_t *async, uv_async_cb cb)
{
    int fd = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
    if (fd < 0) {
        return uv_translate_sys_error(errno);
    }
    uv_poll_init(loop, (uv_poll_t *)async, fd);
    uv_poll_start((uv_poll_t *)async, POLLIN, on_schedule);
    async->m_cb = cb;
    async->m_fd = fd;
    return 0;
}


int uv_async_send(uv_async_t *async)
{
    static const uint64_t val = 1;
    int r;
    do {
        r = write(async->m_fd, &val, sizeof(val));
    }
    while (r == -1 && errno == EINTR);
    if (r == sizeof(val) || (r == 1 && (errno == EAGAIN || errno == EWOULDBLOCK))) {
        return 0;
    }
    abort();
}



} // namespace xmrig
#endif
