#pragma once

#include <uv.h>


// since 2019.05.16, Version 1.29.0 (Stable)
#if (UV_VERSION_MAJOR >= 1) && (UV_VERSION_MINOR >= 29) && defined(__linux__)
#define XMRIG_UV_PERFORMANCE_BUG
namespace xmrig {


struct uv_async_t: uv_poll_t
{
    typedef void (*uv_async_cb)(uv_async_t* handle);
    ~uv_async_t();
    int m_fd = -1;
    uv_async_cb m_cb = nullptr;
};


using uv_async_cb = uv_async_t::uv_async_cb;
extern int uv_async_init(uv_loop_t *loop, uv_async_t *async, uv_async_cb cb);
extern int uv_async_send(uv_async_t *async);


} // namespace xmrig
#endif
