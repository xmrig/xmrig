#include <winsock2.h>
#include <windows.h>
#include <uv.h>
#include "Platform.h"
#include "version.h"

static inline OSVERSIONINFOEX winOsVersion()
{
    typedef NTSTATUS (NTAPI *RtlGetVersionFunction)(LPOSVERSIONINFO);
    OSVERSIONINFOEX result = { sizeof(OSVERSIONINFOEX), 0, 0, 0, 0, {'\0'}, 0, 0, 0, 0, 0};
    HMODULE ntdll = GetModuleHandleW(L"ntdll.dll");
    if (ntdll ) {
        RtlGetVersionFunction pRtlGetVersion = reinterpret_cast<RtlGetVersionFunction>(GetProcAddress(ntdll, "RtlGetVersion"));
        if (pRtlGetVersion) { pRtlGetVersion((LPOSVERSIONINFO) &result); }
    }
    return result;
}
static inline char *createUserAgent()
{
    const auto osver = winOsVersion();
    const size_t max = 160;
    char *buf = new char[max];
    int length = snprintf(buf, max, "%s/%s (Windows NT %lu.%lu", APP_NAME, APP_VERSION, osver.dwMajorVersion, osver.dwMinorVersion);
#   if defined(__x86_64__) || defined(_M_AMD64)
    length += snprintf(buf + length, max - length, "; Win64; x64) libuv/%s", uv_version_string());
#   else
    length += snprintf(buf + length, max - length, ") libuv/%s", uv_version_string());
#   endif

#   ifdef __GNUC__
    length += snprintf(buf + length, max - length, " gcc/%d.%d.%d", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#   elif _MSC_VER
    length += snprintf(buf + length, max - length, " msvc/%d", MSVC_VERSION);
#   endif
    return buf;
}
void Platform::init(const char *userAgent){ m_userAgent = userAgent ? strdup(userAgent) : createUserAgent(); }
void Platform::release(){ delete [] m_defaultConfigName; delete [] m_userAgent; }
void Platform::setProcessPriority(int priority){}
void Platform::setThreadPriority(int priority) {}
