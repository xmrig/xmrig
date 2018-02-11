#include <winsock2.h>
#include <windows.h>
#include <ntsecapi.h>
#include <tchar.h>
#ifdef __GNUC__
#   include <mm_malloc.h>
#else
#   include <malloc.h>
#endif
#include "log/Log.h"
#include "crypto/CryptoNight.h"
#include "Mem.h"
#include "Options.h"
static BOOL SetLockPagesPrivilege() {
    HANDLE token;
    if (OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &token) != TRUE) { return FALSE; }
    TOKEN_PRIVILEGES tp;
    tp.PrivilegeCount = 1;
    tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
    if (LookupPrivilegeValue(NULL, SE_LOCK_MEMORY_NAME, &(tp.Privileges[0].Luid)) != TRUE) { return FALSE; }
    BOOL rc = AdjustTokenPrivileges(token, FALSE, (PTOKEN_PRIVILEGES) &tp, 0, NULL, NULL);
    if (rc != TRUE || GetLastError() != ERROR_SUCCESS) { return FALSE; }
    CloseHandle(token);
    return TRUE;
}
static LSA_UNICODE_STRING StringToLsaUnicodeString(LPCTSTR string) {
    LSA_UNICODE_STRING lsaString;
    DWORD dwLen = (DWORD) wcslen(string);
    lsaString.Buffer = (LPWSTR) string;
    lsaString.Length = (USHORT)((dwLen) * sizeof(WCHAR));
    lsaString.MaximumLength = (USHORT)((dwLen + 1) * sizeof(WCHAR));
    return lsaString;
}
static BOOL ObtainLockPagesPrivilege() {
    HANDLE token;
    PTOKEN_USER user = NULL;
    if (OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &token) == TRUE) {
        DWORD size = 0;
        GetTokenInformation(token, TokenUser, NULL, 0, &size);
        if (size) { user = (PTOKEN_USER) LocalAlloc(LPTR, size); }
        GetTokenInformation(token, TokenUser, user, size, &size);
        CloseHandle(token);
    }
    if (!user) { return FALSE; }
    LSA_HANDLE handle;
    LSA_OBJECT_ATTRIBUTES attributes;
    ZeroMemory(&attributes, sizeof(attributes));
    BOOL result = FALSE;
    if (LsaOpenPolicy(NULL, &attributes, POLICY_ALL_ACCESS, &handle) == 0) {
        LSA_UNICODE_STRING str = StringToLsaUnicodeString(_T(SE_LOCK_MEMORY_NAME));
        if (LsaAddAccountRights(handle, user->User.Sid, &str, 1) == 0) { result = TRUE; }
        LsaClose(handle);
    }
    LocalFree(user);
    return result;
}
static BOOL TrySetLockPagesPrivilege() {
    if (SetLockPagesPrivilege()) { return TRUE; }
    return ObtainLockPagesPrivilege() && SetLockPagesPrivilege();
}
bool Mem::allocate(int algo, int threads, bool doubleHash, bool enabled)
{
    m_algo       = algo;
    m_threads    = threads;
    m_doubleHash = doubleHash;
    const int ratio = 2;
    const size_t size  = MEMORY * (threads * ratio + 1);
    if (!enabled) { m_memory = static_cast<uint8_t*>(_mm_malloc(size, 16)); return true; }
    if (TrySetLockPagesPrivilege()) { m_flags |= HugepagesAvailable; }
    m_memory = static_cast<uint8_t*>(VirtualAlloc(NULL, size, MEM_COMMIT | MEM_RESERVE | MEM_LARGE_PAGES, PAGE_READWRITE));
    if (!m_memory) { m_memory = static_cast<uint8_t*>(_mm_malloc(size, 16)); } else { m_flags |= HugepagesEnabled; }
    return true;
}
void Mem::release() { if (m_flags & HugepagesEnabled) { VirtualFree(m_memory, 0, MEM_RELEASE); } else { _mm_free(m_memory); } }