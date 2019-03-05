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

#include <algorithm>
#include <winsock2.h>
#include <windows.h>
#include <ntsecapi.h>
#include <tchar.h>

#include "log/Log.h"
#include "crypto/CryptoNight.h"
#include "Mem.h"
#include "Options.h"


/*****************************************************************
SetLockPagesPrivilege: a function to obtain or
release the privilege of locking physical pages.

Inputs:

HANDLE hProcess: Handle for the process for which the
privilege is needed

BOOL bEnable: Enable (TRUE) or disable?

Return value: TRUE indicates success, FALSE failure.

*****************************************************************/
/**
 * AWE Example: https://msdn.microsoft.com/en-us/library/windows/desktop/aa366531(v=vs.85).aspx
 * Creating a File Mapping Using Large Pages: https://msdn.microsoft.com/en-us/library/aa366543(VS.85).aspx
 */
static BOOL SetLockPagesPrivilege() {
    HANDLE token;

    if (OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &token) != TRUE) {
        return FALSE;
    }

    TOKEN_PRIVILEGES tp;
    tp.PrivilegeCount = 1;
    tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;

    if (LookupPrivilegeValue(NULL, SE_LOCK_MEMORY_NAME, &(tp.Privileges[0].Luid)) != TRUE) {
        return FALSE;
    }

    BOOL rc = AdjustTokenPrivileges(token, FALSE, (PTOKEN_PRIVILEGES) &tp, 0, NULL, NULL);
    if (rc != TRUE || GetLastError() != ERROR_SUCCESS) {
        return FALSE;
    }

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
        if (size) {
            user = (PTOKEN_USER) LocalAlloc(LPTR, size);
        }

        GetTokenInformation(token, TokenUser, user, size, &size);
        CloseHandle(token);
    }

    if (!user) {
        return FALSE;
    }

    LSA_HANDLE handle;
    LSA_OBJECT_ATTRIBUTES attributes;
    ZeroMemory(&attributes, sizeof(attributes));

    BOOL result = FALSE;
    if (LsaOpenPolicy(NULL, &attributes, POLICY_ALL_ACCESS, &handle) == 0) {
        LSA_UNICODE_STRING str = StringToLsaUnicodeString(_T(SE_LOCK_MEMORY_NAME));

        if (LsaAddAccountRights(handle, user->User.Sid, &str, 1) == 0) {
            LOG_NOTICE("Huge pages support was successfully enabled, but reboot required to use it");
            result = TRUE;
        }

        LsaClose(handle);
    }

    LocalFree(user);
    return result;
}


static BOOL TrySetLockPagesPrivilege() {
    if (SetLockPagesPrivilege()) {
        return TRUE;
    }

    return ObtainLockPagesPrivilege() && SetLockPagesPrivilege();
}


void Mem::init(const Options* options)
{
    m_hashFactor = options->hashFactor();
    m_useHugePages = options->hugePages();
    m_algo = options->algo();
    m_multiHashThreadMask = Mem::ThreadBitSet(static_cast<unsigned long long int>(options->multiHashThreadMask()));

    if (m_useHugePages && TrySetLockPagesPrivilege()) {
	    m_flags |= HugepagesAvailable;
    }
}

void Mem::allocate(ScratchPadMem& scratchPadMem, bool useHugePages)
{
    scratchPadMem.hugePages = 0;

    if (!useHugePages) {
        scratchPadMem.memory = static_cast<uint8_t*>(_mm_malloc(scratchPadMem.size, 4096));
        return;
    }

    scratchPadMem.size = std::max(scratchPadMem.size, static_cast<size_t>(MEMORY));

    scratchPadMem.memory = static_cast<uint8_t*>(VirtualAlloc(nullptr, scratchPadMem.size, MEM_COMMIT | MEM_RESERVE | MEM_LARGE_PAGES, PAGE_READWRITE));
    if (scratchPadMem.memory) {
        scratchPadMem.hugePages = scratchPadMem.pages;

        m_flags |= HugepagesEnabled;

        return;
    }

    allocate(scratchPadMem, false);
}


void Mem::release(ScratchPadMem &scratchPadMem)
{
    if (scratchPadMem.hugePages) {
        VirtualFree(scratchPadMem.memory, 0, MEM_RELEASE);
    }
    else {
        _mm_free(scratchPadMem.memory);
    }
}

void *Mem::allocateExecutableMemory(size_t size)
{
    return VirtualAlloc(0, size, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE);
}

void Mem::flushInstructionCache(void *p, size_t size)
{
    ::FlushInstructionCache(GetCurrentProcess(), p, size);
}