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

#ifndef __MEMORY_H__
#define __MEMORY_H__

#include <windows.h>
#include "options.h"
#include "persistent_memory.h"


char *persistent_memory;
int persistent_memory_flags = 0;


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
static BOOL SetLockPagesPrivilege(HANDLE hProcess, BOOL bEnable) {
    struct {
        DWORD Count;
        LUID_AND_ATTRIBUTES Privilege[1];
    } Info;

    HANDLE Token;

    if (OpenProcessToken(hProcess, TOKEN_ADJUST_PRIVILEGES, &Token) != TRUE) {
        return FALSE;
    }

    Info.Count = 1;
    Info.Privilege[0].Attributes = bEnable ? SE_PRIVILEGE_ENABLED : 0;

    if (LookupPrivilegeValue(NULL, SE_LOCK_MEMORY_NAME, &(Info.Privilege[0].Luid)) != TRUE) {
        return FALSE;
    }

    if (AdjustTokenPrivileges(Token, FALSE, (PTOKEN_PRIVILEGES) &Info, 0, NULL, NULL) != TRUE) {
        return FALSE;
    }

    if (GetLastError() != ERROR_SUCCESS) {
        return FALSE;
    }

    CloseHandle(Token);

    return TRUE;
}


const char * persistent_memory_allocate() {
    const int ratio = opt_double_hash ? 2 : 1;
    const int size = MEMORY * (opt_n_threads * ratio + 1);

    if (SetLockPagesPrivilege(GetCurrentProcess(), TRUE)) {
        persistent_memory_flags |= MEMORY_HUGEPAGES_AVAILABLE;
    }

    persistent_memory = VirtualAlloc(NULL, size, MEM_COMMIT | MEM_RESERVE | MEM_LARGE_PAGES, PAGE_READWRITE);
    if (!persistent_memory) {
        persistent_memory = _mm_malloc(size, 16);
    }
    else {
        persistent_memory_flags |= MEMORY_HUGEPAGES_ENABLED;
    }

    return persistent_memory;
}


void persistent_memory_free() {
    if (persistent_memory_flags & MEMORY_HUGEPAGES_ENABLED) {
        VirtualFree(persistent_memory, 0, MEM_RELEASE);
    }
    else {
        _mm_free(persistent_memory);
    }
}

#endif /* __MEMORY_H__ */
