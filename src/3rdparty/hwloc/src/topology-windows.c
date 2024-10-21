/*
 * Copyright © 2009 CNRS
 * Copyright © 2009-2024 Inria.  All rights reserved.
 * Copyright © 2009-2012, 2020 Université Bordeaux
 * Copyright © 2011 Cisco Systems, Inc.  All rights reserved.
 * See COPYING in top-level directory.
 */

/* To try to get all declarations duplicated below.  */
#define _WIN32_WINNT 0x0601

#include "private/autogen/config.h"
#include "hwloc.h"
#include "hwloc/windows.h"
#include "private/private.h"
#include "private/windows.h" /* must be before windows.h */
#include "private/debug.h"

#include <windows.h>

#ifndef HAVE_KAFFINITY
typedef ULONG_PTR KAFFINITY, *PKAFFINITY;
#endif

#ifndef HAVE_PROCESSOR_CACHE_TYPE
typedef enum _PROCESSOR_CACHE_TYPE {
  CacheUnified,
  CacheInstruction,
  CacheData,
  CacheTrace
} PROCESSOR_CACHE_TYPE;
#endif

#ifndef CACHE_FULLY_ASSOCIATIVE
#define CACHE_FULLY_ASSOCIATIVE 0xFF
#endif

#ifndef MAXIMUM_PROC_PER_GROUP /* missing in MinGW */
#define MAXIMUM_PROC_PER_GROUP 64
#endif

#ifndef HAVE_CACHE_DESCRIPTOR
typedef struct _CACHE_DESCRIPTOR {
  BYTE Level;
  BYTE Associativity;
  WORD LineSize;
  DWORD Size; /* in bytes */
  PROCESSOR_CACHE_TYPE Type;
} CACHE_DESCRIPTOR, *PCACHE_DESCRIPTOR;
#endif

#ifndef HAVE_LOGICAL_PROCESSOR_RELATIONSHIP
typedef enum _LOGICAL_PROCESSOR_RELATIONSHIP {
  RelationProcessorCore,
  RelationNumaNode,
  RelationCache,
  RelationProcessorPackage,
  RelationGroup,
  RelationAll = 0xffff
} LOGICAL_PROCESSOR_RELATIONSHIP;
#else /* HAVE_LOGICAL_PROCESSOR_RELATIONSHIP */
#  ifndef HAVE_RELATIONPROCESSORPACKAGE
#    define RelationProcessorPackage 3
#    define RelationGroup 4
#    define RelationAll 0xffff
#  endif /* HAVE_RELATIONPROCESSORPACKAGE */
#endif /* HAVE_LOGICAL_PROCESSOR_RELATIONSHIP */

#ifndef HAVE_GROUP_AFFINITY
typedef struct _GROUP_AFFINITY {
  KAFFINITY Mask;
  WORD Group;
  WORD Reserved[3];
} GROUP_AFFINITY, *PGROUP_AFFINITY;
#endif

/* always use our own structure because the EfficiencyClass field didn't exist before Win10 */
typedef struct HWLOC_PROCESSOR_RELATIONSHIP {
  BYTE Flags;
  BYTE EfficiencyClass; /* for RelationProcessorCore, higher means greater performance but less efficiency */
  BYTE Reserved[20];
  WORD GroupCount;
  GROUP_AFFINITY GroupMask[ANYSIZE_ARRAY];
} HWLOC_PROCESSOR_RELATIONSHIP;

/* always use our own structure because the GroupCount and GroupMasks fields didn't exist in some Win10 */
typedef struct HWLOC_NUMA_NODE_RELATIONSHIP {
  DWORD NodeNumber;
  BYTE Reserved[18];
  WORD GroupCount;
  _ANONYMOUS_UNION
  union {
    GROUP_AFFINITY GroupMask;
    GROUP_AFFINITY GroupMasks[ANYSIZE_ARRAY];
  } DUMMYUNIONNAME;
} HWLOC_NUMA_NODE_RELATIONSHIP;

typedef struct HWLOC_CACHE_RELATIONSHIP {
  BYTE Level;
  BYTE Associativity;
  WORD LineSize;
  DWORD CacheSize;
  PROCESSOR_CACHE_TYPE Type;
  BYTE Reserved[18];
  WORD GroupCount;
  union {
    GROUP_AFFINITY GroupMask;
    GROUP_AFFINITY GroupMasks[ANYSIZE_ARRAY];
  } DUMMYUNIONNAME;
} HWLOC_CACHE_RELATIONSHIP;

#ifndef HAVE_PROCESSOR_GROUP_INFO
typedef struct _PROCESSOR_GROUP_INFO {
  BYTE MaximumProcessorCount;
  BYTE ActiveProcessorCount;
  BYTE Reserved[38];
  KAFFINITY ActiveProcessorMask;
} PROCESSOR_GROUP_INFO, *PPROCESSOR_GROUP_INFO;
#endif

#ifndef HAVE_GROUP_RELATIONSHIP
typedef struct _GROUP_RELATIONSHIP {
  WORD MaximumGroupCount;
  WORD ActiveGroupCount;
  ULONGLONG Reserved[2];
  PROCESSOR_GROUP_INFO GroupInfo[ANYSIZE_ARRAY];
} GROUP_RELATIONSHIP, *PGROUP_RELATIONSHIP;
#endif

/* always use our own structure because we need our own HWLOC_PROCESSOR/CACHE/NUMA_NODE_RELATIONSHIP */
typedef struct HWLOC_SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX {
  LOGICAL_PROCESSOR_RELATIONSHIP Relationship;
  DWORD Size;
  _ANONYMOUS_UNION
  union {
    HWLOC_PROCESSOR_RELATIONSHIP Processor;
    HWLOC_NUMA_NODE_RELATIONSHIP NumaNode;
    HWLOC_CACHE_RELATIONSHIP Cache;
    GROUP_RELATIONSHIP Group;
    /* Odd: no member to tell the cpu mask of the package... */
  } DUMMYUNIONNAME;
} HWLOC_SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX;

#ifndef HAVE_PSAPI_WORKING_SET_EX_BLOCK
typedef union _PSAPI_WORKING_SET_EX_BLOCK {
  ULONG_PTR Flags;
  struct {
    unsigned Valid  :1;
    unsigned ShareCount  :3;
    unsigned Win32Protection  :11;
    unsigned Shared  :1;
    unsigned Node  :6;
    unsigned Locked  :1;
    unsigned LargePage  :1;
  };
} PSAPI_WORKING_SET_EX_BLOCK;
#endif

#ifndef HAVE_PSAPI_WORKING_SET_EX_INFORMATION
typedef struct _PSAPI_WORKING_SET_EX_INFORMATION {
  PVOID VirtualAddress;
  PSAPI_WORKING_SET_EX_BLOCK VirtualAttributes;
} PSAPI_WORKING_SET_EX_INFORMATION;
#endif

#ifndef HAVE_PROCESSOR_NUMBER
typedef struct _PROCESSOR_NUMBER {
  WORD Group;
  BYTE Number;
  BYTE Reserved;
} PROCESSOR_NUMBER, *PPROCESSOR_NUMBER;
#endif

/* Function pointers */

typedef WORD (WINAPI *PFN_GETACTIVEPROCESSORGROUPCOUNT)(void);
static PFN_GETACTIVEPROCESSORGROUPCOUNT GetActiveProcessorGroupCountProc;

typedef WORD (WINAPI *PFN_GETACTIVEPROCESSORCOUNT)(WORD);
static PFN_GETACTIVEPROCESSORCOUNT GetActiveProcessorCountProc;

typedef DWORD (WINAPI *PFN_GETCURRENTPROCESSORNUMBER)(void);
static PFN_GETCURRENTPROCESSORNUMBER GetCurrentProcessorNumberProc;

typedef VOID (WINAPI *PFN_GETCURRENTPROCESSORNUMBEREX)(PPROCESSOR_NUMBER);
static PFN_GETCURRENTPROCESSORNUMBEREX GetCurrentProcessorNumberExProc;

typedef BOOL (WINAPI *PFN_GETLOGICALPROCESSORINFORMATIONEX)(LOGICAL_PROCESSOR_RELATIONSHIP relationship, HWLOC_SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *Buffer, PDWORD ReturnLength);
static PFN_GETLOGICALPROCESSORINFORMATIONEX GetLogicalProcessorInformationExProc;

typedef BOOL (WINAPI *PFN_SETTHREADGROUPAFFINITY)(HANDLE hThread, const GROUP_AFFINITY *GroupAffinity, PGROUP_AFFINITY PreviousGroupAffinity);
static PFN_SETTHREADGROUPAFFINITY SetThreadGroupAffinityProc;

typedef BOOL (WINAPI *PFN_GETTHREADGROUPAFFINITY)(HANDLE hThread, PGROUP_AFFINITY GroupAffinity);
static PFN_GETTHREADGROUPAFFINITY GetThreadGroupAffinityProc;

typedef BOOL (WINAPI *PFN_GETNUMAAVAILABLEMEMORYNODE)(UCHAR Node, PULONGLONG AvailableBytes);
static PFN_GETNUMAAVAILABLEMEMORYNODE GetNumaAvailableMemoryNodeProc;

typedef BOOL (WINAPI *PFN_GETNUMAAVAILABLEMEMORYNODEEX)(USHORT Node, PULONGLONG AvailableBytes);
static PFN_GETNUMAAVAILABLEMEMORYNODEEX GetNumaAvailableMemoryNodeExProc;

typedef LPVOID (WINAPI *PFN_VIRTUALALLOCEXNUMA)(HANDLE hProcess, LPVOID lpAddress, SIZE_T dwSize, DWORD flAllocationType, DWORD flProtect, DWORD nndPreferred);
static PFN_VIRTUALALLOCEXNUMA VirtualAllocExNumaProc;

typedef BOOL (WINAPI *PFN_VIRTUALFREEEX)(HANDLE hProcess, LPVOID lpAddress, SIZE_T dwSize, DWORD dwFreeType);
static PFN_VIRTUALFREEEX VirtualFreeExProc;

typedef BOOL (WINAPI *PFN_QUERYWORKINGSETEX)(HANDLE hProcess, PVOID pv, DWORD cb);
static PFN_QUERYWORKINGSETEX QueryWorkingSetExProc;

typedef NTSTATUS (WINAPI *PFN_RTLGETVERSION)(OSVERSIONINFOEX*);
PFN_RTLGETVERSION RtlGetVersionProc;

static void hwloc_win_get_function_ptrs(void)
{
  HMODULE kernel32, ntdll;

#if HWLOC_HAVE_GCC_W_CAST_FUNCTION_TYPE
#pragma GCC diagnostic ignored "-Wcast-function-type"
#endif

    kernel32 = LoadLibrary(TEXT("kernel32.dll"));
    if (kernel32) {
      GetActiveProcessorGroupCountProc =
	(PFN_GETACTIVEPROCESSORGROUPCOUNT) GetProcAddress(kernel32, "GetActiveProcessorGroupCount");
      GetActiveProcessorCountProc =
	(PFN_GETACTIVEPROCESSORCOUNT) GetProcAddress(kernel32, "GetActiveProcessorCount");
      GetCurrentProcessorNumberProc =
	(PFN_GETCURRENTPROCESSORNUMBER) GetProcAddress(kernel32, "GetCurrentProcessorNumber");
      GetCurrentProcessorNumberExProc =
	(PFN_GETCURRENTPROCESSORNUMBEREX) GetProcAddress(kernel32, "GetCurrentProcessorNumberEx");
      SetThreadGroupAffinityProc =
	(PFN_SETTHREADGROUPAFFINITY) GetProcAddress(kernel32, "SetThreadGroupAffinity");
      GetThreadGroupAffinityProc =
	(PFN_GETTHREADGROUPAFFINITY) GetProcAddress(kernel32, "GetThreadGroupAffinity");
      GetNumaAvailableMemoryNodeProc =
	(PFN_GETNUMAAVAILABLEMEMORYNODE) GetProcAddress(kernel32, "GetNumaAvailableMemoryNode");
      GetNumaAvailableMemoryNodeExProc =
	(PFN_GETNUMAAVAILABLEMEMORYNODEEX) GetProcAddress(kernel32, "GetNumaAvailableMemoryNodeEx");
      GetLogicalProcessorInformationExProc =
	(PFN_GETLOGICALPROCESSORINFORMATIONEX)GetProcAddress(kernel32, "GetLogicalProcessorInformationEx");
      QueryWorkingSetExProc =
	(PFN_QUERYWORKINGSETEX) GetProcAddress(kernel32, "K32QueryWorkingSetEx");
      VirtualAllocExNumaProc =
	(PFN_VIRTUALALLOCEXNUMA) GetProcAddress(kernel32, "VirtualAllocExNuma");
      VirtualFreeExProc =
	(PFN_VIRTUALFREEEX) GetProcAddress(kernel32, "VirtualFreeEx");
    }

    if (!QueryWorkingSetExProc) {
      HMODULE psapi = LoadLibrary(TEXT("psapi.dll"));
      if (psapi)
        QueryWorkingSetExProc = (PFN_QUERYWORKINGSETEX) GetProcAddress(psapi, "QueryWorkingSetEx");
    }

    ntdll = GetModuleHandle(TEXT("ntdll"));
    RtlGetVersionProc = (PFN_RTLGETVERSION) GetProcAddress(ntdll, "RtlGetVersion");

#if HWLOC_HAVE_GCC_W_CAST_FUNCTION_TYPE
#pragma GCC diagnostic warning "-Wcast-function-type"
#endif
}

/*
 * ULONG_PTR and DWORD_PTR are 64/32bits depending on the arch
 * while bitmaps use unsigned long (always 32bits)
 */

static void hwloc_bitmap_from_ULONG_PTR(hwloc_bitmap_t set, ULONG_PTR mask)
{
#if SIZEOF_VOID_P == 8
  hwloc_bitmap_from_ulong(set, mask & 0xffffffff);
  hwloc_bitmap_set_ith_ulong(set, 1, mask >> 32);
#else
  hwloc_bitmap_from_ulong(set, mask);
#endif
}

static void hwloc_bitmap_from_ith_ULONG_PTR(hwloc_bitmap_t set, unsigned i, ULONG_PTR mask)
{
#if SIZEOF_VOID_P == 8
  hwloc_bitmap_from_ith_ulong(set, 2*i, mask & 0xffffffff);
  hwloc_bitmap_set_ith_ulong(set, 2*i+1, mask >> 32);
#else
  hwloc_bitmap_from_ith_ulong(set, i, mask);
#endif
}

static void hwloc_bitmap_set_ith_ULONG_PTR(hwloc_bitmap_t set, unsigned i, ULONG_PTR mask)
{
#if SIZEOF_VOID_P == 8
  hwloc_bitmap_set_ith_ulong(set, 2*i, mask & 0xffffffff);
  hwloc_bitmap_set_ith_ulong(set, 2*i+1, mask >> 32);
#else
  hwloc_bitmap_set_ith_ulong(set, i, mask);
#endif
}

static ULONG_PTR hwloc_bitmap_to_ULONG_PTR(hwloc_const_bitmap_t set)
{
#if SIZEOF_VOID_P == 8
  ULONG_PTR up = hwloc_bitmap_to_ith_ulong(set, 1);
  up <<= 32;
  up |= hwloc_bitmap_to_ulong(set);
  return up;
#else
  return hwloc_bitmap_to_ulong(set);
#endif
}

static ULONG_PTR hwloc_bitmap_to_ith_ULONG_PTR(hwloc_const_bitmap_t set, unsigned i)
{
#if SIZEOF_VOID_P == 8
  ULONG_PTR up = hwloc_bitmap_to_ith_ulong(set, 2*i+1);
  up <<= 32;
  up |= hwloc_bitmap_to_ith_ulong(set, 2*i);
  return up;
#else
  return hwloc_bitmap_to_ith_ulong(set, i);
#endif
}

/* convert set into index+mask if all set bits are in the same ULONG.
 * otherwise return -1.
 */
static int hwloc_bitmap_to_single_ULONG_PTR(hwloc_const_bitmap_t set, unsigned *index, ULONG_PTR *mask)
{
  unsigned first_ulp, last_ulp;
  if (hwloc_bitmap_weight(set) == -1)
    return -1;
  first_ulp = hwloc_bitmap_first(set) / (sizeof(ULONG_PTR)*8);
  last_ulp = hwloc_bitmap_last(set) / (sizeof(ULONG_PTR)*8);
  if (first_ulp != last_ulp)
    return -1;
  *mask = hwloc_bitmap_to_ith_ULONG_PTR(set, first_ulp);
  *index = first_ulp;
  return 0;
}

/**********************
 * Processor Groups
 */

static unsigned long max_numanode_index = 0;

static unsigned long nr_processor_groups = 1;
static hwloc_cpuset_t * processor_group_cpusets = NULL;

static void
hwloc_win_get_processor_groups(void)
{
  HWLOC_SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *procInfoTotal, *tmpprocInfoTotal, *procInfo;
  DWORD length;
  unsigned i;

  hwloc_debug("querying windows processor groups\n");

  if (!GetLogicalProcessorInformationExProc)
    goto error;

  nr_processor_groups = GetActiveProcessorGroupCountProc();
  if (!nr_processor_groups)
    goto error;

  hwloc_debug("found %lu windows processor groups\n", nr_processor_groups);

  if (nr_processor_groups > 1 && SIZEOF_VOID_P == 4) {
    if (HWLOC_SHOW_ALL_ERRORS())
      fprintf(stderr, "hwloc/windows: multiple processor groups found on 32bits Windows, topology may be invalid/incomplete.\n");
  }

  length = 0;
  procInfoTotal = NULL;

  while (1) {
    if (GetLogicalProcessorInformationExProc(RelationGroup, procInfoTotal, &length))
      break;
    if (GetLastError() != ERROR_INSUFFICIENT_BUFFER)
      goto error;
    tmpprocInfoTotal = realloc(procInfoTotal, length);
    if (!tmpprocInfoTotal)
      goto error_with_procinfo;
    procInfoTotal = tmpprocInfoTotal;
  }

  processor_group_cpusets = calloc(nr_processor_groups, sizeof(*processor_group_cpusets));
  if (!processor_group_cpusets)
    goto error_with_procinfo;

  for (procInfo = procInfoTotal;
       (void*) procInfo < (void*) ((uintptr_t) procInfoTotal + length);
       procInfo = (void*) ((uintptr_t) procInfo + procInfo->Size)) {
    unsigned id;

    assert(procInfo->Relationship == RelationGroup);

    hwloc_debug("Found %u active windows processor groups\n",
                (unsigned) procInfo->Group.ActiveGroupCount);
    for (id = 0; id < procInfo->Group.ActiveGroupCount; id++) {
      KAFFINITY mask;
      hwloc_bitmap_t set;

      set = hwloc_bitmap_alloc();
      if (!set)
        goto error_with_cpusets;

      mask = procInfo->Group.GroupInfo[id].ActiveProcessorMask;
      hwloc_debug("group %u with %u cpus mask 0x%llx\n", id,
                  (unsigned) procInfo->Group.GroupInfo[id].ActiveProcessorCount, (unsigned long long) mask);
      /* KAFFINITY is ULONG_PTR */
      hwloc_bitmap_set_ith_ULONG_PTR(set, id, mask);
      /* FIXME: what if running 32bits on a 64bits windows with 64-processor groups?
       * ULONG_PTR is 32bits, so half the group is invisible?
       * maybe scale id to id*8/sizeof(ULONG_PTR) so that groups are 64-PU aligned?
       */
      hwloc_debug_2args_bitmap("group %u %d bitmap %s\n", id, procInfo->Group.GroupInfo[id].ActiveProcessorCount, set);
      processor_group_cpusets[id] = set;
    }
  }

  free(procInfoTotal);
  return;

 error_with_cpusets:
  for(i=0; i<nr_processor_groups; i++) {
    if (processor_group_cpusets[i])
      hwloc_bitmap_free(processor_group_cpusets[i]);
  }
  free(processor_group_cpusets);
  processor_group_cpusets = NULL;
 error_with_procinfo:
  free(procInfoTotal);
 error:
  /* on error set nr to 1 and keep cpusets NULL. We'll use the topology cpuset whenever needed */
  nr_processor_groups = 1;
}

static void
hwloc_win_free_processor_groups(void)
{
  unsigned i;
  for(i=0; i<nr_processor_groups; i++) {
    if (processor_group_cpusets[i])
      hwloc_bitmap_free(processor_group_cpusets[i]);
  }
  free(processor_group_cpusets);
  processor_group_cpusets = NULL;
  nr_processor_groups = 1;
}


int
hwloc_windows_get_nr_processor_groups(hwloc_topology_t topology, unsigned long flags)
{
  if (!topology->is_loaded || !topology->is_thissystem) {
    errno = EINVAL;
    return -1;
  }

  if (flags) {
    errno = EINVAL;
    return -1;
  }

  return nr_processor_groups;
}

int
hwloc_windows_get_processor_group_cpuset(hwloc_topology_t topology, unsigned pg_index, hwloc_cpuset_t cpuset, unsigned long flags)
{
  if (!topology->is_loaded || !topology->is_thissystem) {
    errno = EINVAL;
    return -1;
  }

  if (!cpuset) {
    errno = EINVAL;
    return -1;
  }

  if (flags) {
    errno = EINVAL;
    return -1;
  }

  if (pg_index >= nr_processor_groups) {
    errno = ENOENT;
    return -1;
  }

  if (!processor_group_cpusets) {
    assert(nr_processor_groups == 1);
    /* we found no processor groups, return the entire topology as a single one */
    hwloc_bitmap_copy(cpuset, topology->levels[0][0]->cpuset);
    return 0;
  }

  if (!processor_group_cpusets[pg_index]) {
    errno = ENOENT;
    return -1;
  }

  hwloc_bitmap_copy(cpuset, processor_group_cpusets[pg_index]);
  return 0;
}

/**************************************************************
 * hwloc PU numbering with respect to Windows processor groups
 *
 * Everywhere below we reserve 64 physical indexes per processor groups because that's
 * the maximum (MAXIMUM_PROC_PER_GROUP). Windows may actually use less bits than that
 * in some groups (either to avoid splitting NUMA nodes across groups, or because of OS
 * tweaks such as "bcdedit /set groupsize 8") but we keep some unused indexes for simplicity.
 * That means PU physical indexes and cpusets may be non-contigous.
 * That also means hwloc_fallback_nbprocessors() below must return the last PU index + 1
 * instead the actual number of processors.
 */

/********************
 * last_cpu_location
 */

static int
hwloc_win_get_thisthread_last_cpu_location(hwloc_topology_t topology __hwloc_attribute_unused, hwloc_cpuset_t set, int flags __hwloc_attribute_unused)
{
  assert(GetCurrentProcessorNumberExProc || (GetCurrentProcessorNumberProc && nr_processor_groups == 1));

  if (nr_processor_groups > 1 || !GetCurrentProcessorNumberProc) {
    PROCESSOR_NUMBER num;
    GetCurrentProcessorNumberExProc(&num);
    hwloc_bitmap_from_ith_ULONG_PTR(set, num.Group, ((ULONG_PTR)1) << num.Number);
    return 0;
  }

  hwloc_bitmap_from_ith_ULONG_PTR(set, 0, ((ULONG_PTR)1) << GetCurrentProcessorNumberProc());
  return 0;
}

/* TODO: hwloc_win_get_thisproc_last_cpu_location() using
 * CreateToolhelp32Snapshot(), Thread32First/Next()
 * th.th32OwnerProcessID == GetCurrentProcessId() for filtering within process
 * OpenThread(THREAD_SET_INFORMATION|THREAD_QUERY_INFORMATION, FALSE, te32.th32ThreadID) to get a handle.
 */


/******************************
 * set cpu/membind for threads
 */

/* TODO: SetThreadIdealProcessor{,Ex} */

static int
hwloc_win_set_thread_cpubind(hwloc_topology_t topology __hwloc_attribute_unused, hwloc_thread_t thread, hwloc_const_bitmap_t hwloc_set, int flags)
{
  DWORD_PTR mask;
  unsigned group;

  if (flags & HWLOC_CPUBIND_NOMEMBIND) {
    errno = ENOSYS;
    return -1;
  }

  if (hwloc_bitmap_to_single_ULONG_PTR(hwloc_set, &group, &mask) < 0) {
    errno = ENOSYS;
    return -1;
  }

  assert(nr_processor_groups == 1 || SetThreadGroupAffinityProc);

  if (nr_processor_groups > 1) {
    GROUP_AFFINITY aff;
    memset(&aff, 0, sizeof(aff)); /* we get Invalid Parameter error if Reserved field isn't cleared */
    aff.Group = group;
    aff.Mask = mask;
    if (!SetThreadGroupAffinityProc(thread, &aff, NULL))
      return -1;

  } else {
    /* SetThreadAffinityMask() only changes the mask inside the current processor group */
    /* The resulting binding is always strict */
    if (!SetThreadAffinityMask(thread, mask))
      return -1;
  }
  return 0;
}

static int
hwloc_win_set_thisthread_cpubind(hwloc_topology_t topology, hwloc_const_bitmap_t hwloc_set, int flags)
{
  return hwloc_win_set_thread_cpubind(topology, GetCurrentThread(), hwloc_set, flags);
}

static int
hwloc_win_set_thisthread_membind(hwloc_topology_t topology, hwloc_const_nodeset_t nodeset, hwloc_membind_policy_t policy, int flags)
{
  int ret;
  hwloc_const_cpuset_t cpuset;
  hwloc_cpuset_t _cpuset = NULL;

  if ((policy != HWLOC_MEMBIND_DEFAULT && policy != HWLOC_MEMBIND_BIND)
      || flags & HWLOC_MEMBIND_NOCPUBIND) {
    errno = ENOSYS;
    return -1;
  }

  if (policy == HWLOC_MEMBIND_DEFAULT) {
    cpuset = hwloc_topology_get_complete_cpuset(topology);
  } else {
    cpuset = _cpuset = hwloc_bitmap_alloc();
    hwloc_cpuset_from_nodeset(topology, _cpuset, nodeset);
  }

  ret = hwloc_win_set_thisthread_cpubind(topology, cpuset,
					 (flags & HWLOC_MEMBIND_STRICT) ? HWLOC_CPUBIND_STRICT : 0);
  hwloc_bitmap_free(_cpuset);
  return ret;
}


/******************************
 * get cpu/membind for threads
 */

static int
hwloc_win_get_thread_cpubind(hwloc_topology_t topology __hwloc_attribute_unused, hwloc_thread_t thread, hwloc_cpuset_t set, int flags __hwloc_attribute_unused)
{
  GROUP_AFFINITY aff;

  assert(GetThreadGroupAffinityProc);

  if (!GetThreadGroupAffinityProc(thread, &aff))
    return -1;
  hwloc_bitmap_from_ith_ULONG_PTR(set, aff.Group, aff.Mask);
  return 0;
}

static int
hwloc_win_get_thisthread_cpubind(hwloc_topology_t topology __hwloc_attribute_unused, hwloc_cpuset_t set, int flags __hwloc_attribute_unused)
{
  return hwloc_win_get_thread_cpubind(topology, GetCurrentThread(), set, flags);
}

static int
hwloc_win_get_thisthread_membind(hwloc_topology_t topology, hwloc_nodeset_t nodeset, hwloc_membind_policy_t * policy, int flags)
{
  int ret;
  hwloc_cpuset_t cpuset = hwloc_bitmap_alloc();
  ret = hwloc_win_get_thread_cpubind(topology, GetCurrentThread(), cpuset, flags);
  if (!ret) {
    *policy = HWLOC_MEMBIND_BIND;
    hwloc_cpuset_to_nodeset(topology, cpuset, nodeset);
  }
  hwloc_bitmap_free(cpuset);
  return ret;
}


/********************************
 * set cpu/membind for processes
 */

static int
hwloc_win_set_proc_cpubind(hwloc_topology_t topology __hwloc_attribute_unused, hwloc_pid_t proc, hwloc_const_bitmap_t hwloc_set, int flags)
{
  DWORD_PTR mask;

  assert(nr_processor_groups == 1);

  if (flags & HWLOC_CPUBIND_NOMEMBIND) {
    errno = ENOSYS;
    return -1;
  }

  /* TODO: SetThreadGroupAffinity() for all threads doesn't enforce the whole process affinity,
   * maybe because of process-specific resource locality */
  /* TODO: if we are in a single group (check with GetProcessGroupAffinity()),
   * SetProcessAffinityMask() changes the binding within that same group.
   */
  /* TODO: NtSetInformationProcess() works very well for binding to any mask in a single group,
   * but it's an internal routine.
   */
  /* TODO: checks whether hwloc-bind.c needs to pass INHERIT_PARENT_AFFINITY to CreateProcess() instead of execvp(). */

  /* The resulting binding is always strict */
  mask = hwloc_bitmap_to_ULONG_PTR(hwloc_set);
  if (!SetProcessAffinityMask(proc, mask))
    return -1;
  return 0;
}

static int
hwloc_win_set_thisproc_cpubind(hwloc_topology_t topology, hwloc_const_bitmap_t hwloc_set, int flags)
{
  return hwloc_win_set_proc_cpubind(topology, GetCurrentProcess(), hwloc_set, flags);
}

static int
hwloc_win_set_proc_membind(hwloc_topology_t topology, hwloc_pid_t pid, hwloc_const_nodeset_t nodeset, hwloc_membind_policy_t policy, int flags)
{
  int ret;
  hwloc_const_cpuset_t cpuset;
  hwloc_cpuset_t _cpuset = NULL;

  if ((policy != HWLOC_MEMBIND_DEFAULT && policy != HWLOC_MEMBIND_BIND)
      || flags & HWLOC_MEMBIND_NOCPUBIND) {
    errno = ENOSYS;
    return -1;
  }

  if (policy == HWLOC_MEMBIND_DEFAULT) {
    cpuset = hwloc_topology_get_complete_cpuset(topology);
  } else {
    cpuset = _cpuset = hwloc_bitmap_alloc();
    hwloc_cpuset_from_nodeset(topology, _cpuset, nodeset);
  }

  ret = hwloc_win_set_proc_cpubind(topology, pid, cpuset,
				   (flags & HWLOC_MEMBIND_STRICT) ? HWLOC_CPUBIND_STRICT : 0);
  hwloc_bitmap_free(_cpuset);
  return ret;
}

static int
hwloc_win_set_thisproc_membind(hwloc_topology_t topology, hwloc_const_nodeset_t nodeset, hwloc_membind_policy_t policy, int flags)
{
  return hwloc_win_set_proc_membind(topology, GetCurrentProcess(), nodeset, policy, flags);
}


/********************************
 * get cpu/membind for processes
 */

static int
hwloc_win_get_proc_cpubind(hwloc_topology_t topology __hwloc_attribute_unused, hwloc_pid_t proc, hwloc_bitmap_t hwloc_set, int flags)
{
  DWORD_PTR proc_mask, sys_mask;

  assert(nr_processor_groups == 1);

  if (flags & HWLOC_CPUBIND_NOMEMBIND) {
    errno = ENOSYS;
    return -1;
  }

  /* TODO: if we are in a single group (check with GetProcessGroupAffinity()),
   * GetProcessAffinityMask() gives the mask within that group.
   */
  /* TODO: if we are in multiple groups, GetProcessGroupAffinity() gives their IDs,
   * but we don't know their masks.
   */
  /* TODO: GetThreadGroupAffinity() for all threads can be smaller than the whole process affinity,
   * maybe because of process-specific resource locality.
   */

  if (!GetProcessAffinityMask(proc, &proc_mask, &sys_mask))
    return -1;
  hwloc_bitmap_from_ULONG_PTR(hwloc_set, proc_mask);
  return 0;
}

static int
hwloc_win_get_proc_membind(hwloc_topology_t topology, hwloc_pid_t pid, hwloc_nodeset_t nodeset, hwloc_membind_policy_t * policy, int flags)
{
  int ret;
  hwloc_cpuset_t cpuset = hwloc_bitmap_alloc();
  ret = hwloc_win_get_proc_cpubind(topology, pid, cpuset,
				   (flags & HWLOC_MEMBIND_STRICT) ? HWLOC_CPUBIND_STRICT : 0);
  if (!ret) {
    *policy = HWLOC_MEMBIND_BIND;
    hwloc_cpuset_to_nodeset(topology, cpuset, nodeset);
  }
  hwloc_bitmap_free(cpuset);
  return ret;
}

static int
hwloc_win_get_thisproc_cpubind(hwloc_topology_t topology, hwloc_bitmap_t hwloc_cpuset, int flags)
{
  return hwloc_win_get_proc_cpubind(topology, GetCurrentProcess(), hwloc_cpuset, flags);
}

static int
hwloc_win_get_thisproc_membind(hwloc_topology_t topology, hwloc_nodeset_t nodeset, hwloc_membind_policy_t * policy, int flags)
{
  return hwloc_win_get_proc_membind(topology, GetCurrentProcess(), nodeset, policy, flags);
}


/************************
 * membind alloc/free
 */

static void *
hwloc_win_alloc(hwloc_topology_t topology __hwloc_attribute_unused, size_t len) {
  return VirtualAlloc(NULL, len, MEM_COMMIT|MEM_RESERVE, PAGE_EXECUTE_READWRITE);
}

static void *
hwloc_win_alloc_membind(hwloc_topology_t topology __hwloc_attribute_unused, size_t len, hwloc_const_nodeset_t nodeset, hwloc_membind_policy_t policy, int flags) {
  int node;

  switch (policy) {
    case HWLOC_MEMBIND_DEFAULT:
    case HWLOC_MEMBIND_BIND:
      break;
    default:
      errno = ENOSYS;
      return hwloc_alloc_or_fail(topology, len, flags);
  }

  if (flags & HWLOC_MEMBIND_STRICT) {
    errno = ENOSYS;
    return NULL;
  }

  if (policy == HWLOC_MEMBIND_DEFAULT
      || hwloc_bitmap_isequal(nodeset, hwloc_topology_get_complete_nodeset(topology)))
    return hwloc_win_alloc(topology, len);

  if (hwloc_bitmap_weight(nodeset) != 1) {
    /* Not a single node, can't do this */
    errno = EXDEV;
    return hwloc_alloc_or_fail(topology, len, flags);
  }

  node = hwloc_bitmap_first(nodeset);
  return VirtualAllocExNumaProc(GetCurrentProcess(), NULL, len, MEM_COMMIT|MEM_RESERVE, PAGE_EXECUTE_READWRITE, node);
}

static int
hwloc_win_free_membind(hwloc_topology_t topology __hwloc_attribute_unused, void *addr, size_t len __hwloc_attribute_unused) {
  if (!addr)
    return 0;
  if (!VirtualFreeExProc(GetCurrentProcess(), addr, 0, MEM_RELEASE))
    return -1;
  return 0;
}


/**********************
 * membind for areas
 */

static int
hwloc_win_get_area_memlocation(hwloc_topology_t topology __hwloc_attribute_unused, const void *addr, size_t len, hwloc_nodeset_t nodeset, int flags __hwloc_attribute_unused)
{
  SYSTEM_INFO SystemInfo;
  DWORD page_size;
  uintptr_t start;
  unsigned nb;
  PSAPI_WORKING_SET_EX_INFORMATION *pv;
  unsigned i;

  GetSystemInfo(&SystemInfo);
  page_size = SystemInfo.dwPageSize;

  start = (((uintptr_t) addr) / page_size) * page_size;
  nb = (unsigned)((((uintptr_t) addr + len - start) + page_size - 1) / page_size);

  if (!nb)
    nb = 1;

  pv = calloc(nb, sizeof(*pv));
  if (!pv)
    return -1;

  for (i = 0; i < nb; i++)
    pv[i].VirtualAddress = (void*) (start + i * page_size);
  if (!QueryWorkingSetExProc(GetCurrentProcess(), pv, nb * sizeof(*pv))) {
    free(pv);
    return -1;
  }

  for (i = 0; i < nb; i++) {
    if (pv[i].VirtualAttributes.Valid)
      hwloc_bitmap_set(nodeset, pv[i].VirtualAttributes.Node);
  }

  free(pv);
  return 0;
}



/*************************
 * Efficiency classes
 */

struct hwloc_win_efficiency_classes {
  unsigned nr_classes;
  unsigned nr_classes_allocated;
  struct hwloc_win_efficiency_class {
    unsigned value;
    hwloc_bitmap_t cpuset;
  } *classes;
};

static void
hwloc_win_efficiency_classes_init(struct hwloc_win_efficiency_classes *classes)
{
  classes->classes = NULL;
  classes->nr_classes_allocated = 0;
  classes->nr_classes = 0;
}

static int
hwloc_win_efficiency_classes_add(struct hwloc_win_efficiency_classes *classes,
                                 hwloc_const_bitmap_t cpuset,
                                 unsigned value)
{
  unsigned i;

  /* look for existing class with that efficiency value */
  for(i=0; i<classes->nr_classes; i++) {
    if (classes->classes[i].value == value) {
      hwloc_bitmap_or(classes->classes[i].cpuset, classes->classes[i].cpuset, cpuset);
      return 0;
    }
  }

  /* extend the array if needed */
  if (classes->nr_classes == classes->nr_classes_allocated) {
    struct hwloc_win_efficiency_class *tmp;
    unsigned new_nr_allocated = 2*classes->nr_classes_allocated;
    if (!new_nr_allocated) {
#define HWLOC_WIN_EFFICIENCY_CLASSES_DEFAULT_MAX 4 /* 2 should be enough is most cases */
      new_nr_allocated = HWLOC_WIN_EFFICIENCY_CLASSES_DEFAULT_MAX;
    }
    tmp = realloc(classes->classes, new_nr_allocated * sizeof(*classes->classes));
    if (!tmp)
      return -1;
    classes->classes = tmp;
    classes->nr_classes_allocated = new_nr_allocated;
  }

  /* add new class */
  classes->classes[classes->nr_classes].cpuset = hwloc_bitmap_alloc();
  if (!classes->classes[classes->nr_classes].cpuset)
    return -1;
  classes->classes[classes->nr_classes].value = value;
  hwloc_bitmap_copy(classes->classes[classes->nr_classes].cpuset, cpuset);
  classes->nr_classes++;
  return 0;
}

static void
hwloc_win_efficiency_classes_register(hwloc_topology_t topology,
                                      struct hwloc_win_efficiency_classes *classes)
{
  unsigned i;
  for(i=0; i<classes->nr_classes; i++) {
    hwloc_internal_cpukinds_register(topology, classes->classes[i].cpuset, classes->classes[i].value, NULL, 0, 0);
    classes->classes[i].cpuset = NULL; /* given to cpukinds */
  }
}

static void
hwloc_win_efficiency_classes_destroy(struct hwloc_win_efficiency_classes *classes)
{
  unsigned i;
  for(i=0; i<classes->nr_classes; i++)
    hwloc_bitmap_free(classes->classes[i].cpuset);
  free(classes->classes);
}

/*************************
 * discovery
 */

static int
hwloc_look_windows(struct hwloc_backend *backend, struct hwloc_disc_status *dstatus)
{
  /*
   * This backend uses the underlying OS.
   * However we don't enforce topology->is_thissystem so that
   * we may still force use this backend when debugging with !thissystem.
   */

  struct hwloc_topology *topology = backend->topology;
  hwloc_bitmap_t groups_pu_set = NULL;
  SYSTEM_INFO SystemInfo;
  DWORD length;
  int gotnuma = 0;
  int gotnumamemory = 0;
  OSVERSIONINFOEX osvi;
  char versionstr[20];
  char hostname[122] = "";
#if !defined(__CYGWIN__)
  DWORD hostname_size = sizeof(hostname);
#else
  size_t hostname_size = sizeof(hostname);
#endif
  int has_efficiencyclass = 0;
  struct hwloc_win_efficiency_classes eclasses;
  char *env = getenv("HWLOC_WINDOWS_PROCESSOR_GROUP_OBJS");
  int keep_pgroup_objs = (env && atoi(env));

  assert(dstatus->phase == HWLOC_DISC_PHASE_CPU);

  if (topology->levels[0][0]->cpuset)
    /* somebody discovered things */
    return -1;

  ZeroMemory(&osvi, sizeof(OSVERSIONINFOEX));
  osvi.dwOSVersionInfoSize = sizeof(OSVERSIONINFOEX);

  if (RtlGetVersionProc) {
    /* RtlGetVersion() returns the currently-running Windows version */
    RtlGetVersionProc(&osvi);
  } else {
    /* GetVersionEx() and isWindows10OrGreater() depend on what the manifest says
     * (manifest of the program, not of libhwloc.dll), they may return old versions
     * if the currently-running Windows is not listed in the manifest.
     */
    GetVersionEx((LPOSVERSIONINFO)&osvi);
  }

  if (osvi.dwMajorVersion >= 10) {
    has_efficiencyclass = 1;
    hwloc_win_efficiency_classes_init(&eclasses);
  }

  hwloc_alloc_root_sets(topology->levels[0][0]);

  GetSystemInfo(&SystemInfo);

  if (GetLogicalProcessorInformationExProc) {
      HWLOC_SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *procInfoTotal, *tmpprocInfoTotal, *procInfo;
      unsigned id;
      struct hwloc_obj *obj;
      hwloc_obj_type_t type;

      length = 0;
      procInfoTotal = NULL;

      while (1) {
	if (GetLogicalProcessorInformationExProc(RelationAll, procInfoTotal, &length))
	  break;
	if (GetLastError() != ERROR_INSUFFICIENT_BUFFER)
	  return -1;
        tmpprocInfoTotal = realloc(procInfoTotal, length);
	if (!tmpprocInfoTotal) {
	  free(procInfoTotal);
	  goto out;
	}
	procInfoTotal = tmpprocInfoTotal;
      }

      for (procInfo = procInfoTotal;
	   (void*) procInfo < (void*) ((uintptr_t) procInfoTotal + length);
	   procInfo = (void*) ((uintptr_t) procInfo + procInfo->Size)) {
        unsigned num, i;
        unsigned efficiency_class = 0;
        GROUP_AFFINITY *GroupMask;

	if (procInfo->Relationship == RelationCache) {
          if (!topology->want_some_cpu_caches)
            /* TODO: check if RelationAll&~RelationCache works? */
            continue;
          if (procInfo->Cache.Type != CacheUnified
              && procInfo->Cache.Type != CacheData
              && procInfo->Cache.Type != CacheInstruction)
            /* Ignore unknown caches */
            continue;
        }

	id = HWLOC_UNKNOWN_INDEX;
	switch (procInfo->Relationship) {
	  case RelationNumaNode:
	    type = HWLOC_OBJ_NUMANODE;
            /* Starting with Windows 11 and Server 2022, the GroupCount field is valid and >=1
             * and we may read GroupMasks[]. Older releases have GroupCount==0 and we must read GroupMask.
             */
            if (procInfo->NumaNode.GroupCount) {
              num = procInfo->NumaNode.GroupCount;
              GroupMask = procInfo->NumaNode.GroupMasks;
            } else {
              num = 1;
              GroupMask = &procInfo->NumaNode.GroupMask;
            }
	    id = procInfo->NumaNode.NodeNumber;
	    gotnuma++;
	    if (id > max_numanode_index)
	      max_numanode_index = id;
	    break;
	  case RelationProcessorPackage:
	    type = HWLOC_OBJ_PACKAGE;
            num = procInfo->Processor.GroupCount;
            GroupMask = procInfo->Processor.GroupMask;
	    break;
	  case RelationCache:
	    type = (procInfo->Cache.Type == CacheInstruction ? HWLOC_OBJ_L1ICACHE : HWLOC_OBJ_L1CACHE) + procInfo->Cache.Level - 1;
            /* GroupCount added approximately with NumaNode.GroupCount above */
            if (procInfo->Cache.GroupCount) {
              num = procInfo->Cache.GroupCount;
              GroupMask = procInfo->Cache.GroupMasks;
            } else {
              num = 1;
              GroupMask = &procInfo->Cache.GroupMask;
            }
	    break;
	  case RelationProcessorCore:
	    type = HWLOC_OBJ_CORE;
            num = procInfo->Processor.GroupCount;
            GroupMask = procInfo->Processor.GroupMask;
            efficiency_class = procInfo->Processor.EfficiencyClass;
	    break;
	  case RelationGroup:
	    /* So strange an interface... */
	    for (id = 0; id < procInfo->Group.ActiveGroupCount; id++) {
              KAFFINITY mask;
	      hwloc_bitmap_t set;

	      set = hwloc_bitmap_alloc();
	      mask = procInfo->Group.GroupInfo[id].ActiveProcessorMask;
	      hwloc_debug("group %u %d cpus mask %lx\n", id,
			  procInfo->Group.GroupInfo[id].ActiveProcessorCount, mask);
	      /* KAFFINITY is ULONG_PTR */
	      hwloc_bitmap_set_ith_ULONG_PTR(set, id, mask);
	      /* FIXME: what if running 32bits on a 64bits windows with 64-processor groups?
	       * ULONG_PTR is 32bits, so half the group is invisible?
	       * maybe scale id to id*8/sizeof(ULONG_PTR) so that groups are 64-PU aligned?
	       */
	      hwloc_debug_2args_bitmap("group %u %d bitmap %s\n", id, procInfo->Group.GroupInfo[id].ActiveProcessorCount, set);

	      /* save the set of PUs so that we can create them at the end */
	      if (!groups_pu_set)
		groups_pu_set = hwloc_bitmap_alloc();
	      hwloc_bitmap_or(groups_pu_set, groups_pu_set, set);

              /* Ignore processor groups unless requested and filtered-in */
              if (keep_pgroup_objs && hwloc_filter_check_keep_object_type(topology, HWLOC_OBJ_GROUP)) {
		obj = hwloc_alloc_setup_object(topology, HWLOC_OBJ_GROUP, id);
		obj->cpuset = set;
		obj->attr->group.kind = HWLOC_GROUP_KIND_WINDOWS_PROCESSOR_GROUP;
		hwloc__insert_object_by_cpuset(topology, NULL, obj, "windows:GetLogicalProcessorInformationEx:ProcessorGroup");
	      } else
		hwloc_bitmap_free(set);
	    }
	    continue;
	  default:
	    /* Don't know how to get the mask.  */
            hwloc_debug("unknown relation %d\n", procInfo->Relationship);
	    continue;
	}

	if (!hwloc_filter_check_keep_object_type(topology, type))
	  continue;

	obj = hwloc_alloc_setup_object(topology, type, id);
        obj->cpuset = hwloc_bitmap_alloc();
        for (i = 0; i < num; i++) {
          hwloc_debug("%s#%u %d: mask %d:%lx\n", hwloc_obj_type_string(type), id, i, GroupMask[i].Group, GroupMask[i].Mask);
	  /* GROUP_AFFINITY.Mask is KAFFINITY, which is ULONG_PTR */
	  hwloc_bitmap_set_ith_ULONG_PTR(obj->cpuset, GroupMask[i].Group, GroupMask[i].Mask);
	  /* FIXME: scale id to id*8/sizeof(ULONG_PTR) as above? */
        }
	hwloc_debug_2args_bitmap("%s#%u bitmap %s\n", hwloc_obj_type_string(type), id, obj->cpuset);
	switch (type) {
        case HWLOC_OBJ_CORE: {
          if (has_efficiencyclass)
            hwloc_win_efficiency_classes_add(&eclasses, obj->cpuset, efficiency_class);
          break;
        }
	  case HWLOC_OBJ_NUMANODE:
	    {
	      ULONGLONG avail;
	      obj->nodeset = hwloc_bitmap_alloc();
	      hwloc_bitmap_set(obj->nodeset, id);
	      if ((GetNumaAvailableMemoryNodeExProc && GetNumaAvailableMemoryNodeExProc(id, &avail))
		  || (GetNumaAvailableMemoryNodeProc && GetNumaAvailableMemoryNodeProc(id, &avail))) {
	        obj->attr->numanode.local_memory = avail;
		gotnumamemory++;
	      }
	      obj->attr->numanode.page_types = malloc(2 * sizeof(*obj->attr->numanode.page_types));
	      memset(obj->attr->numanode.page_types, 0, 2 * sizeof(*obj->attr->numanode.page_types));
	      obj->attr->numanode.page_types_len = 1;
	      obj->attr->numanode.page_types[0].size = SystemInfo.dwPageSize;
#if HAVE_DECL__SC_LARGE_PAGESIZE
	      obj->attr->numanode.page_types_len++;
	      obj->attr->numanode.page_types[1].size = sysconf(_SC_LARGE_PAGESIZE);
#endif
	      break;
	    }
	  case HWLOC_OBJ_L1CACHE:
	  case HWLOC_OBJ_L2CACHE:
	  case HWLOC_OBJ_L3CACHE:
	  case HWLOC_OBJ_L4CACHE:
	  case HWLOC_OBJ_L5CACHE:
	  case HWLOC_OBJ_L1ICACHE:
	  case HWLOC_OBJ_L2ICACHE:
	  case HWLOC_OBJ_L3ICACHE:
	    obj->attr->cache.size = procInfo->Cache.CacheSize;
	    obj->attr->cache.associativity = procInfo->Cache.Associativity == CACHE_FULLY_ASSOCIATIVE ? -1 : procInfo->Cache.Associativity ;
	    obj->attr->cache.linesize = procInfo->Cache.LineSize;
	    obj->attr->cache.depth = procInfo->Cache.Level;
	    switch (procInfo->Cache.Type) {
	      case CacheUnified:
		obj->attr->cache.type = HWLOC_OBJ_CACHE_UNIFIED;
		break;
	      case CacheData:
		obj->attr->cache.type = HWLOC_OBJ_CACHE_DATA;
		break;
	      case CacheInstruction:
		obj->attr->cache.type = HWLOC_OBJ_CACHE_INSTRUCTION;
		break;
	      default:
		hwloc_free_unlinked_object(obj);
		continue;
	    }
	    break;
	  default:
	    break;
	}
	hwloc__insert_object_by_cpuset(topology, NULL, obj, "windows:GetLogicalProcessorInformationEx");
      }
      free(procInfoTotal);
  }

  topology->support.discovery->pu = 1;
  topology->support.discovery->numa = gotnuma;
  topology->support.discovery->numa_memory = gotnumamemory;

  if (groups_pu_set) {
    /* the system supports multiple Groups.
     * PU indexes may be discontiguous, especially if Groups contain less than 64 procs.
     */
    hwloc_obj_t obj;
    unsigned idx;
    hwloc_bitmap_foreach_begin(idx, groups_pu_set) {
      obj = hwloc_alloc_setup_object(topology, HWLOC_OBJ_PU, idx);
      obj->cpuset = hwloc_bitmap_alloc();
      hwloc_bitmap_only(obj->cpuset, idx);
      hwloc_debug_1arg_bitmap("cpu %u has cpuset %s\n",
			      idx, obj->cpuset);
      hwloc__insert_object_by_cpuset(topology, NULL, obj, "windows:ProcessorGroup:pu");
    } hwloc_bitmap_foreach_end();
    hwloc_bitmap_free(groups_pu_set);
  } else {
    /* no processor groups */
    hwloc_obj_t obj;
    unsigned idx;
    for(idx=0; idx<32; idx++)
      if (SystemInfo.dwActiveProcessorMask & (((DWORD_PTR)1)<<idx)) {
	obj = hwloc_alloc_setup_object(topology, HWLOC_OBJ_PU, idx);
	obj->cpuset = hwloc_bitmap_alloc();
	hwloc_bitmap_only(obj->cpuset, idx);
	hwloc_debug_1arg_bitmap("cpu %u has cpuset %s\n",
				idx, obj->cpuset);
	hwloc__insert_object_by_cpuset(topology, NULL, obj, "windows:pu");
      }
  }

  if (has_efficiencyclass) {
    topology->support.discovery->cpukind_efficiency = 1;
    hwloc_win_efficiency_classes_register(topology, &eclasses);
  }

 out:
  if (has_efficiencyclass)
    hwloc_win_efficiency_classes_destroy(&eclasses);

  /* emulate uname instead of calling hwloc_add_uname_info() */
  hwloc_obj_add_info(topology->levels[0][0], "Backend", "Windows");
  hwloc_obj_add_info(topology->levels[0][0], "OSName", "Windows");

#if defined(__CYGWIN__)
  hwloc_obj_add_info(topology->levels[0][0], "WindowsBuildEnvironment", "Cygwin");
#elif defined(__MINGW32__)
  hwloc_obj_add_info(topology->levels[0][0], "WindowsBuildEnvironment", "MinGW");
#endif

  /* see https://docs.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-osversioninfoexa */
  if (osvi.dwMajorVersion == 10) {
    if (osvi.dwMinorVersion == 0)
      hwloc_obj_add_info(topology->levels[0][0], "OSRelease", "10");
  } else if (osvi.dwMajorVersion == 6) {
    if (osvi.dwMinorVersion == 3)
      hwloc_obj_add_info(topology->levels[0][0], "OSRelease", "8.1"); /* or "Server 2012 R2" */
    else if (osvi.dwMinorVersion == 2)
      hwloc_obj_add_info(topology->levels[0][0], "OSRelease", "8"); /* or "Server 2012" */
    else if (osvi.dwMinorVersion == 1)
      hwloc_obj_add_info(topology->levels[0][0], "OSRelease", "7"); /* or "Server 2008 R2" */
    else if (osvi.dwMinorVersion == 0)
      hwloc_obj_add_info(topology->levels[0][0], "OSRelease", "Vista"); /* or "Server 2008" */
  } /* earlier versions are ignored */

  snprintf(versionstr, sizeof(versionstr), "%u.%u.%u", osvi.dwMajorVersion, osvi.dwMinorVersion, osvi.dwBuildNumber);
  hwloc_obj_add_info(topology->levels[0][0], "OSVersion", versionstr);

#if !defined(__CYGWIN__)
  GetComputerName(hostname, &hostname_size);
#else
  gethostname(hostname, hostname_size);
#endif
  if (*hostname)
    hwloc_obj_add_info(topology->levels[0][0], "Hostname", hostname);

  /* convert to unix-like architecture strings */
  switch (SystemInfo.wProcessorArchitecture) {
  case 0:
    hwloc_obj_add_info(topology->levels[0][0], "Architecture", "i686");
    break;
  case 9:
    hwloc_obj_add_info(topology->levels[0][0], "Architecture", "x86_64");
    break;
  case 5:
    hwloc_obj_add_info(topology->levels[0][0], "Architecture", "arm");
    break;
  case 12:
    hwloc_obj_add_info(topology->levels[0][0], "Architecture", "arm64");
    break;
  case 6:
    hwloc_obj_add_info(topology->levels[0][0], "Architecture", "ia64");
    break;
  }

  return 0;
}

void
hwloc_set_windows_hooks(struct hwloc_binding_hooks *hooks,
			struct hwloc_topology_support *support)
{
  if (GetCurrentProcessorNumberExProc || (GetCurrentProcessorNumberProc && nr_processor_groups == 1))
    hooks->get_thisthread_last_cpu_location = hwloc_win_get_thisthread_last_cpu_location;

  if (nr_processor_groups == 1) {
    hooks->set_proc_cpubind = hwloc_win_set_proc_cpubind;
    hooks->get_proc_cpubind = hwloc_win_get_proc_cpubind;
    hooks->set_thisproc_cpubind = hwloc_win_set_thisproc_cpubind;
    hooks->get_thisproc_cpubind = hwloc_win_get_thisproc_cpubind;
    hooks->set_proc_membind = hwloc_win_set_proc_membind;
    hooks->get_proc_membind = hwloc_win_get_proc_membind;
    hooks->set_thisproc_membind = hwloc_win_set_thisproc_membind;
    hooks->get_thisproc_membind = hwloc_win_get_thisproc_membind;
  }
  if (nr_processor_groups == 1 || SetThreadGroupAffinityProc) {
    hooks->set_thread_cpubind = hwloc_win_set_thread_cpubind;
    hooks->set_thisthread_cpubind = hwloc_win_set_thisthread_cpubind;
    hooks->set_thisthread_membind = hwloc_win_set_thisthread_membind;
  }
  if (GetThreadGroupAffinityProc) {
    hooks->get_thread_cpubind = hwloc_win_get_thread_cpubind;
    hooks->get_thisthread_cpubind = hwloc_win_get_thisthread_cpubind;
    hooks->get_thisthread_membind = hwloc_win_get_thisthread_membind;
  }

  if (VirtualAllocExNumaProc) {
    hooks->alloc_membind = hwloc_win_alloc_membind;
    hooks->alloc = hwloc_win_alloc;
    hooks->free_membind = hwloc_win_free_membind;
    support->membind->bind_membind = 1;
  }

  if (QueryWorkingSetExProc && max_numanode_index <= 63 /* PSAPI_WORKING_SET_EX_BLOCK.Node is 6 bits only */)
    hooks->get_area_memlocation = hwloc_win_get_area_memlocation;
}

static int hwloc_windows_component_init(unsigned long flags __hwloc_attribute_unused)
{
  hwloc_win_get_function_ptrs();
  hwloc_win_get_processor_groups();
  return 0;
}

static void hwloc_windows_component_finalize(unsigned long flags __hwloc_attribute_unused)
{
  hwloc_win_free_processor_groups();
}

static struct hwloc_backend *
hwloc_windows_component_instantiate(struct hwloc_topology *topology,
				    struct hwloc_disc_component *component,
				    unsigned excluded_phases __hwloc_attribute_unused,
				    const void *_data1 __hwloc_attribute_unused,
				    const void *_data2 __hwloc_attribute_unused,
				    const void *_data3 __hwloc_attribute_unused)
{
  struct hwloc_backend *backend;
  backend = hwloc_backend_alloc(topology, component);
  if (!backend)
    return NULL;
  backend->discover = hwloc_look_windows;
  return backend;
}

static struct hwloc_disc_component hwloc_windows_disc_component = {
  "windows",
  HWLOC_DISC_PHASE_CPU,
  HWLOC_DISC_PHASE_GLOBAL,
  hwloc_windows_component_instantiate,
  50,
  1,
  NULL
};

const struct hwloc_component hwloc_windows_component = {
  HWLOC_COMPONENT_ABI,
  hwloc_windows_component_init, hwloc_windows_component_finalize,
  HWLOC_COMPONENT_TYPE_DISC,
  0,
  &hwloc_windows_disc_component
};

int
hwloc_fallback_nbprocessors(unsigned flags __hwloc_attribute_unused) {
  int n;
  SYSTEM_INFO sysinfo;

  /* TODO handle flags & HWLOC_FALLBACK_NBPROCESSORS_INCLUDE_OFFLINE */

  /* by default, ignore groups (return only the number in the current group) */
  GetSystemInfo(&sysinfo);
  n = sysinfo.dwNumberOfProcessors; /* FIXME could be non-contigous, rather return a mask from dwActiveProcessorMask? */

  if (nr_processor_groups > 1) {
    /* assume n-1 groups are complete, since that's how we store things in cpusets */
    if (GetActiveProcessorCountProc)
      n = MAXIMUM_PROC_PER_GROUP*(nr_processor_groups-1)
	+ GetActiveProcessorCountProc((WORD)nr_processor_groups-1);
    else
      n = MAXIMUM_PROC_PER_GROUP*nr_processor_groups;
  }

  return n;
}

int64_t
hwloc_fallback_memsize(void) {
  /* Unused */
  return -1;
}
