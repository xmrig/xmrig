# Processor Groups Fix — Changes Explained

## Summary

This branch (`fix/processor-groups-numa`) fixes NUMA memory binding and thread affinity on Windows systems with more than 64 logical processors (multiple processor groups). On such systems, XMRIG previously failed to bind RandomX datasets to their correct NUMA nodes, causing the miner to fall back to slow mode or skip dataset allocation entirely.

The fix adds a new API method, improves fallback behavior, and handles CPUs beyond group 0 when hwloc is disabled. All changes are backward-compatible — systems with fewer than 64 CPUs see no behavioral change.

---

## Non-Technical Overview

### The Problem

Windows splits processors into "groups" of up to 64 each. A standard desktop or server has one group (CPU IDs 0–63). Large servers with 128, 256, or more CPUs span multiple groups (group 0: CPUs 0–63, group 1: CPUs 64–127, etc.).

XMRIG uses NUMA binding to keep each miner thread's RandomX dataset in the memory physically closest to that CPU. This is critical for performance — without it, the miner accesses remote memory across sockets or groups, which can be 3-5x slower.

On Windows with multiple processor groups, XMRIG's NUMA binding code had a bug: when it tried to bind memory to a NUMA node, an internal conversion step failed silently, and the function gave up entirely — no memory binding, no thread affinity, nothing. The miner would then skip dataset allocation for that node or fall back to slow mode.

### What This Fix Does

1. **Uses the right API directly** — Instead of converting between different address-space representations (which caused the failure), we now call hwloc's native NUMA-node binding function directly.
2. **Adds fallback layers** — If the best method fails, we try progressively simpler methods instead of giving up immediately. Even partial binding is better than none.
3. **Handles CPUs beyond ID 63** — When hwloc is disabled (rare), the old code used a Windows API that only works for CPU IDs 0–63. The new code uses the modern `SetThreadGroupAffinity` API for higher-numbered CPUs.

### Who Benefits

- **Multi-CPU servers** (128+ cores) running XMRIG on Windows — this is where the bug manifests
- **All other systems** — no change in behavior; they use the primary code path which already worked correctly

---

## Technical Details

### Files Changed (6 files, ~74 lines added, 9 removed)

#### 1. `src/backend/cpu/interfaces/ICpuInfo.h`

**What changed:**
- Added typedef: `using hwloc_const_nodeset_t = hwloc_const_bitmap_t;`
- Renamed parameter in `membind()` from `nodeset` to `cpuset` (it was misnamed — the type is a cpuset, not a nodeset)
- Added new pure virtual method: `virtual bool membind_nodeset(hwloc_const_nodeset_t nodeset) = 0;`

**Why:** XMRIG defines its own hwloc type aliases but was missing `hwloc_const_nodeset_t`. The old `membind()` signature used the wrong name, making it confusing and error-prone. The new `membind_nodeset()` method lets callers pass a NUMA node's nodeset directly without going through an unnecessary conversion.

#### 2. `src/backend/cpu/platform/HwlocCpuInfo.h`

**What changed:**
- Renamed parameter in `membind()` override from `nodeset` to `cpuset`
- Added declaration: `bool membind_nodeset(hwloc_const_nodeset_t nodeset) override;`

**Why:** Matches the interface change. Declares the new method implementation.

#### 3. `src/backend/cpu/platform/HwlocCpuInfo.cpp`

**What changed:**
- Renamed parameter in existing `membind()` from `nodeset` to `cpuset` throughout
- Added new `membind_nodeset()` implementation that calls `hwloc_set_membind_nodeset()` directly

```cpp
bool xmrig::HwlocCpuInfo::membind_nodeset(hwloc_const_nodeset_t nodeset)
{
    if (!hwloc_topology_get_support(m_topology)->membind->set_thisthread_membind) {
        return false;
    }
    return hwloc_set_membind_nodeset(m_topology, nodeset, HWLOC_MEMBIND_BIND, HWLOC_MEMBIND_THREAD) >= 0;
}
```

**Why:** The old `membind()` called `hwloc_set_membind()` with the `HWLOC_MEMBIND_BYNODESET` flag. On Windows, this internally converts the bitmap to a nodeset and back — a round-trip that fails when bits span multiple processor groups (the `hwloc_bitmap_to_single_ULONG_PTR()` check in hwloc's Windows backend rejects cross-group bitmaps). The new method calls `hwloc_set_membind_nodeset()` directly, bypassing this conversion entirely.

#### 4. `src/crypto/rx/RxNUMAStorage.cpp`

**What changed:**
- `bindToNUMANode()` now tries three methods in sequence:
  1. `membind_nodeset(node->nodeset)` — direct nodeset binding (best)
  2. `membind(node->cpuset)` — cpuset-based fallback (existing path)
  3. `Platform::setThreadAffinity(first_cpu_in_node)` — thread affinity only (last resort)

**Why:** Previously, if `membind()` failed, the function returned false immediately with no binding at all. This meant the worker thread was completely unbound and could end up on any CPU, potentially far from its dataset's NUMA node. The new fallback chain ensures that even in worst-case scenarios, the thread is at least bound to *some* CPU within the correct NUMA node.

#### 5. `src/crypto/common/VirtualMemory_hwloc.cpp`

**What changed:**
- Separated the null check from the membind call for clarity
- Added two fallback paths: `membind_nodeset()` first, then `membind(cpuset)` as a secondary fallback
- Only returns node 0 (wrong NUMA node) if both binding methods fail

```cpp
// Try direct nodeset binding first (avoids cpuset round-trip on Windows processor groups)
if (Cpu::info()->membind_nodeset(pu->nodeset)) {
    return hwloc_bitmap_first(pu->nodeset);
}

// Fallback to cpuset-based binding
if (Cpu::info()->membind(pu->cpuset)) {
    return hwloc_bitmap_first(pu->nodeset);
}
```

**Why:** Same pattern as RxNUMAStorage — the old code returned node 0 immediately if `membind()` failed, potentially allocating memory on the wrong NUMA node. The new code tries multiple methods before giving up.

#### 6. `src/base/kernel/Platform_win.cpp` (Windows-only)

**What changed:**
- When hwloc is disabled (`#ifndef XMRIG_FEATURE_HWLOC`) and CPU ID >= 64, uses dynamic loading of `SetThreadGroupAffinity()` from kernel32.dll
- Falls back to the legacy `SetThreadAffinityMask()` for CPUs < 64 or if the function isn't available

```cpp
if (cpu_id >= 64) {
    typedef BOOL (WINAPI *PFN_SETTHREADGROUPAFFINITY)(HANDLE, const GROUP_AFFINITY*, PGROUP_AFFINITY);
    static PFN_SETTHREADGROUPAFFINITY pSetThreadGroupAffinity = nullptr;
    
    if (!pSetThreadGroupAffinity) {
        HMODULE kernel32 = GetModuleHandleW(L"kernel32.dll");
        pSetThreadGroupAffinity = reinterpret_cast<PFN_SETTHREADGROUPAFFINITY>(GetProcAddress(kernel32, "SetThreadGroupAffinity"));
    }
    
    if (pSetThreadGroupAffinity) {
        WORD group = static_cast<WORD>(cpu_id / 64);
        ULONG_PTR mask = 1ULL << (cpu_id % 64);
        GROUP_AFFINITY aff{};
        aff.Group     = group;
        aff.Mask      = mask;
        const bool result = pSetThreadGroupAffinity(GetCurrentThread(), &aff, nullptr) != 0;
        Sleep(1);
        return result;
    }
}
```

**Why:** The legacy `SetThreadAffinityMask()` API only accepts a 64-bit mask — it can't address CPUs beyond ID 63. Starting with Windows Server 2008 R2 SP1, Microsoft introduced `SetThreadGroupAffinity()` which takes a `GROUP_AFFINITY` structure specifying both the group number and the bit within that group. We dynamically load this function (to maintain compatibility with older Windows versions) and use it when CPU >= 64.

---

## How It Works: The Call Chain

```
Worker constructor (backend/common/Worker.cpp)
│
├── VirtualMemory::bindToNUMANode(affinity)
│   ├── membind_nodeset(pu->nodeset)     ← NEW: direct nodeset binding
│   └── membind(pu->cpuset)              ← existing path (fallback)
│        └── hwloc_set_membind()         ← Windows backend uses SetThreadGroupAffinity
│
├── Platform::trySetThreadAffinity(affinity)
│   └── Platform::setThreadAffinity(cpu_id)
│       ├── Platform_hwloc.cpp: hwloc_set_cpubind()  ← handles groups correctly (already worked)
│       └── Platform_win.cpp (no hwloc):
│           ├── SetThreadGroupAffinity() for CPU >= 64  ← NEW
│           └── SetThreadAffinityMask() for CPU < 64    ← existing

RxNUMAStoragePrivate::allocate() (crypto/rx/RxNUMAStorage.cpp)
├── bindToNUMANode(nodeId)
│   ├── membind_nodeset(node->nodeset)     ← NEW: direct nodeset binding
│   └── membind(node->cpuset)              ← existing path (fallback)
│        └── Platform::setThreadAffinity()  ← thread affinity after memory bind
├── [NEW fallback] setThreadAffinity(first_cpu_in_node)  ← last resort
└── new RxDataset(..., nodeId)             ← dataset allocated on correct NUMA node
```

---

## Why the Old Code Failed on Windows Processor Groups

The root cause is a conversion round-trip in hwloc's Windows backend:

1. XMRIG called `membind()` passing a **nodeset** (but named it "nodeset" even though the type was `hwloc_const_bitmap_t` — a cpuset)
2. On Windows, hwloc internally converts nodesets to cpusets via `hwloc_cpuset_from_nodeset()`
3. The Windows backend then tries to convert that cpuset back to a single ULONG_PTR mask via `hwloc_bitmap_to_single_ULONG_PTR()`
4. **This fails** when the cpuset spans multiple processor groups (bits in both group 0 and group 1, for example) because a single ULONG_PTR can only represent 64 bits
5. The conversion returns -1, binding fails, and XMRIG's callers gave up entirely

The fix bypasses step 2-3 by calling `hwloc_set_membind_nodeset()` directly with the original nodeset, which hwloc handles natively on Windows without requiring a single-mask representation.

---

## Verification

### On multi-CPU systems (128+ cores):
```bash
xmrig --url=stratum+tcp://pool.example.com:3333 --user=WALLET --pass=x --print-time=60
```
Look for "dataset ready" messages per NUMA node in the output. Previously these would show as "skipped (can't bind memory)".

### On standard systems (<64 CPUs):
No behavioral change expected. The new `membind_nodeset()` path succeeds wherever the old path did, and the fallback chain is never exercised.

### Debugging:
```bash
set HWLOC_DEBUG=1
xmrig --url=...
```
Prints processor group detection and binding decisions to stdout.

---

## Backward Compatibility

- **Linux/macOS:** No change — these platforms don't have the Windows-specific round-trip issue, but benefit from the cleaner API
- **Windows <64 CPUs:** No change — `membind_nodeset()` succeeds on the first try, fallbacks are never used
- **Windows 64+ CPUs with hwloc enabled:** Fixed — direct nodeset binding works correctly
- **Windows 64+ CPUs without hwloc:** Fixed — `SetThreadGroupAffinity` handles groups >0
