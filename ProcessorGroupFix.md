# Windows Processor Groups Fix for XMRIG NUMA Binding

## Problem Statement

On Windows systems with more than 64 logical processors (multiple processor groups), XMRIG's NUMA memory binding and thread affinity mechanisms fail. This breaks RandomX dataset allocation on multi-socket/multi-group servers, causing the miner to fall back to slow mode or crash.

## Root Cause Analysis

### Background: Windows Processor Groups

Windows historically limited each process/thread affinity mask to 64 bits (one processor group). Starting with Windows Server 2008 R2 SP1 and Windows 7 SP1, Microsoft introduced `SetThreadGroupAffinity()` to allow binding threads across groups. However, the legacy `SetThreadAffinityMask()` API still only handles a single 64-bit mask within the current group.

XMRIG bundles hwloc 2.12.0 (API version 0x00020c00), which has proper support for NUMA across processor groups since v2.7.0. The problem is that XMRIG's own code paths bypass or break this support in several places.

### Issue #1: `RxNUMAStorage.cpp::bindToNUMANode()` — No Fallback on membind Failure

**File:** `src/crypto/rx/RxNUMAStorage.cpp` (lines 45-60)

```cpp
static bool bindToNUMANode(uint32_t nodeId)
{
    auto node = hwloc_get_numanode_obj_by_os_index(Cpu::info()->topology(), nodeId);
    if (!node) {
        return false;
    }

    if (Cpu::info()->membind(node->nodeset)) {
        Platform::setThreadAffinity(static_cast<uint64_t>(hwloc_bitmap_first(node->cpuset)));
        return true;
    }

    return false;  // <-- PROBLEM: gives up entirely, no thread affinity set
}
```

When `membind()` fails (which can happen on Windows with processor groups due to cpuset round-trip conversion issues), the function returns `false` without setting any thread affinity. This means the worker thread is completely unbound and may end up on a NUMA node far from its dataset, or worse — the entire dataset allocation for that NUMA node is skipped.

### Issue #2: `VirtualMemory_hwloc.cpp::bindToNUMANode()` — Same No-Fallback Problem

**File:** `src/crypto/common/VirtualMemory_hwloc.cpp` (lines 29-43)

```cpp
uint32_t xmrig::VirtualMemory::bindToNUMANode(int64_t affinity)
{
    if (affinity < 0 || Cpu::info()->nodes() < 2) {
        return 0;
    }

    auto pu = hwloc_get_pu_obj_by_os_index(Cpu::info()->topology(), static_cast<unsigned>(affinity));

    if (pu == nullptr || !Cpu::info()->membind(pu->nodeset)) {
        LOG_WARN("CPU #%02" PRId64 " warning: \"can't bind memory\"", affinity);
        return 0;  // <-- PROBLEM: no fallback, returns node 0
    }

    return hwloc_bitmap_first(pu->nodeset);
}
```

Same pattern — when `membind()` fails, the function logs a warning and returns 0 (node 0), potentially binding memory to the wrong NUMA node.

### Issue #3: `HwlocCpuInfo::membind()` — Unnecessary cpuset Round-Trip

**File:** `src/backend/cpu/platform/HwlocCpuInfo.cpp` (lines 194-205)

```cpp
bool xmrig::HwlocCpuInfo::membind(hwloc_const_bitmap_t nodeset)  // <-- misnamed: takes cpuset, calls it "nodeset"
{
    if (!hwloc_topology_get_support(m_topology)->membind->set_thisthread_membind) {
        return false;
    }

#   if HWLOC_API_VERSION >= 0x20000
    return hwloc_set_membind(m_topology, nodeset, HWLOC_MEMBIND_BIND, HWLOC_MEMBIND_THREAD | HWLOC_MEMBIND_BYNODESET) >= 0;
#   else
    return hwloc_set_membind_nodeset(m_topology, nodeset, HWLOC_MEMBIND_BIND, HWLOC_MEMBIND_THREAD) >= 0;
#   endif
}
```

The parameter is named `nodeset` but its type is `hwloc_const_bitmap_t` (a cpuset). When callers pass a NUMA node's `nodeset` field (which is actually a nodeset), hwloc internally converts it to a cpuset via `hwloc_cpuset_from_nodeset()`, then the Windows backend in hwloc tries to convert that back. This round-trip can fail on Windows when:

1. The resulting cpuset spans multiple processor groups (the `hwloc_bitmap_to_single_ULONG_PTR()` check fails)
2. The cpuset bits fall outside a single ULONG_PTR range within one group

**The fix:** Add a separate `membind_nodeset()` method that uses hwloc's direct nodeset API (`hwloc_set_membind_nodeset`), avoiding the unnecessary round-trip conversion entirely.

### Issue #4: `Platform_win.cpp` Fallback — `SetThreadAffinityMask()` Can't Handle CPU >= 64

**File:** `src/base/kernel/Platform_win.cpp` (lines 86-93)

```cpp
#ifndef XMRIG_FEATURE_HWLOC
bool xmrig::Platform::setThreadAffinity(uint64_t cpu_id)
{
    const bool result = (SetThreadAffinityMask(GetCurrentThread(), 1ULL << cpu_id) != 0);
    Sleep(1);
    return result;
}
#endif
```

When hwloc is disabled, this fallback uses `SetThreadAffinityMask()` which only works for CPU IDs 0-63. On systems with more than 64 CPUs and no hwloc, any thread affinity request for CPU >= 64 silently fails.

### Issue #5: `Platform_hwloc.cpp` — Also Missing Fallback

**File:** `src/base/kernel/Platform_hwloc.cpp` (lines 23-44)

```cpp
bool xmrig::Platform::setThreadAffinity(uint64_t cpu_id)
{
    auto topology = Cpu::info()->topology();
    auto pu       = hwloc_get_pu_obj_by_os_index(topology, static_cast<unsigned>(cpu_id));

    if (pu == nullptr) {
        return false;  // <-- PROBLEM: no fallback for out-of-range CPU IDs
    }

    if (hwloc_set_cpubind(topology, pu->cpuset, HWLOC_CPUBIND_THREAD | HWLOC_CPUBIND_STRICT) >= 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        return true;
    }

    const bool result = (hwloc_set_cpubind(topology, pu->cpuset, HWLOC_CPUBIND_THREAD) >= 0);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    return result;
}
```

When `HWLOC_CPUBIND_STRICT` fails, it falls back to non-strict binding. This is actually reasonable behavior — the hwloc backend handles processor groups correctly via `SetThreadGroupAffinity`. The real problem is that callers don't handle the case where this entire function returns false.

## Call Chain Analysis

The NUMA binding flow on Windows:

```
Worker constructor (backend/common/Worker.cpp)
  ├── VirtualMemory::bindToNUMANode(affinity)
  │     └── HwlocCpuInfo::membind(pu->nodeset)   ← Issue #3: round-trip conversion
  │          └── hwloc_set_membind() / hwloc_set_membind_nodeset()
  │               └── Windows backend (hwloc/src/topology-windows.c)
  │                    ├── hwloc_win_set_thisthread_cpubind()
  │                    │     └── SetThreadGroupAffinity() or SetThreadAffinityMask()
  │                    └── hwloc_bitmap_to_single_ULONG_PTR() ← can fail for cross-group cpusets
  │
  └── Platform::trySetThreadAffinity(affinity)
        └── Platform::setThreadAffinity(cpu_id)
              ├── Platform_hwloc.cpp: hwloc_set_cpubind()   ← handles groups correctly
              └── Platform_win.cpp (no hwloc): SetThreadAffinityMask()  ← Issue #4: fails for CPU>=64

RxNUMAStoragePrivate::allocate() (crypto/rx/RxNUMAStorage.cpp)
  ├── bindToNUMANode(nodeId)   ← Issue #1: no fallback on membind failure
  │     └── HwlocCpuInfo::membind(node->nodeset)
  │          └── same chain as above
  └── new RxDataset(hugePages, oneGbPages, false, RxConfig::FastMode, nodeId)
```

## Proposed Fixes

### Fix #1: Add `membind_nodeset()` to ICpuInfo interface and HwlocCpuInfo implementation

**Files:** 
- `src/backend/cpu/interfaces/ICpuInfo.h` — add virtual method
- `src/backend/cpu/platform/HwlocCpuInfo.h` — declare override
- `src/backend/cpu/platform/HwlocCpuInfo.cpp` — implement using direct nodeset API

```cpp
// In ICpuInfo.h, alongside existing membind():
virtual bool membind_nodeset(hwloc_const_nodeset_t nodeset) = 0;

// In HwlocCpuInfo.cpp:
bool xmrig::HwlocCpuInfo::membind_nodeset(hwloc_const_nodeset_t nodeset)
{
    if (!hwloc_topology_get_support(m_topology)->membind->set_thisthread_membind) {
        return false;
    }

#   if HWLOC_API_VERSION >= 0x20000
    return hwloc_set_membind_nodeset(m_topology, nodeset, HWLOC_MEMBIND_BIND, HWLOC_MEMBIND_THREAD) >= 0;
#   else
    // Pre-2.0 API: use the existing path (already uses nodeset internally)
    return hwloc_set_membind_nodeset(m_topology, nodeset, HWLOC_MEMBIND_BIND, HWLOC_MEMBIND_THREAD) >= 0;
#   endif
}
```

### Fix #2: Update callers to use `membind_nodeset()` directly

**Files:**
- `src/crypto/rx/RxNUMAStorage.cpp` — change `Cpu::info()->membind(node->nodeset)` to `Cpu::info()->membind_nodeset(node->nodeset)`
- `src/crypto/common/VirtualMemory_hwloc.cpp` — same change

This avoids the unnecessary cpuset round-trip entirely.

### Fix #3: Add fallback thread affinity in `RxNUMAStorage.cpp::bindToNUMANode()`

**File:** `src/crypto/rx/RxNUMAStorage.cpp` (lines 45-60)

```cpp
static bool bindToNUMANode(uint32_t nodeId)
{
    auto node = hwloc_get_numanode_obj_by_os_index(Cpu::info()->topology(), nodeId);
    if (!node) {
        return false;
    }

    // Try direct nodeset binding first (avoids cpuset round-trip on Windows processor groups)
    if (Cpu::info()->membind_nodeset(node->nodeset)) {
        Platform::setThreadAffinity(static_cast<uint64_t>(hwloc_bitmap_first(node->cpuset)));
        return true;
    }

    // Fallback: try cpuset-based binding
    if (Cpu::info()->membind(node->cpuset)) {
        Platform::setThreadAffinity(static_cast<uint64_t>(hwloc_bitmap_first(node->cpuset)));
        return true;
    }

    // Last resort: at least set thread affinity to a CPU in this NUMA node
    int first_cpu = hwloc_bitmap_first(node->cpuset);
    if (first_cpu >= 0) {
        Platform::setThreadAffinity(static_cast<uint64_t>(first_cpu));
        return true;
    }

    return false;
}
```

### Fix #4: Add fallback thread affinity in `VirtualMemory_hwloc.cpp`

**File:** `src/crypto/common/VirtualMemory_hwloc.cpp` (lines 29-43)

```cpp
uint32_t xmrig::VirtualMemory::bindToNUMANode(int64_t affinity)
{
    if (affinity < 0 || Cpu::info()->nodes() < 2) {
        return 0;
    }

    auto pu = hwloc_get_pu_obj_by_os_index(Cpu::info()->topology(), static_cast<unsigned>(affinity));

    if (pu == nullptr) {
        LOG_WARN("CPU #%02" PRId64 " warning: \"PU not found\"", affinity);
        return 0;
    }

    // Try direct nodeset binding first
    if (Cpu::info()->membind_nodeset(pu->nodeset)) {
        return hwloc_bitmap_first(pu->nodeset);
    }

    // Fallback to cpuset-based binding
    if (Cpu::info()->membind(pu->cpuset)) {
        return hwloc_bitmap_first(pu->nodeset);
    }

    LOG_WARN("CPU #%02" PRId64 " warning: \"can't bind memory\"", affinity);
    return 0;
}
```

### Fix #5: Update `Platform_win.cpp` to use `SetThreadGroupAffinity()` for CPU >= 64

**File:** `src/base/kernel/Platform_win.cpp` (lines 86-93)

When hwloc is disabled, the fallback should handle CPUs beyond group 0:

```cpp
#ifndef XMRIG_FEATURE_HWLOC
bool xmrig::Platform::setThreadAffinity(uint64_t cpu_id)
{
    if (cpu_id >= 64) {
        // Use SetThreadGroupAffinity for CPUs in groups > 0
        typedef BOOL (WINAPI *PFN_SETTHREADGROUPAFFINITY)(HANDLE, const GROUP_AFFINITY*, PGROUP_AFFINITY);
        static PFN_SETTHREADGROUPAFFINITY pSetThreadGroupAffinity = nullptr;
        
        if (!pSetThreadGroupAffinity) {
            HMODULE kernel32 = GetModuleHandleW(L"kernel32.dll");
            pSetThreadGroupAffinity = 
                (PFN_SETTHREADGROUPAFFINITY)GetProcAddress(kernel32, "SetThreadGroupAffinity");
        }
        
        if (pSetThreadGroupAffinity) {
            WORD group = static_cast<WORD>(cpu_id / 64);
            ULONG_PTR mask = 1ULL << (cpu_id % 64);
            GROUP_AFFINITY aff{};
            aff.Group = group;
            aff.Mask = mask;
            const bool result = pSetThreadGroupAffinity(GetCurrentThread(), &aff, nullptr) != 0;
            Sleep(1);
            return result;
        }
    }

    // Fallback for CPU < 64 (or if SetThreadGroupAffinity unavailable)
    const bool result = (SetThreadAffinityMask(GetCurrentThread(), 1ULL << cpu_id) != 0);
    Sleep(1);
    return result;
}
#endif
```

## Verification Plan

### Test Environment
- Windows Server with >64 logical processors (multiple processor groups)
- Multi-socket system with NUMA nodes spanning processor groups

### Manual Testing Steps

1. **Build XMRIG** with hwloc enabled (`WITH_HWLOC=ON`, default)
2. **Run on multi-group system:**
   ```bash
   xmrig --url=stratum+tcp://pool.example.com:3333 --user=WALLET --pass=x --print-time=10
   ```
3. **Verify NUMA binding in logs:** Look for "dataset ready" messages per NUMA node (no "skipped" warnings)
4. **Check thread affinity:** Use `Process Explorer` or `GetThreadGroupAffinity` debugging to verify threads are bound correctly

### Debugging Aid
Set the hwloc debug environment variable:
```bash
set HWLOC_DEBUG=1
xmrig --url=...
```
This will print processor group detection and binding decisions.

For topology export (to inspect the detected NUMA layout):
```bash
xmrig --export-topology
```
This writes `topology.xml` to the executable directory, which can be inspected with `lstopo`.

### Regression Test (single-socket / <64 CPU systems)
1. Build and run on a standard desktop/server (<64 CPUs)
2. Verify NUMA binding still works correctly (if multi-socket)
3. Verify single-NUMA behavior is unchanged
4. Test with hwloc disabled (`WITH_HWLOC=OFF`) to ensure `SetThreadAffinityMask` fallback works

### CI / Automated Check
Add a test that:
1. Detects Windows processor group count via `GetActiveProcessorGroupCount()`
2. If >1, verifies XMRIG starts without NUMA binding errors in logs
3. Verifies dataset allocation succeeds for all detected NUMA nodes

## Files to Modify Summary

| File | Change |
|------|--------|
| `src/backend/cpu/interfaces/ICpuInfo.h` | Add `membind_nodeset()` virtual method |
| `src/backend/cpu/platform/HwlocCpuInfo.h` | Declare `membind_nodeset()` override |
| `src/backend/cpu/platform/HwlocCpuInfo.cpp` | Implement `membind_nodeset()`, rename param in existing `membind()` from `nodeset` to `cpuset` for clarity |
| `src/crypto/rx/RxNUMAStorage.cpp` | Use `membind_nodeset()`, add fallback thread affinity |
| `src/crypto/common/VirtualMemory_hwloc.cpp` | Use `membind_nodeset()`, add fallback binding path |
| `src/base/kernel/Platform_win.cpp` | Handle CPU >= 64 via `SetThreadGroupAffinity()` when hwloc disabled |

## Notes on hwloc Internal Behavior

The bundled hwloc (2.12.0) already handles Windows processor groups correctly in its backend:

- **`hwloc/src/topology-windows.c::hwloc_win_set_thread_cpubind()`** — Uses `SetThreadGroupAffinity()` when `nr_processor_groups > 1`, falls back to `SetThreadAffinityMask()` for single-group systems
- **`hwloc/src/topology-windows.c::hwloc_bitmap_to_single_ULONG_PTR()`** — Returns -1 if bits span multiple ULONG_PTR ranges (i.e., cross group boundary), which causes the cpuset-based binding path to fail gracefully

The key insight is that `hwloc_set_membind_nodeset()` bypasses the problematic cpuset round-trip entirely, making it the correct API for XMRIG's use case where we already have a nodeset from `node->nodeset`.
