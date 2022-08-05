/*
 * Copyright © 2009 CNRS
 * Copyright © 2009-2020 Inria.  All rights reserved.
 * Copyright © 2009-2010, 2012 Université Bordeaux
 * Copyright © 2011-2015 Cisco Systems, Inc.  All rights reserved.
 * See COPYING in top-level directory.
 */

#include "private/autogen/config.h"
#include "hwloc.h"
#include "private/private.h"
#include "hwloc/helper.h"

#ifdef HAVE_SYS_MMAN_H
#  include <sys/mman.h>
#endif
/* <malloc.h> is only needed if we don't have posix_memalign() */
#if defined(hwloc_getpagesize) && !defined(HAVE_POSIX_MEMALIGN) && defined(HAVE_MEMALIGN) && defined(HAVE_MALLOC_H)
#include <malloc.h>
#endif
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif
#include <stdlib.h>
#include <errno.h>

/* TODO: HWLOC_GNU_SYS,
 *
 * We could use glibc's sched_setaffinity generically when it is available
 *
 * Darwin and OpenBSD don't seem to have binding facilities.
 */

#define HWLOC_CPUBIND_ALLFLAGS (HWLOC_CPUBIND_PROCESS|HWLOC_CPUBIND_THREAD|HWLOC_CPUBIND_STRICT|HWLOC_CPUBIND_NOMEMBIND)

static hwloc_const_bitmap_t
hwloc_fix_cpubind(hwloc_topology_t topology, hwloc_const_bitmap_t set)
{
  hwloc_const_bitmap_t topology_set = hwloc_topology_get_topology_cpuset(topology);
  hwloc_const_bitmap_t complete_set = hwloc_topology_get_complete_cpuset(topology);

  if (hwloc_bitmap_iszero(set)) {
    errno = EINVAL;
    return NULL;
  }

  if (!hwloc_bitmap_isincluded(set, complete_set)) {
    errno = EINVAL;
    return NULL;
  }

  if (hwloc_bitmap_isincluded(topology_set, set))
    set = complete_set;

  return set;
}

int
hwloc_set_cpubind(hwloc_topology_t topology, hwloc_const_bitmap_t set, int flags)
{
  if (flags & ~HWLOC_CPUBIND_ALLFLAGS) {
    errno = EINVAL;
    return -1;
  }

  set = hwloc_fix_cpubind(topology, set);
  if (!set)
    return -1;

  if (flags & HWLOC_CPUBIND_PROCESS) {
    if (topology->binding_hooks.set_thisproc_cpubind)
      return topology->binding_hooks.set_thisproc_cpubind(topology, set, flags);
  } else if (flags & HWLOC_CPUBIND_THREAD) {
    if (topology->binding_hooks.set_thisthread_cpubind)
      return topology->binding_hooks.set_thisthread_cpubind(topology, set, flags);
  } else {
    if (topology->binding_hooks.set_thisproc_cpubind) {
      int err = topology->binding_hooks.set_thisproc_cpubind(topology, set, flags);
      if (err >= 0 || errno != ENOSYS)
        return err;
      /* ENOSYS, fallback */
    }
    if (topology->binding_hooks.set_thisthread_cpubind)
      return topology->binding_hooks.set_thisthread_cpubind(topology, set, flags);
  }

  errno = ENOSYS;
  return -1;
}

int
hwloc_get_cpubind(hwloc_topology_t topology, hwloc_bitmap_t set, int flags)
{
  if (flags & ~HWLOC_CPUBIND_ALLFLAGS) {
    errno = EINVAL;
    return -1;
  }

  if (flags & HWLOC_CPUBIND_PROCESS) {
    if (topology->binding_hooks.get_thisproc_cpubind)
      return topology->binding_hooks.get_thisproc_cpubind(topology, set, flags);
  } else if (flags & HWLOC_CPUBIND_THREAD) {
    if (topology->binding_hooks.get_thisthread_cpubind)
      return topology->binding_hooks.get_thisthread_cpubind(topology, set, flags);
  } else {
    if (topology->binding_hooks.get_thisproc_cpubind) {
      int err = topology->binding_hooks.get_thisproc_cpubind(topology, set, flags);
      if (err >= 0 || errno != ENOSYS)
        return err;
      /* ENOSYS, fallback */
    }
    if (topology->binding_hooks.get_thisthread_cpubind)
      return topology->binding_hooks.get_thisthread_cpubind(topology, set, flags);
  }

  errno = ENOSYS;
  return -1;
}

int
hwloc_set_proc_cpubind(hwloc_topology_t topology, hwloc_pid_t pid, hwloc_const_bitmap_t set, int flags)
{
  if (flags & ~HWLOC_CPUBIND_ALLFLAGS) {
    errno = EINVAL;
    return -1;
  }

  set = hwloc_fix_cpubind(topology, set);
  if (!set)
    return -1;

  if (topology->binding_hooks.set_proc_cpubind)
    return topology->binding_hooks.set_proc_cpubind(topology, pid, set, flags);

  errno = ENOSYS;
  return -1;
}

int
hwloc_get_proc_cpubind(hwloc_topology_t topology, hwloc_pid_t pid, hwloc_bitmap_t set, int flags)
{
  if (flags & ~HWLOC_CPUBIND_ALLFLAGS) {
    errno = EINVAL;
    return -1;
  }

  if (topology->binding_hooks.get_proc_cpubind)
    return topology->binding_hooks.get_proc_cpubind(topology, pid, set, flags);

  errno = ENOSYS;
  return -1;
}

#ifdef hwloc_thread_t
int
hwloc_set_thread_cpubind(hwloc_topology_t topology, hwloc_thread_t tid, hwloc_const_bitmap_t set, int flags)
{
  if (flags & ~HWLOC_CPUBIND_ALLFLAGS) {
    errno = EINVAL;
    return -1;
  }

  set = hwloc_fix_cpubind(topology, set);
  if (!set)
    return -1;

  if (topology->binding_hooks.set_thread_cpubind)
    return topology->binding_hooks.set_thread_cpubind(topology, tid, set, flags);

  errno = ENOSYS;
  return -1;
}

int
hwloc_get_thread_cpubind(hwloc_topology_t topology, hwloc_thread_t tid, hwloc_bitmap_t set, int flags)
{
  if (flags & ~HWLOC_CPUBIND_ALLFLAGS) {
    errno = EINVAL;
    return -1;
  }

  if (topology->binding_hooks.get_thread_cpubind)
    return topology->binding_hooks.get_thread_cpubind(topology, tid, set, flags);

  errno = ENOSYS;
  return -1;
}
#endif

int
hwloc_get_last_cpu_location(hwloc_topology_t topology, hwloc_bitmap_t set, int flags)
{
  if (flags & ~HWLOC_CPUBIND_ALLFLAGS) {
    errno = EINVAL;
    return -1;
  }

  if (flags & HWLOC_CPUBIND_PROCESS) {
    if (topology->binding_hooks.get_thisproc_last_cpu_location)
      return topology->binding_hooks.get_thisproc_last_cpu_location(topology, set, flags);
  } else if (flags & HWLOC_CPUBIND_THREAD) {
    if (topology->binding_hooks.get_thisthread_last_cpu_location)
      return topology->binding_hooks.get_thisthread_last_cpu_location(topology, set, flags);
  } else {
    if (topology->binding_hooks.get_thisproc_last_cpu_location) {
      int err = topology->binding_hooks.get_thisproc_last_cpu_location(topology, set, flags);
      if (err >= 0 || errno != ENOSYS)
        return err;
      /* ENOSYS, fallback */
    }
    if (topology->binding_hooks.get_thisthread_last_cpu_location)
      return topology->binding_hooks.get_thisthread_last_cpu_location(topology, set, flags);
  }

  errno = ENOSYS;
  return -1;
}

int
hwloc_get_proc_last_cpu_location(hwloc_topology_t topology, hwloc_pid_t pid, hwloc_bitmap_t set, int flags)
{
  if (flags & ~HWLOC_CPUBIND_ALLFLAGS) {
    errno = EINVAL;
    return -1;
  }

  if (topology->binding_hooks.get_proc_last_cpu_location)
    return topology->binding_hooks.get_proc_last_cpu_location(topology, pid, set, flags);

  errno = ENOSYS;
  return -1;
}

#define HWLOC_MEMBIND_ALLFLAGS (HWLOC_MEMBIND_PROCESS|HWLOC_MEMBIND_THREAD|HWLOC_MEMBIND_STRICT|HWLOC_MEMBIND_MIGRATE|HWLOC_MEMBIND_NOCPUBIND|HWLOC_MEMBIND_BYNODESET)

static hwloc_const_nodeset_t
hwloc_fix_membind(hwloc_topology_t topology, hwloc_const_nodeset_t nodeset)
{
  hwloc_const_bitmap_t topology_nodeset = hwloc_topology_get_topology_nodeset(topology);
  hwloc_const_bitmap_t complete_nodeset = hwloc_topology_get_complete_nodeset(topology);

  if (hwloc_bitmap_iszero(nodeset)) {
    errno = EINVAL;
    return NULL;
  }

  if (!hwloc_bitmap_isincluded(nodeset, complete_nodeset)) {
    errno = EINVAL;
    return NULL;
  }

  if (hwloc_bitmap_isincluded(topology_nodeset, nodeset))
    return complete_nodeset;

  return nodeset;
}

static int
hwloc_fix_membind_cpuset(hwloc_topology_t topology, hwloc_nodeset_t nodeset, hwloc_const_cpuset_t cpuset)
{
  hwloc_const_bitmap_t topology_set = hwloc_topology_get_topology_cpuset(topology);
  hwloc_const_bitmap_t complete_set = hwloc_topology_get_complete_cpuset(topology);
  hwloc_const_bitmap_t complete_nodeset = hwloc_topology_get_complete_nodeset(topology);

  if (hwloc_bitmap_iszero(cpuset)) {
    errno = EINVAL;
    return -1;
  }

  if (!hwloc_bitmap_isincluded(cpuset, complete_set)) {
    errno = EINVAL;
    return -1;
  }

  if (hwloc_bitmap_isincluded(topology_set, cpuset)) {
    hwloc_bitmap_copy(nodeset, complete_nodeset);
    return 0;
  }

  hwloc_cpuset_to_nodeset(topology, cpuset, nodeset);
  return 0;
}

static __hwloc_inline int hwloc__check_membind_policy(hwloc_membind_policy_t policy)
{
  if (policy == HWLOC_MEMBIND_DEFAULT
      || policy == HWLOC_MEMBIND_FIRSTTOUCH
      || policy == HWLOC_MEMBIND_BIND
      || policy == HWLOC_MEMBIND_INTERLEAVE
      || policy == HWLOC_MEMBIND_NEXTTOUCH)
    return 0;
  return -1;
}

static int
hwloc_set_membind_by_nodeset(hwloc_topology_t topology, hwloc_const_nodeset_t nodeset, hwloc_membind_policy_t policy, int flags)
{
  if ((flags & ~HWLOC_MEMBIND_ALLFLAGS) || hwloc__check_membind_policy(policy) < 0) {
    errno = EINVAL;
    return -1;
  }

  nodeset = hwloc_fix_membind(topology, nodeset);
  if (!nodeset)
    return -1;

  if (flags & HWLOC_MEMBIND_PROCESS) {
    if (topology->binding_hooks.set_thisproc_membind)
      return topology->binding_hooks.set_thisproc_membind(topology, nodeset, policy, flags);
  } else if (flags & HWLOC_MEMBIND_THREAD) {
    if (topology->binding_hooks.set_thisthread_membind)
      return topology->binding_hooks.set_thisthread_membind(topology, nodeset, policy, flags);
  } else {
    if (topology->binding_hooks.set_thisproc_membind) {
      int err = topology->binding_hooks.set_thisproc_membind(topology, nodeset, policy, flags);
      if (err >= 0 || errno != ENOSYS)
        return err;
      /* ENOSYS, fallback */
    }
    if (topology->binding_hooks.set_thisthread_membind)
      return topology->binding_hooks.set_thisthread_membind(topology, nodeset, policy, flags);
  }

  errno = ENOSYS;
  return -1;
}

int
hwloc_set_membind(hwloc_topology_t topology, hwloc_const_bitmap_t set, hwloc_membind_policy_t policy, int flags)
{
  int ret;

  if (flags & HWLOC_MEMBIND_BYNODESET) {
    ret = hwloc_set_membind_by_nodeset(topology, set, policy, flags);
  } else {
    hwloc_nodeset_t nodeset = hwloc_bitmap_alloc();
    if (hwloc_fix_membind_cpuset(topology, nodeset, set))
      ret = -1;
    else
      ret = hwloc_set_membind_by_nodeset(topology, nodeset, policy, flags);
    hwloc_bitmap_free(nodeset);
  }
  return ret;
}

static int
hwloc_get_membind_by_nodeset(hwloc_topology_t topology, hwloc_nodeset_t nodeset, hwloc_membind_policy_t * policy, int flags)
{
  if (flags & ~HWLOC_MEMBIND_ALLFLAGS) {
    errno = EINVAL;
    return -1;
  }

  if (flags & HWLOC_MEMBIND_PROCESS) {
    if (topology->binding_hooks.get_thisproc_membind)
      return topology->binding_hooks.get_thisproc_membind(topology, nodeset, policy, flags);
  } else if (flags & HWLOC_MEMBIND_THREAD) {
    if (topology->binding_hooks.get_thisthread_membind)
      return topology->binding_hooks.get_thisthread_membind(topology, nodeset, policy, flags);
  } else {
    if (topology->binding_hooks.get_thisproc_membind) {
      int err = topology->binding_hooks.get_thisproc_membind(topology, nodeset, policy, flags);
      if (err >= 0 || errno != ENOSYS)
        return err;
      /* ENOSYS, fallback */
    }
    if (topology->binding_hooks.get_thisthread_membind)
      return topology->binding_hooks.get_thisthread_membind(topology, nodeset, policy, flags);
  }

  errno = ENOSYS;
  return -1;
}

int
hwloc_get_membind(hwloc_topology_t topology, hwloc_bitmap_t set, hwloc_membind_policy_t * policy, int flags)
{
  int ret;

  if (flags & HWLOC_MEMBIND_BYNODESET) {
    ret = hwloc_get_membind_by_nodeset(topology, set, policy, flags);
  } else {
    hwloc_nodeset_t nodeset = hwloc_bitmap_alloc();
    ret = hwloc_get_membind_by_nodeset(topology, nodeset, policy, flags);
    if (!ret)
      hwloc_cpuset_from_nodeset(topology, set, nodeset);
    hwloc_bitmap_free(nodeset);
  }

  return ret;
}

static int
hwloc_set_proc_membind_by_nodeset(hwloc_topology_t topology, hwloc_pid_t pid, hwloc_const_nodeset_t nodeset, hwloc_membind_policy_t policy, int flags)
{
  if ((flags & ~HWLOC_MEMBIND_ALLFLAGS) || hwloc__check_membind_policy(policy) < 0) {
    errno = EINVAL;
    return -1;
  }

  nodeset = hwloc_fix_membind(topology, nodeset);
  if (!nodeset)
    return -1;

  if (topology->binding_hooks.set_proc_membind)
    return topology->binding_hooks.set_proc_membind(topology, pid, nodeset, policy, flags);

  errno = ENOSYS;
  return -1;
}


int
hwloc_set_proc_membind(hwloc_topology_t topology, hwloc_pid_t pid, hwloc_const_bitmap_t set, hwloc_membind_policy_t policy, int flags)
{
  int ret;

  if (flags & HWLOC_MEMBIND_BYNODESET) {
    ret = hwloc_set_proc_membind_by_nodeset(topology, pid, set, policy, flags);
  } else {
    hwloc_nodeset_t nodeset = hwloc_bitmap_alloc();
    if (hwloc_fix_membind_cpuset(topology, nodeset, set))
      ret = -1;
    else
      ret = hwloc_set_proc_membind_by_nodeset(topology, pid, nodeset, policy, flags);
    hwloc_bitmap_free(nodeset);
  }

  return ret;
}

static int
hwloc_get_proc_membind_by_nodeset(hwloc_topology_t topology, hwloc_pid_t pid, hwloc_nodeset_t nodeset, hwloc_membind_policy_t * policy, int flags)
{
  if (flags & ~HWLOC_MEMBIND_ALLFLAGS) {
    errno = EINVAL;
    return -1;
  }

  if (topology->binding_hooks.get_proc_membind)
    return topology->binding_hooks.get_proc_membind(topology, pid, nodeset, policy, flags);

  errno = ENOSYS;
  return -1;
}

int
hwloc_get_proc_membind(hwloc_topology_t topology, hwloc_pid_t pid, hwloc_bitmap_t set, hwloc_membind_policy_t * policy, int flags)
{
  int ret;

  if (flags & HWLOC_MEMBIND_BYNODESET) {
    ret = hwloc_get_proc_membind_by_nodeset(topology, pid, set, policy, flags);
  } else {
    hwloc_nodeset_t nodeset = hwloc_bitmap_alloc();
    ret = hwloc_get_proc_membind_by_nodeset(topology, pid, nodeset, policy, flags);
    if (!ret)
      hwloc_cpuset_from_nodeset(topology, set, nodeset);
    hwloc_bitmap_free(nodeset);
  }

  return ret;
}

static int
hwloc_set_area_membind_by_nodeset(hwloc_topology_t topology, const void *addr, size_t len, hwloc_const_nodeset_t nodeset, hwloc_membind_policy_t policy, int flags)
{
  if ((flags & ~HWLOC_MEMBIND_ALLFLAGS) || hwloc__check_membind_policy(policy) < 0) {
    errno = EINVAL;
    return -1;
  }

  if (!len)
    /* nothing to do */
    return 0;

  nodeset = hwloc_fix_membind(topology, nodeset);
  if (!nodeset)
    return -1;

  if (topology->binding_hooks.set_area_membind)
    return topology->binding_hooks.set_area_membind(topology, addr, len, nodeset, policy, flags);

  errno = ENOSYS;
  return -1;
}

int
hwloc_set_area_membind(hwloc_topology_t topology, const void *addr, size_t len, hwloc_const_bitmap_t set, hwloc_membind_policy_t policy, int flags)
{
  int ret;

  if (flags & HWLOC_MEMBIND_BYNODESET) {
    ret = hwloc_set_area_membind_by_nodeset(topology, addr, len, set, policy, flags);
  } else {
    hwloc_nodeset_t nodeset = hwloc_bitmap_alloc();
    if (hwloc_fix_membind_cpuset(topology, nodeset, set))
      ret = -1;
    else
      ret = hwloc_set_area_membind_by_nodeset(topology, addr, len, nodeset, policy, flags);
    hwloc_bitmap_free(nodeset);
  }

  return ret;
}

static int
hwloc_get_area_membind_by_nodeset(hwloc_topology_t topology, const void *addr, size_t len, hwloc_nodeset_t nodeset, hwloc_membind_policy_t * policy, int flags)
{
  if (flags & ~HWLOC_MEMBIND_ALLFLAGS) {
    errno = EINVAL;
    return -1;
  }

  if (!len) {
    /* nothing to query */
    errno = EINVAL;
    return -1;
  }

  if (topology->binding_hooks.get_area_membind)
    return topology->binding_hooks.get_area_membind(topology, addr, len, nodeset, policy, flags);

  errno = ENOSYS;
  return -1;
}

int
hwloc_get_area_membind(hwloc_topology_t topology, const void *addr, size_t len, hwloc_bitmap_t set, hwloc_membind_policy_t * policy, int flags)
{
  int ret;

  if (flags & HWLOC_MEMBIND_BYNODESET) {
    ret = hwloc_get_area_membind_by_nodeset(topology, addr, len, set, policy, flags);
  } else {
    hwloc_nodeset_t nodeset = hwloc_bitmap_alloc();
    ret = hwloc_get_area_membind_by_nodeset(topology, addr, len, nodeset, policy, flags);
    if (!ret)
      hwloc_cpuset_from_nodeset(topology, set, nodeset);
    hwloc_bitmap_free(nodeset);
  }

  return ret;
}

static int
hwloc_get_area_memlocation_by_nodeset(hwloc_topology_t topology, const void *addr, size_t len, hwloc_nodeset_t nodeset, int flags)
{
  if (flags & ~HWLOC_MEMBIND_ALLFLAGS) {
    errno = EINVAL;
    return -1;
  }

  if (!len)
    /* nothing to do */
    return 0;

  if (topology->binding_hooks.get_area_memlocation)
    return topology->binding_hooks.get_area_memlocation(topology, addr, len, nodeset, flags);

  errno = ENOSYS;
  return -1;
}

int
hwloc_get_area_memlocation(hwloc_topology_t topology, const void *addr, size_t len, hwloc_cpuset_t set, int flags)
{
  int ret;

  if (flags & HWLOC_MEMBIND_BYNODESET) {
    ret = hwloc_get_area_memlocation_by_nodeset(topology, addr, len, set, flags);
  } else {
    hwloc_nodeset_t nodeset = hwloc_bitmap_alloc();
    ret = hwloc_get_area_memlocation_by_nodeset(topology, addr, len, nodeset, flags);
    if (!ret)
      hwloc_cpuset_from_nodeset(topology, set, nodeset);
    hwloc_bitmap_free(nodeset);
  }

  return ret;
}

void *
hwloc_alloc_heap(hwloc_topology_t topology __hwloc_attribute_unused, size_t len)
{
  void *p = NULL;
#if defined(hwloc_getpagesize) && defined(HAVE_POSIX_MEMALIGN)
  errno = posix_memalign(&p, hwloc_getpagesize(), len);
  if (errno)
    p = NULL;
#elif defined(hwloc_getpagesize) && defined(HAVE_MEMALIGN)
  p = memalign(hwloc_getpagesize(), len);
#else
  p = malloc(len);
#endif
  return p;
}

#ifdef MAP_ANONYMOUS
void *
hwloc_alloc_mmap(hwloc_topology_t topology __hwloc_attribute_unused, size_t len)
{
  void * buffer = mmap(NULL, len, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
  return buffer == MAP_FAILED ? NULL : buffer;
}
#endif

int
hwloc_free_heap(hwloc_topology_t topology __hwloc_attribute_unused, void *addr, size_t len __hwloc_attribute_unused)
{
  free(addr);
  return 0;
}

#ifdef MAP_ANONYMOUS
int
hwloc_free_mmap(hwloc_topology_t topology __hwloc_attribute_unused, void *addr, size_t len)
{
  if (!addr)
    return 0;
  return munmap(addr, len);
}
#endif

void *
hwloc_alloc(hwloc_topology_t topology, size_t len)
{
  if (topology->binding_hooks.alloc)
    return topology->binding_hooks.alloc(topology, len);
  return hwloc_alloc_heap(topology, len);
}

static void *
hwloc_alloc_membind_by_nodeset(hwloc_topology_t topology, size_t len, hwloc_const_nodeset_t nodeset, hwloc_membind_policy_t policy, int flags)
{
  void *p;

  if ((flags & ~HWLOC_MEMBIND_ALLFLAGS) || hwloc__check_membind_policy(policy) < 0) {
    errno = EINVAL;
    return NULL;
  }

  nodeset = hwloc_fix_membind(topology, nodeset);
  if (!nodeset)
    goto fallback;
  if (flags & HWLOC_MEMBIND_MIGRATE) {
    errno = EINVAL;
    goto fallback;
  }

  if (topology->binding_hooks.alloc_membind)
    return topology->binding_hooks.alloc_membind(topology, len, nodeset, policy, flags);
  else if (topology->binding_hooks.set_area_membind) {
    p = hwloc_alloc(topology, len);
    if (!p)
      return NULL;
    if (topology->binding_hooks.set_area_membind(topology, p, len, nodeset, policy, flags) && flags & HWLOC_MEMBIND_STRICT) {
      int error = errno;
      free(p);
      errno = error;
      return NULL;
    }
    return p;
  } else {
    errno = ENOSYS;
  }

fallback:
  if (flags & HWLOC_MEMBIND_STRICT)
    /* Report error */
    return NULL;
  /* Never mind, allocate anyway */
  return hwloc_alloc(topology, len);
}

void *
hwloc_alloc_membind(hwloc_topology_t topology, size_t len, hwloc_const_bitmap_t set, hwloc_membind_policy_t policy, int flags)
{
  void *ret;

  if (flags & HWLOC_MEMBIND_BYNODESET) {
    ret = hwloc_alloc_membind_by_nodeset(topology, len, set, policy, flags);
  } else {
    hwloc_nodeset_t nodeset = hwloc_bitmap_alloc();
    if (hwloc_fix_membind_cpuset(topology, nodeset, set)) {
      if (flags & HWLOC_MEMBIND_STRICT)
	ret = NULL;
      else
	ret = hwloc_alloc(topology, len);
    } else
      ret = hwloc_alloc_membind_by_nodeset(topology, len, nodeset, policy, flags);
    hwloc_bitmap_free(nodeset);
  }

  return ret;
}

int
hwloc_free(hwloc_topology_t topology, void *addr, size_t len)
{
  if (topology->binding_hooks.free_membind)
    return topology->binding_hooks.free_membind(topology, addr, len);
  return hwloc_free_heap(topology, addr, len);
}

/*
 * Empty binding hooks always returning success
 */

static int dontset_return_complete_cpuset(hwloc_topology_t topology, hwloc_cpuset_t set)
{
  hwloc_bitmap_copy(set, hwloc_topology_get_complete_cpuset(topology));
  return 0;
}

static int dontset_thisthread_cpubind(hwloc_topology_t topology __hwloc_attribute_unused, hwloc_const_bitmap_t set __hwloc_attribute_unused, int flags __hwloc_attribute_unused)
{
  return 0;
}
static int dontget_thisthread_cpubind(hwloc_topology_t topology __hwloc_attribute_unused, hwloc_bitmap_t set, int flags __hwloc_attribute_unused)
{
  return dontset_return_complete_cpuset(topology, set);
}
static int dontset_thisproc_cpubind(hwloc_topology_t topology __hwloc_attribute_unused, hwloc_const_bitmap_t set __hwloc_attribute_unused, int flags __hwloc_attribute_unused)
{
  return 0;
}
static int dontget_thisproc_cpubind(hwloc_topology_t topology __hwloc_attribute_unused, hwloc_bitmap_t set, int flags __hwloc_attribute_unused)
{
  return dontset_return_complete_cpuset(topology, set);
}
static int dontset_proc_cpubind(hwloc_topology_t topology __hwloc_attribute_unused, hwloc_pid_t pid __hwloc_attribute_unused, hwloc_const_bitmap_t set __hwloc_attribute_unused, int flags __hwloc_attribute_unused)
{
  return 0;
}
static int dontget_proc_cpubind(hwloc_topology_t topology __hwloc_attribute_unused, hwloc_pid_t pid __hwloc_attribute_unused, hwloc_bitmap_t cpuset, int flags __hwloc_attribute_unused)
{
  return dontset_return_complete_cpuset(topology, cpuset);
}
#ifdef hwloc_thread_t
static int dontset_thread_cpubind(hwloc_topology_t topology __hwloc_attribute_unused, hwloc_thread_t tid __hwloc_attribute_unused, hwloc_const_bitmap_t set __hwloc_attribute_unused, int flags __hwloc_attribute_unused)
{
  return 0;
}
static int dontget_thread_cpubind(hwloc_topology_t topology __hwloc_attribute_unused, hwloc_thread_t tid __hwloc_attribute_unused, hwloc_bitmap_t cpuset, int flags __hwloc_attribute_unused)
{
  return dontset_return_complete_cpuset(topology, cpuset);
}
#endif

static int dontset_return_complete_nodeset(hwloc_topology_t topology, hwloc_nodeset_t set, hwloc_membind_policy_t *policy)
{
  hwloc_bitmap_copy(set, hwloc_topology_get_complete_nodeset(topology));
  *policy = HWLOC_MEMBIND_MIXED;
  return 0;
}

static int dontset_thisproc_membind(hwloc_topology_t topology __hwloc_attribute_unused, hwloc_const_bitmap_t set __hwloc_attribute_unused, hwloc_membind_policy_t policy __hwloc_attribute_unused, int flags __hwloc_attribute_unused)
{
  return 0;
}
static int dontget_thisproc_membind(hwloc_topology_t topology __hwloc_attribute_unused, hwloc_bitmap_t set, hwloc_membind_policy_t * policy, int flags __hwloc_attribute_unused)
{
  return dontset_return_complete_nodeset(topology, set, policy);
}

static int dontset_thisthread_membind(hwloc_topology_t topology __hwloc_attribute_unused, hwloc_const_bitmap_t set __hwloc_attribute_unused, hwloc_membind_policy_t policy __hwloc_attribute_unused, int flags __hwloc_attribute_unused)
{
  return 0;
}
static int dontget_thisthread_membind(hwloc_topology_t topology __hwloc_attribute_unused, hwloc_bitmap_t set, hwloc_membind_policy_t * policy, int flags __hwloc_attribute_unused)
{
  return dontset_return_complete_nodeset(topology, set, policy);
}

static int dontset_proc_membind(hwloc_topology_t topology __hwloc_attribute_unused, hwloc_pid_t pid __hwloc_attribute_unused, hwloc_const_bitmap_t set __hwloc_attribute_unused, hwloc_membind_policy_t policy __hwloc_attribute_unused, int flags __hwloc_attribute_unused)
{
  return 0;
}
static int dontget_proc_membind(hwloc_topology_t topology __hwloc_attribute_unused, hwloc_pid_t pid __hwloc_attribute_unused, hwloc_bitmap_t set, hwloc_membind_policy_t * policy, int flags __hwloc_attribute_unused)
{
  return dontset_return_complete_nodeset(topology, set, policy);
}

static int dontset_area_membind(hwloc_topology_t topology __hwloc_attribute_unused, const void *addr __hwloc_attribute_unused, size_t size __hwloc_attribute_unused, hwloc_const_bitmap_t set __hwloc_attribute_unused, hwloc_membind_policy_t policy __hwloc_attribute_unused, int flags __hwloc_attribute_unused)
{
  return 0;
}
static int dontget_area_membind(hwloc_topology_t topology __hwloc_attribute_unused, const void *addr __hwloc_attribute_unused, size_t size __hwloc_attribute_unused, hwloc_bitmap_t set, hwloc_membind_policy_t * policy, int flags __hwloc_attribute_unused)
{
  return dontset_return_complete_nodeset(topology, set, policy);
}
static int dontget_area_memlocation(hwloc_topology_t topology __hwloc_attribute_unused, const void *addr __hwloc_attribute_unused, size_t size __hwloc_attribute_unused, hwloc_bitmap_t set, int flags __hwloc_attribute_unused)
{
  hwloc_membind_policy_t policy;
  return dontset_return_complete_nodeset(topology, set, &policy);
}

static void * dontalloc_membind(hwloc_topology_t topology __hwloc_attribute_unused, size_t size __hwloc_attribute_unused, hwloc_const_bitmap_t set __hwloc_attribute_unused, hwloc_membind_policy_t policy __hwloc_attribute_unused, int flags __hwloc_attribute_unused)
{
  return malloc(size);
}
static int dontfree_membind(hwloc_topology_t topology __hwloc_attribute_unused, void *addr __hwloc_attribute_unused, size_t size __hwloc_attribute_unused)
{
  free(addr);
  return 0;
}

static void hwloc_set_dummy_hooks(struct hwloc_binding_hooks *hooks,
				  struct hwloc_topology_support *support __hwloc_attribute_unused)
{
  hooks->set_thisproc_cpubind = dontset_thisproc_cpubind;
  hooks->get_thisproc_cpubind = dontget_thisproc_cpubind;
  hooks->set_thisthread_cpubind = dontset_thisthread_cpubind;
  hooks->get_thisthread_cpubind = dontget_thisthread_cpubind;
  hooks->set_proc_cpubind = dontset_proc_cpubind;
  hooks->get_proc_cpubind = dontget_proc_cpubind;
#ifdef hwloc_thread_t
  hooks->set_thread_cpubind = dontset_thread_cpubind;
  hooks->get_thread_cpubind = dontget_thread_cpubind;
#endif
  hooks->get_thisproc_last_cpu_location = dontget_thisproc_cpubind; /* cpubind instead of last_cpu_location is ok */
  hooks->get_thisthread_last_cpu_location = dontget_thisthread_cpubind; /* cpubind instead of last_cpu_location is ok */
  hooks->get_proc_last_cpu_location = dontget_proc_cpubind; /* cpubind instead of last_cpu_location is ok */
  /* TODO: get_thread_last_cpu_location */
  hooks->set_thisproc_membind = dontset_thisproc_membind;
  hooks->get_thisproc_membind = dontget_thisproc_membind;
  hooks->set_thisthread_membind = dontset_thisthread_membind;
  hooks->get_thisthread_membind = dontget_thisthread_membind;
  hooks->set_proc_membind = dontset_proc_membind;
  hooks->get_proc_membind = dontget_proc_membind;
  hooks->set_area_membind = dontset_area_membind;
  hooks->get_area_membind = dontget_area_membind;
  hooks->get_area_memlocation = dontget_area_memlocation;
  hooks->alloc_membind = dontalloc_membind;
  hooks->free_membind = dontfree_membind;
}

void
hwloc_set_native_binding_hooks(struct hwloc_binding_hooks *hooks, struct hwloc_topology_support *support)
{
#    ifdef HWLOC_LINUX_SYS
    hwloc_set_linuxfs_hooks(hooks, support);
#    endif /* HWLOC_LINUX_SYS */

#    ifdef HWLOC_BGQ_SYS
    hwloc_set_bgq_hooks(hooks, support);
#    endif /* HWLOC_BGQ_SYS */

#    ifdef HWLOC_AIX_SYS
    hwloc_set_aix_hooks(hooks, support);
#    endif /* HWLOC_AIX_SYS */

#    ifdef HWLOC_SOLARIS_SYS
    hwloc_set_solaris_hooks(hooks, support);
#    endif /* HWLOC_SOLARIS_SYS */

#    ifdef HWLOC_WIN_SYS
    hwloc_set_windows_hooks(hooks, support);
#    endif /* HWLOC_WIN_SYS */

#    ifdef HWLOC_DARWIN_SYS
    hwloc_set_darwin_hooks(hooks, support);
#    endif /* HWLOC_DARWIN_SYS */

#    ifdef HWLOC_FREEBSD_SYS
    hwloc_set_freebsd_hooks(hooks, support);
#    endif /* HWLOC_FREEBSD_SYS */

#    ifdef HWLOC_NETBSD_SYS
    hwloc_set_netbsd_hooks(hooks, support);
#    endif /* HWLOC_NETBSD_SYS */

#    ifdef HWLOC_HPUX_SYS
    hwloc_set_hpux_hooks(hooks, support);
#    endif /* HWLOC_HPUX_SYS */
}

/* If the represented system is actually not this system, use dummy binding hooks. */
void
hwloc_set_binding_hooks(struct hwloc_topology *topology)
{
  if (topology->is_thissystem) {
    hwloc_set_native_binding_hooks(&topology->binding_hooks, &topology->support);
    /* every hook not set above will return ENOSYS */
  } else {
    /* not this system, use dummy binding hooks that do nothing (but don't return ENOSYS) */
    hwloc_set_dummy_hooks(&topology->binding_hooks, &topology->support);

    /* Linux has some hooks that also work in this case, but they are not strictly needed yet. */
  }

  /* if not is_thissystem, set_cpubind is fake
   * and get_cpubind returns the whole system cpuset,
   * so don't report that set/get_cpubind as supported
   */
  if (topology->is_thissystem) {
#define DO(which,kind) \
    if (topology->binding_hooks.kind) \
      topology->support.which##bind->kind = 1;
    DO(cpu,set_thisproc_cpubind);
    DO(cpu,get_thisproc_cpubind);
    DO(cpu,set_proc_cpubind);
    DO(cpu,get_proc_cpubind);
    DO(cpu,set_thisthread_cpubind);
    DO(cpu,get_thisthread_cpubind);
#ifdef hwloc_thread_t
    DO(cpu,set_thread_cpubind);
    DO(cpu,get_thread_cpubind);
#endif
    DO(cpu,get_thisproc_last_cpu_location);
    DO(cpu,get_proc_last_cpu_location);
    DO(cpu,get_thisthread_last_cpu_location);
    DO(mem,set_thisproc_membind);
    DO(mem,get_thisproc_membind);
    DO(mem,set_thisthread_membind);
    DO(mem,get_thisthread_membind);
    DO(mem,set_proc_membind);
    DO(mem,get_proc_membind);
    DO(mem,set_area_membind);
    DO(mem,get_area_membind);
    DO(mem,get_area_memlocation);
    DO(mem,alloc_membind);
#undef DO
  }
}
