/*
 * Copyright © 2009 CNRS
 * Copyright © 2009-2020 Inria.  All rights reserved.
 * Copyright © 2009-2011 Université Bordeaux
 * Copyright © 2011 Cisco Systems, Inc.  All rights reserved.
 * See COPYING in top-level directory.
 */

/** \file
 * \brief Macros to help interaction between hwloc and glibc scheduling routines.
 *
 * Applications that use both hwloc and glibc scheduling routines such as
 * sched_getaffinity() or pthread_attr_setaffinity_np() may want to include
 * this file so as to ease conversion between their respective types.
 */

#ifndef HWLOC_GLIBC_SCHED_H
#define HWLOC_GLIBC_SCHED_H

#include "hwloc.h"
#include "hwloc/helper.h"

#include <assert.h>

#if !defined _GNU_SOURCE || (!defined _SCHED_H && !defined _SCHED_H_) || (!defined CPU_SETSIZE && !defined sched_priority)
#error Please make sure to include sched.h before including glibc-sched.h, and define _GNU_SOURCE before any inclusion of sched.h
#endif


#ifdef __cplusplus
extern "C" {
#endif


#ifdef HWLOC_HAVE_CPU_SET


/** \defgroup hwlocality_glibc_sched Interoperability with glibc sched affinity
 *
 * This interface offers ways to convert between hwloc cpusets and glibc cpusets
 * such as those manipulated by sched_getaffinity() or pthread_attr_setaffinity_np().
 *
 * \note Topology \p topology must match the current machine.
 *
 * @{
 */


/** \brief Convert hwloc CPU set \p toposet into glibc sched affinity CPU set \p schedset
 *
 * This function may be used before calling sched_setaffinity or any other function
 * that takes a cpu_set_t as input parameter.
 *
 * \p schedsetsize should be sizeof(cpu_set_t) unless \p schedset was dynamically allocated with CPU_ALLOC
 */
static __hwloc_inline int
hwloc_cpuset_to_glibc_sched_affinity(hwloc_topology_t topology __hwloc_attribute_unused, hwloc_const_cpuset_t hwlocset,
				    cpu_set_t *schedset, size_t schedsetsize)
{
#ifdef CPU_ZERO_S
  unsigned cpu;
  CPU_ZERO_S(schedsetsize, schedset);
  hwloc_bitmap_foreach_begin(cpu, hwlocset)
    CPU_SET_S(cpu, schedsetsize, schedset);
  hwloc_bitmap_foreach_end();
#else /* !CPU_ZERO_S */
  unsigned cpu;
  CPU_ZERO(schedset);
  assert(schedsetsize == sizeof(cpu_set_t));
  hwloc_bitmap_foreach_begin(cpu, hwlocset)
    CPU_SET(cpu, schedset);
  hwloc_bitmap_foreach_end();
#endif /* !CPU_ZERO_S */
  return 0;
}

/** \brief Convert glibc sched affinity CPU set \p schedset into hwloc CPU set
 *
 * This function may be used before calling sched_setaffinity  or any other function
 * that takes a cpu_set_t  as input parameter.
 *
 * \p schedsetsize should be sizeof(cpu_set_t) unless \p schedset was dynamically allocated with CPU_ALLOC
 */
static __hwloc_inline int
hwloc_cpuset_from_glibc_sched_affinity(hwloc_topology_t topology __hwloc_attribute_unused, hwloc_cpuset_t hwlocset,
                                       const cpu_set_t *schedset, size_t schedsetsize)
{
  int cpu;
#ifdef CPU_ZERO_S
  int count;
#endif
  hwloc_bitmap_zero(hwlocset);
#ifdef CPU_ZERO_S
  count = CPU_COUNT_S(schedsetsize, schedset);
  cpu = 0;
  while (count) {
    if (CPU_ISSET_S(cpu, schedsetsize, schedset)) {
      hwloc_bitmap_set(hwlocset, cpu);
      count--;
    }
    cpu++;
  }
#else /* !CPU_ZERO_S */
  /* sched.h does not support dynamic cpu_set_t (introduced in glibc 2.7),
   * assume we have a very old interface without CPU_COUNT (added in 2.6)
   */
  assert(schedsetsize == sizeof(cpu_set_t));
  for(cpu=0; cpu<CPU_SETSIZE; cpu++)
    if (CPU_ISSET(cpu, schedset))
      hwloc_bitmap_set(hwlocset, cpu);
#endif /* !CPU_ZERO_S */
  return 0;
}

/** @} */


#endif /* CPU_SET */


#ifdef __cplusplus
} /* extern "C" */
#endif


#endif /* HWLOC_GLIBC_SCHED_H */
