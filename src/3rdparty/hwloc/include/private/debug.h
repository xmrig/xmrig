/*
 * Copyright © 2009 CNRS
 * Copyright © 2009-2017 Inria.  All rights reserved.
 * Copyright © 2009, 2011 Université Bordeaux
 * Copyright © 2011 Cisco Systems, Inc.  All rights reserved.
 * See COPYING in top-level directory.
 */

/* The configuration file */

#ifndef HWLOC_DEBUG_H
#define HWLOC_DEBUG_H

#include "private/autogen/config.h"
#include "private/misc.h"

#ifdef HWLOC_DEBUG
#include <stdarg.h>
#include <stdio.h>
#endif

/* Compile-time assertion */
#define HWLOC_BUILD_ASSERT(condition) ((void)sizeof(char[1 - 2*!(condition)]))

#ifdef HWLOC_DEBUG
static __hwloc_inline int hwloc_debug_enabled(void)
{
  static int checked = 0;
  static int enabled = 1;
  if (!checked) {
    const char *env = getenv("HWLOC_DEBUG_VERBOSE");
    if (env)
      enabled = atoi(env);
    if (enabled)
      fprintf(stderr, "hwloc verbose debug enabled, may be disabled with HWLOC_DEBUG_VERBOSE=0 in the environment.\n");
    checked = 1;
  }
  return enabled;
}
#endif

static __hwloc_inline void hwloc_debug(const char *s __hwloc_attribute_unused, ...) __hwloc_attribute_format(printf, 1, 2);
static __hwloc_inline void hwloc_debug(const char *s __hwloc_attribute_unused, ...)
{
#ifdef HWLOC_DEBUG
  if (hwloc_debug_enabled()) {
    va_list ap;
    va_start(ap, s);
    vfprintf(stderr, s, ap);
    va_end(ap);
  }
#endif
}

#ifdef HWLOC_DEBUG
#define hwloc_debug_bitmap(fmt, bitmap) do { \
if (hwloc_debug_enabled()) { \
  char *s; \
  hwloc_bitmap_asprintf(&s, bitmap); \
  fprintf(stderr, fmt, s); \
  free(s); \
} } while (0)
#define hwloc_debug_1arg_bitmap(fmt, arg1, bitmap) do { \
if (hwloc_debug_enabled()) { \
  char *s; \
  hwloc_bitmap_asprintf(&s, bitmap); \
  fprintf(stderr, fmt, arg1, s); \
  free(s); \
} } while (0)
#define hwloc_debug_2args_bitmap(fmt, arg1, arg2, bitmap) do { \
if (hwloc_debug_enabled()) { \
  char *s; \
  hwloc_bitmap_asprintf(&s, bitmap); \
  fprintf(stderr, fmt, arg1, arg2, s); \
  free(s); \
} } while (0)
#else
#define hwloc_debug_bitmap(s, bitmap) do { } while(0)
#define hwloc_debug_1arg_bitmap(s, arg1, bitmap) do { } while(0)
#define hwloc_debug_2args_bitmap(s, arg1, arg2, bitmap) do { } while(0)
#endif

#endif /* HWLOC_DEBUG_H */
