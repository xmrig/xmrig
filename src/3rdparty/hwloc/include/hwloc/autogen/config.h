/*
 * Copyright © 2009 CNRS
 * Copyright © 2009-2020 Inria.  All rights reserved.
 * Copyright © 2009-2012 Université Bordeaux
 * Copyright © 2009-2011 Cisco Systems, Inc.  All rights reserved.
 * See COPYING in top-level directory.
 */

/* The configuration file */

#ifndef HWLOC_CONFIG_H
#define HWLOC_CONFIG_H

#define HWLOC_VERSION "2.4.1"
#define HWLOC_VERSION_MAJOR 2
#define HWLOC_VERSION_MINOR 4
#define HWLOC_VERSION_RELEASE 1
#define HWLOC_VERSION_GREEK ""

#define __hwloc_restrict
#define __hwloc_inline __inline

#define __hwloc_attribute_unused
#define __hwloc_attribute_malloc
#define __hwloc_attribute_const
#define __hwloc_attribute_pure
#define __hwloc_attribute_deprecated
#define __hwloc_attribute_may_alias
#define __hwloc_attribute_warn_unused_result

/* Defined to 1 if you have the `windows.h' header. */
#define HWLOC_HAVE_WINDOWS_H 1
#define hwloc_pid_t HANDLE
#define hwloc_thread_t HANDLE

#include <windows.h>
#include <BaseTsd.h>
typedef DWORDLONG hwloc_uint64_t;

#if defined( _USRDLL ) /* dynamic linkage */
#if defined( DECLSPEC_EXPORTS )
#define HWLOC_DECLSPEC __declspec(dllexport)
#else
#define HWLOC_DECLSPEC __declspec(dllimport)
#endif
#else /* static linkage */
#define HWLOC_DECLSPEC
#endif

/* Whether we need to re-define all the hwloc public symbols or not */
#define HWLOC_SYM_TRANSFORM 0

/* The hwloc symbol prefix */
#define HWLOC_SYM_PREFIX hwloc_

/* The hwloc symbol prefix in all caps */
#define HWLOC_SYM_PREFIX_CAPS HWLOC_

#endif /* HWLOC_CONFIG_H */
