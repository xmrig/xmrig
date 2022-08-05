/*
 * Copyright © 2009 Université Bordeaux
 * Copyright © 2020 Inria.  All rights reserved.
 *
 * See COPYING in top-level directory.
 */

#ifndef HWLOC_PRIVATE_WINDOWS_H
#define HWLOC_PRIVATE_WINDOWS_H

#ifdef __GNUC__
#define _ANONYMOUS_UNION __extension__
#define _ANONYMOUS_STRUCT __extension__
#else
#define _ANONYMOUS_UNION
#define _ANONYMOUS_STRUCT
#endif /* __GNUC__ */
#define DUMMYUNIONNAME
#define DUMMYSTRUCTNAME

#endif /* HWLOC_PRIVATE_WINDOWS_H */
