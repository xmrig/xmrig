/*
 * Copyright © 2009-2010 Oracle and/or its affiliates.  All rights reserved.
 *
 * Copyright © 2017 Inria.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */


#ifdef HWLOC_INSIDE_PLUGIN
/*
 * these declarations are internal only, they are not available to plugins
 * (functions below are internal static symbols).
 */
#error This file should not be used in plugins
#endif


#ifndef HWLOC_PRIVATE_SOLARIS_CHIPTYPE_H
#define HWLOC_PRIVATE_SOLARIS_CHIPTYPE_H

struct hwloc_solaris_chip_info_s {
  char *model;
  char *type;
  /* L1i, L1d, L2, L3 */
#define HWLOC_SOLARIS_CHIP_INFO_L1I 0
#define HWLOC_SOLARIS_CHIP_INFO_L1D 1
#define HWLOC_SOLARIS_CHIP_INFO_L2I 2
#define HWLOC_SOLARIS_CHIP_INFO_L2D 3
#define HWLOC_SOLARIS_CHIP_INFO_L3  4
  long cache_size[5]; /* cleared to -1 if we don't want of that cache */
  unsigned cache_linesize[5];
  unsigned cache_associativity[5];
  int l2_unified;
};

/* fills the structure with 0 on error */
extern void hwloc_solaris_get_chip_info(struct hwloc_solaris_chip_info_s *info);

#endif /* HWLOC_PRIVATE_SOLARIS_CHIPTYPE_H */
