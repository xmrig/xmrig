/*
 * Copyright © 2009 CNRS
 * Copyright © 2009-2023 Inria.  All rights reserved.
 * Copyright © 2009-2012, 2020 Université Bordeaux
 * Copyright © 2009-2011 Cisco Systems, Inc.  All rights reserved.
 * Copyright © 2022 IBM Corporation.  All rights reserved.
 * See COPYING in top-level directory.
 */

#include "private/autogen/config.h"

#define _ATFILE_SOURCE
#include <assert.h>
#include <sys/types.h>
#ifdef HAVE_DIRENT_H
#include <dirent.h>
#endif
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif
#include <string.h>
#include <errno.h>
#include <stdio.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <limits.h>
#include <float.h>

#include "hwloc.h"
#include "private/private.h"
#include "private/debug.h"
#include "private/misc.h"

#ifdef HAVE_MACH_MACH_INIT_H
#include <mach/mach_init.h>
#endif
#ifdef HAVE_MACH_INIT_H
#include <mach_init.h>
#endif
#ifdef HAVE_MACH_MACH_HOST_H
#include <mach/mach_host.h>
#endif

#ifdef HAVE_SYS_PARAM_H
#include <sys/param.h>
#endif

#ifdef HAVE_SYS_SYSCTL_H
#include <sys/sysctl.h>
#endif

#ifdef HWLOC_WIN_SYS
#include <windows.h>
#endif


#ifdef HWLOC_HAVE_LEVELZERO
/*
 * Define ZES_ENABLE_SYSMAN=1 early so that the LevelZero backend gets Sysman enabled.
 *
 * Only if the levelzero was enabled in this build so that we don't enable sysman
 * for external levelzero users when hwloc doesn't need it. If somebody ever loads
 * an external levelzero plugin in a hwloc library built without levelzero (unlikely),
 * he may have to manually set ZES_ENABLE_SYSMAN=1.
 *
 * Use the constructor if supported and/or the Windows DllMain callback.
 * Do it in the main hwloc library instead of the levelzero component because
 * the latter could be loaded later as a plugin.
 *
 * L0 seems to be using getenv() to check this variable on Windows
 * (at least in the Intel Compute-Runtime of March 2021),
 * but setenv() doesn't seem to exist on Windows, hence use putenv() to set the variable.
 *
 * For the record, Get/SetEnvironmentVariable() is not exactly the same as getenv/putenv():
 * - getenv() doesn't see what was set with SetEnvironmentVariable()
 * - GetEnvironmentVariable() doesn't see putenv() in cygwin (while it does in MSVC and MinGW).
 * Hence, if L0 ever switches from getenv() to GetEnvironmentVariable(),
 * it will break in cygwin, we'll have to use both putenv() and SetEnvironmentVariable().
 * Hopefully L0 will provide a way to enable Sysman without env vars before it happens.
 */
#if HWLOC_HAVE_ATTRIBUTE_CONSTRUCTOR
static void hwloc_constructor(void) __attribute__((constructor));
static void hwloc_constructor(void)
{
  if (!getenv("ZES_ENABLE_SYSMAN"))
#ifdef HWLOC_WIN_SYS
    putenv("ZES_ENABLE_SYSMAN=1");
#else
    setenv("ZES_ENABLE_SYSMAN", "1", 1);
#endif
}
#endif
#ifdef HWLOC_WIN_SYS
BOOL WINAPI DllMain(HINSTANCE hinstDLL, DWORD fdwReason, LPVOID lpReserved)
{
  if (fdwReason == DLL_PROCESS_ATTACH) {
    if (!getenv("ZES_ENABLE_SYSMAN"))
      /* Windows does not have a setenv, so use putenv. */
      putenv((char *) "ZES_ENABLE_SYSMAN=1");
  }
  return TRUE;
}
#endif
#endif /* HWLOC_HAVE_LEVELZERO */


unsigned hwloc_get_api_version(void)
{
  return HWLOC_API_VERSION;
}

int hwloc_topology_abi_check(hwloc_topology_t topology)
{
  return topology->topology_abi != HWLOC_TOPOLOGY_ABI ? -1 : 0;
}

/* callers should rather use wrappers HWLOC_SHOW_ALL_ERRORS() and HWLOC_SHOW_CRITICAL_ERRORS() for clarity */
int hwloc_hide_errors(void)
{
  static int hide = 1; /* only show critical errors by default. lstopo will show others */
  static int checked = 0;
  if (!checked) {
    const char *envvar = getenv("HWLOC_HIDE_ERRORS");
    if (envvar) {
      hide = atoi(envvar);
#ifdef HWLOC_DEBUG
    } else {
      /* if debug is enabled and HWLOC_DEBUG_VERBOSE isn't forced to 0,
       * show all errors jus like we show all debug messages.
       */
      envvar = getenv("HWLOC_DEBUG_VERBOSE");
      if (!envvar || atoi(envvar))
        hide = 0;
#endif
    }
    checked = 1;
  }
  return hide;
}


/* format the obj info to print in error messages */
static void
report_insert_error_format_obj(char *buf, size_t buflen, hwloc_obj_t obj)
{
  char typestr[64];
  char *cpusetstr;
  char *nodesetstr = NULL;
  char indexstr[64] = "";
  char groupstr[64] = "";

  hwloc_obj_type_snprintf(typestr, sizeof(typestr), obj, 0);
  hwloc_bitmap_asprintf(&cpusetstr, obj->cpuset);
  if (obj->os_index != HWLOC_UNKNOWN_INDEX)
    snprintf(indexstr, sizeof(indexstr), "P#%u ", obj->os_index);
  if (obj->type == HWLOC_OBJ_GROUP)
    snprintf(groupstr, sizeof(groupstr), "groupkind %u-%u ", obj->attr->group.kind, obj->attr->group.subkind);
  if (obj->nodeset) /* may be missing during insert */
    hwloc_bitmap_asprintf(&nodesetstr, obj->nodeset);
  snprintf(buf, buflen, "%s (%s%s%s%s%scpuset %s%s%s)",
           typestr,
           indexstr,
           obj->subtype ? "subtype " : "", obj->subtype ? obj->subtype : "", obj->subtype ? " " : "",
           groupstr,
           cpusetstr,
           nodesetstr ? " nodeset " : "", nodesetstr ? nodesetstr : "");
  free(cpusetstr);
  free(nodesetstr);
}

static void report_insert_error(hwloc_obj_t new, hwloc_obj_t old, const char *msg, const char *reason)
{
  static int reported = 0;

  if (reason && !reported && HWLOC_SHOW_CRITICAL_ERRORS()) {
    char newstr[512];
    char oldstr[512];
    report_insert_error_format_obj(newstr, sizeof(newstr), new);
    report_insert_error_format_obj(oldstr, sizeof(oldstr), old);

    fprintf(stderr, "****************************************************************************\n");
    fprintf(stderr, "* hwloc %s received invalid information from the operating system.\n", HWLOC_VERSION);
    fprintf(stderr, "*\n");
    fprintf(stderr, "* Failed with error: %s\n", msg);
    fprintf(stderr, "* while inserting %s\n", newstr);
    fprintf(stderr, "* at %s\n", oldstr);
    fprintf(stderr, "* coming from: %s\n", reason);
    fprintf(stderr, "*\n");
    fprintf(stderr, "* The following FAQ entry in the hwloc documentation may help:\n");
    fprintf(stderr, "*   What should I do when hwloc reports \"operating system\" warnings?\n");
    fprintf(stderr, "* Otherwise please report this error message to the hwloc user's mailing list,\n");
#ifdef HWLOC_LINUX_SYS
    fprintf(stderr, "* along with the files generated by the hwloc-gather-topology script.\n");
#else
    fprintf(stderr, "* along with any relevant topology information from your platform.\n");
#endif
    fprintf(stderr, "* \n");
    fprintf(stderr, "* hwloc will now ignore this invalid topology information and continue.\n");
    fprintf(stderr, "****************************************************************************\n");
    reported = 1;
  }
}

#if defined(HAVE_SYSCTLBYNAME)
int hwloc_get_sysctlbyname(const char *name, int64_t *ret)
{
  union {
    int32_t i32;
    int64_t i64;
  } n;
  size_t size = sizeof(n);
  if (sysctlbyname(name, &n, &size, NULL, 0))
    return -1;
  switch (size) {
    case sizeof(n.i32):
      *ret = n.i32;
      break;
    case sizeof(n.i64):
      *ret = n.i64;
      break;
    default:
      return -1;
  }
  return 0;
}
#endif

#if defined(HAVE_SYSCTL)
int hwloc_get_sysctl(int name[], unsigned namelen, int64_t *ret)
{
  union {
    int32_t i32;
    int64_t i64;
  } n;
  size_t size = sizeof(n);
  if (sysctl(name, namelen, &n, &size, NULL, 0))
    return -1;
  switch (size) {
    case sizeof(n.i32):
      *ret = n.i32;
      break;
    case sizeof(n.i64):
      *ret = n.i64;
      break;
    default:
      return -1;
  }
  return 0;
}
#endif

/* Return the OS-provided number of processors.
 * Assumes topology->is_thissystem is true.
 */
#ifndef HWLOC_WIN_SYS /* The windows implementation is in topology-windows.c */
int
hwloc_fallback_nbprocessors(unsigned flags) {
  int n;

  if (flags & HWLOC_FALLBACK_NBPROCESSORS_INCLUDE_OFFLINE) {
    /* try to get all CPUs for Linux and Solaris that can handle offline CPUs */
#if HAVE_DECL__SC_NPROCESSORS_CONF
    n = sysconf(_SC_NPROCESSORS_CONF);
#elif HAVE_DECL__SC_NPROC_CONF
    n = sysconf(_SC_NPROC_CONF);
#else
    n = -1;
#endif
    if (n != -1)
      return n;
  }

  /* try getting only online CPUs, or whatever we can get */
#if HAVE_DECL__SC_NPROCESSORS_ONLN
  n = sysconf(_SC_NPROCESSORS_ONLN);
#elif HAVE_DECL__SC_NPROC_ONLN
  n = sysconf(_SC_NPROC_ONLN);
#elif HAVE_DECL__SC_NPROCESSORS_CONF
  n = sysconf(_SC_NPROCESSORS_CONF);
#elif HAVE_DECL__SC_NPROC_CONF
  n = sysconf(_SC_NPROC_CONF);
#elif defined(HAVE_HOST_INFO) && HAVE_HOST_INFO
  struct host_basic_info info;
  mach_msg_type_number_t count = HOST_BASIC_INFO_COUNT;
  host_info(mach_host_self(), HOST_BASIC_INFO, (integer_t*) &info, &count);
  n = info.avail_cpus;
#elif defined(HAVE_SYSCTLBYNAME)
  int64_t nn;
  if (hwloc_get_sysctlbyname("hw.ncpu", &nn))
    nn = -1;
  n = nn;
#elif defined(HAVE_SYSCTL) && HAVE_DECL_CTL_HW && HAVE_DECL_HW_NCPU
  static int name[2] = {CTL_HW, HW_NCPU};
  int64_t nn;
  if (hwloc_get_sysctl(name, sizeof(name)/sizeof(*name), &nn))
    n = -1;
  n = nn;
#else
#ifdef __GNUC__
#warning No known way to discover number of available processors on this system
#endif
  n = -1;
#endif
  return n;
}

int64_t
hwloc_fallback_memsize(void) {
  int64_t size;
#if defined(HAVE_HOST_INFO) && HAVE_HOST_INFO
  struct host_basic_info info;
  mach_msg_type_number_t count = HOST_BASIC_INFO_COUNT;
  host_info(mach_host_self(), HOST_BASIC_INFO, (integer_t*) &info, &count);
  size = info.memory_size;
#elif defined(HAVE_SYSCTL) && HAVE_DECL_CTL_HW && (HAVE_DECL_HW_REALMEM64 || HAVE_DECL_HW_MEMSIZE64 || HAVE_DECL_HW_PHYSMEM64 || HAVE_DECL_HW_USERMEM64 || HAVE_DECL_HW_REALMEM || HAVE_DECL_HW_MEMSIZE || HAVE_DECL_HW_PHYSMEM || HAVE_DECL_HW_USERMEM)
#if HAVE_DECL_HW_MEMSIZE64
  static int name[2] = {CTL_HW, HW_MEMSIZE64};
#elif HAVE_DECL_HW_REALMEM64
  static int name[2] = {CTL_HW, HW_REALMEM64};
#elif HAVE_DECL_HW_PHYSMEM64
  static int name[2] = {CTL_HW, HW_PHYSMEM64};
#elif HAVE_DECL_HW_USERMEM64
  static int name[2] = {CTL_HW, HW_USERMEM64};
#elif HAVE_DECL_HW_MEMSIZE
  static int name[2] = {CTL_HW, HW_MEMSIZE};
#elif HAVE_DECL_HW_REALMEM
  static int name[2] = {CTL_HW, HW_REALMEM};
#elif HAVE_DECL_HW_PHYSMEM
  static int name[2] = {CTL_HW, HW_PHYSMEM};
#elif HAVE_DECL_HW_USERMEM
  static int name[2] = {CTL_HW, HW_USERMEM};
#endif
  if (hwloc_get_sysctl(name, sizeof(name)/sizeof(*name), &size))
    size = -1;
#elif defined(HAVE_SYSCTLBYNAME)
  if (hwloc_get_sysctlbyname("hw.memsize", &size) &&
      hwloc_get_sysctlbyname("hw.realmem", &size) &&
      hwloc_get_sysctlbyname("hw.physmem", &size) &&
      hwloc_get_sysctlbyname("hw.usermem", &size))
      size = -1;
#else
  size = -1;
#endif
  return size;
}
#endif /* !HWLOC_WIN_SYS */

/*
 * Use the given number of processors to set a PU level.
 */
void
hwloc_setup_pu_level(struct hwloc_topology *topology,
		     unsigned nb_pus)
{
  struct hwloc_obj *obj;
  unsigned oscpu,cpu;

  hwloc_debug("%s", "\n\n * CPU cpusets *\n\n");
  for (cpu=0,oscpu=0; cpu<nb_pus; oscpu++)
    {
      obj = hwloc_alloc_setup_object(topology, HWLOC_OBJ_PU, oscpu);
      obj->cpuset = hwloc_bitmap_alloc();
      hwloc_bitmap_only(obj->cpuset, oscpu);

      hwloc_debug_2args_bitmap("cpu %u (os %u) has cpuset %s\n",
		 cpu, oscpu, obj->cpuset);
      hwloc__insert_object_by_cpuset(topology, NULL, obj, "core:pulevel");

      cpu++;
    }
}

/* Traverse children of a parent in a safe way: reread the next pointer as
 * appropriate to prevent crash on child deletion:  */
#define for_each_child_safe(child, parent, pchild) \
  for (pchild = &(parent)->first_child, child = *pchild; \
       child; \
       /* Check whether the current child was not dropped.  */ \
       (*pchild == child ? pchild = &(child->next_sibling) : NULL), \
       /* Get pointer to next child.  */ \
        child = *pchild)
#define for_each_memory_child_safe(child, parent, pchild) \
  for (pchild = &(parent)->memory_first_child, child = *pchild; \
       child; \
       /* Check whether the current child was not dropped.  */ \
       (*pchild == child ? pchild = &(child->next_sibling) : NULL), \
       /* Get pointer to next child.  */ \
        child = *pchild)
#define for_each_io_child_safe(child, parent, pchild) \
  for (pchild = &(parent)->io_first_child, child = *pchild; \
       child; \
       /* Check whether the current child was not dropped.  */ \
       (*pchild == child ? pchild = &(child->next_sibling) : NULL), \
       /* Get pointer to next child.  */ \
        child = *pchild)
#define for_each_misc_child_safe(child, parent, pchild) \
  for (pchild = &(parent)->misc_first_child, child = *pchild; \
       child; \
       /* Check whether the current child was not dropped.  */ \
       (*pchild == child ? pchild = &(child->next_sibling) : NULL), \
       /* Get pointer to next child.  */ \
        child = *pchild)

#ifdef HWLOC_DEBUG
/* Just for debugging.  */
static void
hwloc_debug_print_object(int indent __hwloc_attribute_unused, hwloc_obj_t obj)
{
  char type[64], idx[12], attr[1024], *cpuset = NULL;
  hwloc_debug("%*s", 2*indent, "");
  hwloc_obj_type_snprintf(type, sizeof(type), obj, 1);
  if (obj->os_index != HWLOC_UNKNOWN_INDEX)
    snprintf(idx, sizeof(idx), "#%u", obj->os_index);
  else
    *idx = '\0';
  hwloc_obj_attr_snprintf(attr, sizeof(attr), obj, " ", 1);
  hwloc_debug("%s%s%s%s%s", type, idx, *attr ? "(" : "", attr, *attr ? ")" : "");
  if (obj->name)
    hwloc_debug(" name \"%s\"", obj->name);
  if (obj->subtype)
    hwloc_debug(" subtype \"%s\"", obj->subtype);
  if (obj->cpuset) {
    hwloc_bitmap_asprintf(&cpuset, obj->cpuset);
    hwloc_debug(" cpuset %s", cpuset);
    free(cpuset);
  }
  if (obj->complete_cpuset) {
    hwloc_bitmap_asprintf(&cpuset, obj->complete_cpuset);
    hwloc_debug(" complete %s", cpuset);
    free(cpuset);
  }
  if (obj->nodeset) {
    hwloc_bitmap_asprintf(&cpuset, obj->nodeset);
    hwloc_debug(" nodeset %s", cpuset);
    free(cpuset);
  }
  if (obj->complete_nodeset) {
    hwloc_bitmap_asprintf(&cpuset, obj->complete_nodeset);
    hwloc_debug(" completeN %s", cpuset);
    free(cpuset);
  }
  if (obj->arity)
    hwloc_debug(" arity %u", obj->arity);
  hwloc_debug("%s", "\n");
}

static void
hwloc_debug_print_objects(int indent __hwloc_attribute_unused, hwloc_obj_t obj)
{
  if (hwloc_debug_enabled() >= 2) {
    hwloc_obj_t child;
    hwloc_debug_print_object(indent, obj);
    for_each_child (child, obj)
      hwloc_debug_print_objects(indent + 1, child);
    for_each_memory_child (child, obj)
      hwloc_debug_print_objects(indent + 1, child);
    for_each_io_child (child, obj)
      hwloc_debug_print_objects(indent + 1, child);
    for_each_misc_child (child, obj)
      hwloc_debug_print_objects(indent + 1, child);
  }
}
#else /* !HWLOC_DEBUG */
#define hwloc_debug_print_object(indent, obj) do { /* nothing */ } while (0)
#define hwloc_debug_print_objects(indent, obj) do { /* nothing */ } while (0)
#endif /* !HWLOC_DEBUG */

int hwloc_obj_set_subtype(hwloc_topology_t topology __hwloc_attribute_unused, hwloc_obj_t obj, const char *subtype)
{
  char *new = NULL;
  if (subtype) {
    new = strdup(subtype);
    if (!new)
      return -1;
  }
  if (obj->subtype)
    free(obj->subtype);
  obj->subtype = new;
  return 0;
}

void hwloc__free_infos(struct hwloc_info_s *infos, unsigned count)
{
  unsigned i;
  for(i=0; i<count; i++) {
    free(infos[i].name);
    free(infos[i].value);
  }
  free(infos);
}

int hwloc__add_info(struct hwloc_info_s **infosp, unsigned *countp, const char *name, const char *value)
{
  unsigned count = *countp;
  struct hwloc_info_s *infos = *infosp;
#define OBJECT_INFO_ALLOC 8
  /* nothing allocated initially, (re-)allocate by multiple of 8 */
  unsigned alloccount = (count + 1 + (OBJECT_INFO_ALLOC-1)) & ~(OBJECT_INFO_ALLOC-1);
  if (count != alloccount) {
    struct hwloc_info_s *tmpinfos = realloc(infos, alloccount*sizeof(*infos));
    if (!tmpinfos)
      /* failed to allocate, ignore this info */
      goto out_with_array;
    *infosp = infos = tmpinfos;
  }
  infos[count].name = strdup(name);
  if (!infos[count].name)
    goto out_with_array;
  infos[count].value = strdup(value);
  if (!infos[count].value)
    goto out_with_name;
  *countp = count+1;
  return 0;

 out_with_name:
  free(infos[count].name);
 out_with_array:
  /* don't bother reducing the array */
  return -1;
}

int hwloc__add_info_nodup(struct hwloc_info_s **infosp, unsigned *countp,
			  const char *name, const char *value,
			  int replace)
{
  struct hwloc_info_s *infos = *infosp;
  unsigned count = *countp;
  unsigned i;
  for(i=0; i<count; i++) {
    if (!strcmp(infos[i].name, name)) {
      if (replace) {
	char *new = strdup(value);
	if (!new)
	  return -1;
	free(infos[i].value);
	infos[i].value = new;
      }
      return 0;
    }
  }
  return hwloc__add_info(infosp, countp, name, value);
}

int hwloc__move_infos(struct hwloc_info_s **dst_infosp, unsigned *dst_countp,
		      struct hwloc_info_s **src_infosp, unsigned *src_countp)
{
  unsigned dst_count = *dst_countp;
  struct hwloc_info_s *dst_infos = *dst_infosp;
  unsigned src_count = *src_countp;
  struct hwloc_info_s *src_infos = *src_infosp;
  unsigned i;
#define OBJECT_INFO_ALLOC 8
  /* nothing allocated initially, (re-)allocate by multiple of 8 */
  unsigned alloccount = (dst_count + src_count + (OBJECT_INFO_ALLOC-1)) & ~(OBJECT_INFO_ALLOC-1);
  if (dst_count != alloccount) {
    struct hwloc_info_s *tmp_infos = realloc(dst_infos, alloccount*sizeof(*dst_infos));
    if (!tmp_infos)
      /* Failed to realloc, ignore the appended infos */
      goto drop;
    dst_infos = tmp_infos;
  }
  for(i=0; i<src_count; i++, dst_count++) {
    dst_infos[dst_count].name = src_infos[i].name;
    dst_infos[dst_count].value = src_infos[i].value;
  }
  *dst_infosp = dst_infos;
  *dst_countp = dst_count;
  free(src_infos);
  *src_infosp = NULL;
  *src_countp = 0;
  return 0;

 drop:
  /* drop src infos, don't modify dst_infos at all */
  for(i=0; i<src_count; i++) {
    free(src_infos[i].name);
    free(src_infos[i].value);
  }
  free(src_infos);
  *src_infosp = NULL;
  *src_countp = 0;
  return -1;
}

int hwloc_obj_add_info(hwloc_obj_t obj, const char *name, const char *value)
{
  return hwloc__add_info(&obj->infos, &obj->infos_count, name, value);
}

/* This function may be called with topology->tma set, it cannot free() or realloc() */
int hwloc__tma_dup_infos(struct hwloc_tma *tma,
                         struct hwloc_info_s **newip, unsigned *newcp,
                         struct hwloc_info_s *oldi, unsigned oldc)
{
  struct hwloc_info_s *newi;
  unsigned i, j;
  newi = hwloc_tma_calloc(tma, oldc * sizeof(*newi));
  if (!newi)
    return -1;
  for(i=0; i<oldc; i++) {
    newi[i].name = hwloc_tma_strdup(tma, oldi[i].name);
    newi[i].value = hwloc_tma_strdup(tma, oldi[i].value);
    if (!newi[i].name || !newi[i].value)
      goto failed;
  }
  *newip = newi;
  *newcp = oldc;
  return 0;

 failed:
  assert(!tma || !tma->dontfree); /* this tma cannot fail to allocate */
  for(j=0; j<=i; j++) {
    free(newi[i].name);
    free(newi[i].value);
  }
  free(newi);
  *newip = NULL;
  return -1;
}

static void
hwloc__free_object_contents(hwloc_obj_t obj)
{
  switch (obj->type) {
  case HWLOC_OBJ_NUMANODE:
    free(obj->attr->numanode.page_types);
    break;
  default:
    break;
  }
  hwloc__free_infos(obj->infos, obj->infos_count);
  free(obj->attr);
  free(obj->children);
  free(obj->subtype);
  free(obj->name);
  hwloc_bitmap_free(obj->cpuset);
  hwloc_bitmap_free(obj->complete_cpuset);
  hwloc_bitmap_free(obj->nodeset);
  hwloc_bitmap_free(obj->complete_nodeset);
}

/* Free an object and all its content.  */
void
hwloc_free_unlinked_object(hwloc_obj_t obj)
{
  hwloc__free_object_contents(obj);
  free(obj);
}

/* Replace old with contents of new object, and make new freeable by the caller.
 * Requires reconnect (for siblings pointers and group depth),
 * fixup of sets (only the main cpuset was likely compared before merging),
 * and update of total_memory and group depth.
 */
static void
hwloc_replace_linked_object(hwloc_obj_t old, hwloc_obj_t new)
{
  /* drop old fields */
  hwloc__free_object_contents(old);
  /* copy old tree pointers to new */
  new->parent = old->parent;
  new->next_sibling = old->next_sibling;
  new->first_child = old->first_child;
  new->memory_first_child = old->memory_first_child;
  new->io_first_child = old->io_first_child;
  new->misc_first_child = old->misc_first_child;
  /* copy new contents to old now that tree pointers are OK */
  memcpy(old, new, sizeof(*old));
  /* clear new to that we may free it */
  memset(new, 0,sizeof(*new));
}

/* Remove an object and its children from its parent and free them.
 * Only updates next_sibling/first_child pointers,
 * so may only be used during early discovery or during destroy.
 */
static void
unlink_and_free_object_and_children(hwloc_obj_t *pobj)
{
  hwloc_obj_t obj = *pobj, child, *pchild;

  for_each_child_safe(child, obj, pchild)
    unlink_and_free_object_and_children(pchild);
  for_each_memory_child_safe(child, obj, pchild)
    unlink_and_free_object_and_children(pchild);
  for_each_io_child_safe(child, obj, pchild)
    unlink_and_free_object_and_children(pchild);
  for_each_misc_child_safe(child, obj, pchild)
    unlink_and_free_object_and_children(pchild);

  *pobj = obj->next_sibling;
  hwloc_free_unlinked_object(obj);
}

/* Free an object and its children without unlinking from parent.
 */
void
hwloc_free_object_and_children(hwloc_obj_t obj)
{
  if (obj)
    unlink_and_free_object_and_children(&obj);
}

/* Free an object, its next siblings and their children without unlinking from parent.
 */
void
hwloc_free_object_siblings_and_children(hwloc_obj_t obj)
{
  while (obj)
    unlink_and_free_object_and_children(&obj);
}

/* insert the (non-empty) list of sibling starting at firstnew as new children of newparent,
 * and return the address of the pointer to the next one
 */
static hwloc_obj_t *
insert_siblings_list(hwloc_obj_t *firstp, hwloc_obj_t firstnew, hwloc_obj_t newparent)
{
  hwloc_obj_t tmp;
  assert(firstnew);
  *firstp = tmp = firstnew;
  tmp->parent = newparent;
  while (tmp->next_sibling) {
    tmp = tmp->next_sibling;
    tmp->parent = newparent;
  }
  return &tmp->next_sibling;
}

/* Take the new list starting at firstnew and prepend it to the old list starting at *firstp,
 * and mark the new children as children of newparent.
 * May be used during early or late discovery (updates prev_sibling and sibling_rank).
 * List firstnew must be non-NULL.
 */
static void
prepend_siblings_list(hwloc_obj_t *firstp, hwloc_obj_t firstnew, hwloc_obj_t newparent)
{
  hwloc_obj_t *tmpp, tmp, last;
  unsigned length;

  /* update parent pointers and find the length and end of the new list */
  for(length = 0, tmpp = &firstnew, last = NULL ; *tmpp; length++, last = *tmpp, tmpp = &((*tmpp)->next_sibling))
    (*tmpp)->parent = newparent;

  /* update sibling_rank */
  for(tmp = *firstp; tmp; tmp = tmp->next_sibling)
    tmp->sibling_rank += length; /* if it wasn't initialized yet, it'll be overwritten later */

  /* place the existing list at the end of the new one */
  *tmpp = *firstp;
  if (*firstp)
    (*firstp)->prev_sibling = last;

  /* use the beginning of the new list now */
  *firstp = firstnew;
}

/* Take the new list starting at firstnew and append it to the old list starting at *firstp,
 * and mark the new children as children of newparent.
 * May be used during early or late discovery (updates prev_sibling and sibling_rank).
 */
static void
append_siblings_list(hwloc_obj_t *firstp, hwloc_obj_t firstnew, hwloc_obj_t newparent)
{
  hwloc_obj_t *tmpp, tmp, last;
  unsigned length;

  /* find the length and end of the existing list */
  for(length = 0, tmpp = firstp, last = NULL ; *tmpp; length++, last = *tmpp, tmpp = &((*tmpp)->next_sibling));

  /* update parent pointers and sibling_rank */
  for(tmp = firstnew; tmp; tmp = tmp->next_sibling) {
    tmp->parent = newparent;
    tmp->sibling_rank += length; /* if it wasn't set yet, it'll be overwritten later */
  }

  /* place new list at the end of the old one */
  *tmpp = firstnew;
  if (firstnew)
    firstnew->prev_sibling = last;
}

/* Remove an object from its parent and free it.
 * Only updates next_sibling/first_child pointers,
 * so may only be used during early discovery.
 *
 * Children are inserted in the parent.
 * If children should be inserted somewhere else (e.g. when merging with a child),
 * the caller should move them before calling this function.
 */
static void
unlink_and_free_single_object(hwloc_obj_t *pparent)
{
  hwloc_obj_t old = *pparent;
  hwloc_obj_t *lastp;

  if (old->type == HWLOC_OBJ_MISC) {
    /* Misc object */

    /* no normal children */
    assert(!old->first_child);
    /* no memory children */
    assert(!old->memory_first_child);
    /* no I/O children */
    assert(!old->io_first_child);

    if (old->misc_first_child)
      /* insert old misc object children as new siblings below parent instead of old */
      lastp = insert_siblings_list(pparent, old->misc_first_child, old->parent);
    else
      lastp = pparent;
    /* append old siblings back */
    *lastp = old->next_sibling;

  } else if (hwloc__obj_type_is_io(old->type)) {
    /* I/O object */

    /* no normal children */
    assert(!old->first_child);
    /* no memory children */
    assert(!old->memory_first_child);

    if (old->io_first_child)
      /* insert old I/O object children as new siblings below parent instead of old */
      lastp = insert_siblings_list(pparent, old->io_first_child, old->parent);
    else
      lastp = pparent;
    /* append old siblings back */
    *lastp = old->next_sibling;

    /* append old Misc children to parent */
    if (old->misc_first_child)
      append_siblings_list(&old->parent->misc_first_child, old->misc_first_child, old->parent);

  } else if (hwloc__obj_type_is_memory(old->type)) {
    /* memory object */

    /* no normal children */
    assert(!old->first_child);
    /* no I/O children */
    assert(!old->io_first_child);

    if (old->memory_first_child)
      /* insert old memory object children as new siblings below parent instead of old */
      lastp = insert_siblings_list(pparent, old->memory_first_child, old->parent);
    else
      lastp = pparent;
    /* append old siblings back */
    *lastp = old->next_sibling;

    /* append old Misc children to parent */
    if (old->misc_first_child)
      append_siblings_list(&old->parent->misc_first_child, old->misc_first_child, old->parent);

  } else {
    /* Normal object */

    if (old->first_child)
      /* insert old object children as new siblings below parent instead of old */
      lastp = insert_siblings_list(pparent, old->first_child, old->parent);
    else
      lastp = pparent;
    /* append old siblings back */
    *lastp = old->next_sibling;

    /* append old memory, I/O and Misc children to parent
     * old->parent cannot be NULL (removing root), misc children should have been moved by the caller earlier.
     */
    if (old->memory_first_child)
      append_siblings_list(&old->parent->memory_first_child, old->memory_first_child, old->parent);
    if (old->io_first_child)
      append_siblings_list(&old->parent->io_first_child, old->io_first_child, old->parent);
    if (old->misc_first_child)
      append_siblings_list(&old->parent->misc_first_child, old->misc_first_child, old->parent);
  }

  hwloc_free_unlinked_object(old);
}

/* This function may use a tma, it cannot free() or realloc() */
static int
hwloc__duplicate_object(struct hwloc_topology *newtopology,
			struct hwloc_obj *newparent,
			struct hwloc_obj *newobj,
			struct hwloc_obj *src)
{
  struct hwloc_tma *tma = newtopology->tma;
  hwloc_obj_t *level;
  unsigned level_width;
  size_t len;
  unsigned i;
  hwloc_obj_t child, prev;
  int err = 0;

  /* either we're duplicating to an already allocated new root, which has no newparent,
   * or we're duplicating to a non-yet allocated new non-root, which will have a newparent.
   */
  assert(!newparent == !!newobj);

  if (!newobj) {
    newobj = hwloc_alloc_setup_object(newtopology, src->type, src->os_index);
    if (!newobj)
      return -1;
  }

  /* duplicate all non-object-pointer fields */
  newobj->logical_index = src->logical_index;
  newobj->depth = src->depth;
  newobj->sibling_rank = src->sibling_rank;

  newobj->type = src->type;
  newobj->os_index = src->os_index;
  newobj->gp_index = src->gp_index;
  newobj->symmetric_subtree = src->symmetric_subtree;

  if (src->name)
    newobj->name = hwloc_tma_strdup(tma, src->name);
  if (src->subtype)
    newobj->subtype = hwloc_tma_strdup(tma, src->subtype);
  newobj->userdata = src->userdata;

  newobj->total_memory = src->total_memory;

  memcpy(newobj->attr, src->attr, sizeof(*newobj->attr));

  if (src->type == HWLOC_OBJ_NUMANODE && src->attr->numanode.page_types_len) {
    len = src->attr->numanode.page_types_len * sizeof(struct hwloc_memory_page_type_s);
    newobj->attr->numanode.page_types = hwloc_tma_malloc(tma, len);
    memcpy(newobj->attr->numanode.page_types, src->attr->numanode.page_types, len);
  }

  newobj->cpuset = hwloc_bitmap_tma_dup(tma, src->cpuset);
  newobj->complete_cpuset = hwloc_bitmap_tma_dup(tma, src->complete_cpuset);
  newobj->nodeset = hwloc_bitmap_tma_dup(tma, src->nodeset);
  newobj->complete_nodeset = hwloc_bitmap_tma_dup(tma, src->complete_nodeset);

  hwloc__tma_dup_infos(tma, &newobj->infos, &newobj->infos_count, src->infos, src->infos_count);

  /* find our level */
  if (src->depth < 0) {
    i = HWLOC_SLEVEL_FROM_DEPTH(src->depth);
    level = newtopology->slevels[i].objs;
    level_width = newtopology->slevels[i].nbobjs;
    /* deal with first/last pointers of special levels, even if not really needed */
    if (!newobj->logical_index)
      newtopology->slevels[i].first = newobj;
    if (newobj->logical_index == newtopology->slevels[i].nbobjs - 1)
      newtopology->slevels[i].last = newobj;
  } else {
    level = newtopology->levels[src->depth];
    level_width = newtopology->level_nbobjects[src->depth];
  }
  /* place us for real */
  assert(newobj->logical_index < level_width);
  level[newobj->logical_index] = newobj;
  /* link to already-inserted cousins */
  if (newobj->logical_index > 0 && level[newobj->logical_index-1]) {
    newobj->prev_cousin = level[newobj->logical_index-1];
    level[newobj->logical_index-1]->next_cousin = newobj;
  }
  if (newobj->logical_index < level_width-1 && level[newobj->logical_index+1]) {
    newobj->next_cousin = level[newobj->logical_index+1];
    level[newobj->logical_index+1]->prev_cousin = newobj;
  }

  /* prepare for children */
  if (src->arity) {
    newobj->children = hwloc_tma_malloc(tma, src->arity * sizeof(*newobj->children));
    if (!newobj->children)
      return -1;
  }
  newobj->arity = src->arity;
  newobj->memory_arity = src->memory_arity;
  newobj->io_arity = src->io_arity;
  newobj->misc_arity = src->misc_arity;

  /* actually insert children now */
  for_each_child(child, src) {
    err = hwloc__duplicate_object(newtopology, newobj, NULL, child);
    if (err < 0)
      goto out_with_children;
  }
  for_each_memory_child(child, src) {
    err = hwloc__duplicate_object(newtopology, newobj, NULL, child);
    if (err < 0)
      return err;
  }
  for_each_io_child(child, src) {
    err = hwloc__duplicate_object(newtopology, newobj, NULL, child);
    if (err < 0)
      goto out_with_children;
  }
  for_each_misc_child(child, src) {
    err = hwloc__duplicate_object(newtopology, newobj, NULL, child);
    if (err < 0)
      goto out_with_children;
  }

 out_with_children:

  /* link children if all of them where inserted */
  if (!err) {
    /* only next_sibling is set by insert_by_parent().
     * sibling_rank was set above.
     */
    if (newobj->arity) {
      newobj->children[0]->prev_sibling = NULL;
      for(i=1; i<newobj->arity; i++)
	newobj->children[i]->prev_sibling = newobj->children[i-1];
      newobj->last_child = newobj->children[newobj->arity-1];
    }
    if (newobj->memory_arity) {
      child = newobj->memory_first_child;
      prev = NULL;
      while (child) {
	child->prev_sibling = prev;
	prev = child;
	child = child->next_sibling;
      }
    }
    if (newobj->io_arity) {
      child = newobj->io_first_child;
      prev = NULL;
      while (child) {
	child->prev_sibling = prev;
	prev = child;
	child = child->next_sibling;
      }
    }
    if (newobj->misc_arity) {
      child = newobj->misc_first_child;
      prev = NULL;
      while (child) {
	child->prev_sibling = prev;
	prev = child;
	child = child->next_sibling;
      }
    }
  }

  /* some children insertion may have failed, but some children may have been inserted below us already.
   * keep inserting ourself and let the caller clean the entire tree if we return an error.
   */

  if (newparent) {
    /* no need to check the children insert order here, the source topology
     * is supposed to be OK already, and we have debug asserts.
     */
    hwloc_insert_object_by_parent(newtopology, newparent, newobj);

    /* place us inside our parent children array */
    if (hwloc__obj_type_is_normal(newobj->type))
      newparent->children[newobj->sibling_rank] = newobj;
  }

  return err;
}

static int
hwloc__topology_init (struct hwloc_topology **topologyp, unsigned nblevels, struct hwloc_tma *tma);

/* This function may use a tma, it cannot free() or realloc() */
int
hwloc__topology_dup(hwloc_topology_t *newp,
		    hwloc_topology_t old,
		    struct hwloc_tma *tma)
{
  hwloc_topology_t new;
  hwloc_obj_t newroot;
  hwloc_obj_t oldroot = hwloc_get_root_obj(old);
  unsigned i;
  int err;

  if (!old->is_loaded) {
    errno = EINVAL;
    return -1;
  }

  err = hwloc__topology_init(&new, old->nb_levels_allocated, tma);
  if (err < 0)
    goto out;

  new->flags = old->flags;
  memcpy(new->type_filter, old->type_filter, sizeof(old->type_filter));
  new->is_thissystem = old->is_thissystem;
  new->is_loaded = 1;
  new->pid = old->pid;
  new->next_gp_index = old->next_gp_index;

  memcpy(&new->binding_hooks, &old->binding_hooks, sizeof(old->binding_hooks));

  memcpy(new->support.discovery, old->support.discovery, sizeof(*old->support.discovery));
  memcpy(new->support.cpubind, old->support.cpubind, sizeof(*old->support.cpubind));
  memcpy(new->support.membind, old->support.membind, sizeof(*old->support.membind));
  memcpy(new->support.misc, old->support.misc, sizeof(*old->support.misc));

  new->allowed_cpuset = hwloc_bitmap_tma_dup(tma, old->allowed_cpuset);
  new->allowed_nodeset = hwloc_bitmap_tma_dup(tma, old->allowed_nodeset);

  new->userdata_export_cb = old->userdata_export_cb;
  new->userdata_import_cb = old->userdata_import_cb;
  new->userdata_not_decoded = old->userdata_not_decoded;

  assert(!old->machine_memory.local_memory);
  assert(!old->machine_memory.page_types_len);
  assert(!old->machine_memory.page_types);

  for(i = HWLOC_OBJ_TYPE_MIN; i < HWLOC_OBJ_TYPE_MAX; i++)
    new->type_depth[i] = old->type_depth[i];

  /* duplicate levels and we'll place objects there when duplicating objects */
  new->nb_levels = old->nb_levels;
  assert(new->nb_levels_allocated >= new->nb_levels);
  for(i=1 /* root level already allocated */ ; i<new->nb_levels; i++) {
    new->level_nbobjects[i] = old->level_nbobjects[i];
    new->levels[i] = hwloc_tma_calloc(tma, new->level_nbobjects[i] * sizeof(*new->levels[i]));
  }
  for(i=0; i<HWLOC_NR_SLEVELS; i++) {
    new->slevels[i].nbobjs = old->slevels[i].nbobjs;
    if (new->slevels[i].nbobjs)
      new->slevels[i].objs = hwloc_tma_calloc(tma, new->slevels[i].nbobjs * sizeof(*new->slevels[i].objs));
  }

  /* recursively duplicate object children */
  newroot = hwloc_get_root_obj(new);
  err = hwloc__duplicate_object(new, NULL, newroot, oldroot);
  if (err < 0)
    goto out_with_topology;

  err = hwloc_internal_distances_dup(new, old);
  if (err < 0)
    goto out_with_topology;

  err = hwloc_internal_memattrs_dup(new, old);
  if (err < 0)
    goto out_with_topology;

  err = hwloc_internal_cpukinds_dup(new, old);
  if (err < 0)
    goto out_with_topology;

  /* we connected everything during duplication */
  new->modified = 0;

  /* no need to duplicate backends, topology is already loaded */
  new->backends = NULL;
  new->get_pci_busid_cpuset_backend = NULL;

#ifndef HWLOC_DEBUG
  if (getenv("HWLOC_DEBUG_CHECK"))
#endif
    hwloc_topology_check(new);

  *newp = new;
  return 0;

 out_with_topology:
  assert(!tma || !tma->dontfree); /* this tma cannot fail to allocate */
  hwloc_topology_destroy(new);
 out:
  return -1;
}

int
hwloc_topology_dup(hwloc_topology_t *newp,
		   hwloc_topology_t old)
{
  return hwloc__topology_dup(newp, old, NULL);
}

/* WARNING: The indexes of this array MUST match the ordering that of
   the obj_order_type[] array, below.  Specifically, the values must
   be laid out such that:

       obj_order_type[obj_type_order[N]] = N

   for all HWLOC_OBJ_* values of N.  Put differently:

       obj_type_order[A] = B

   where the A values are in order of the hwloc_obj_type_t enum, and
   the B values are the corresponding indexes of obj_order_type.

   We can't use C99 syntax to initialize this in a little safer manner
   -- bummer.  :-(

   Correctness is asserted in hwloc_topology_init() when debug is enabled.
   */
/***** Make sure you update obj_type_priority[] below as well. *****/
static const unsigned obj_type_order[] = {
    /* first entry is HWLOC_OBJ_MACHINE */  0,
    /* next entry is HWLOC_OBJ_PACKAGE */  4,
    /* next entry is HWLOC_OBJ_CORE */     14,
    /* next entry is HWLOC_OBJ_PU */       18,
    /* next entry is HWLOC_OBJ_L1CACHE */  12,
    /* next entry is HWLOC_OBJ_L2CACHE */  10,
    /* next entry is HWLOC_OBJ_L3CACHE */  8,
    /* next entry is HWLOC_OBJ_L4CACHE */  7,
    /* next entry is HWLOC_OBJ_L5CACHE */  6,
    /* next entry is HWLOC_OBJ_L1ICACHE */ 13,
    /* next entry is HWLOC_OBJ_L2ICACHE */ 11,
    /* next entry is HWLOC_OBJ_L3ICACHE */ 9,
    /* next entry is HWLOC_OBJ_GROUP */    1,
    /* next entry is HWLOC_OBJ_NUMANODE */ 3,
    /* next entry is HWLOC_OBJ_BRIDGE */   15,
    /* next entry is HWLOC_OBJ_PCI_DEVICE */  16,
    /* next entry is HWLOC_OBJ_OS_DEVICE */   17,
    /* next entry is HWLOC_OBJ_MISC */     19,
    /* next entry is HWLOC_OBJ_MEMCACHE */ 2,
    /* next entry is HWLOC_OBJ_DIE */      5
};

#ifndef NDEBUG /* only used in debug check assert if !NDEBUG */
static const hwloc_obj_type_t obj_order_type[] = {
  HWLOC_OBJ_MACHINE,
  HWLOC_OBJ_GROUP,
  HWLOC_OBJ_MEMCACHE,
  HWLOC_OBJ_NUMANODE,
  HWLOC_OBJ_PACKAGE,
  HWLOC_OBJ_DIE,
  HWLOC_OBJ_L5CACHE,
  HWLOC_OBJ_L4CACHE,
  HWLOC_OBJ_L3CACHE,
  HWLOC_OBJ_L3ICACHE,
  HWLOC_OBJ_L2CACHE,
  HWLOC_OBJ_L2ICACHE,
  HWLOC_OBJ_L1CACHE,
  HWLOC_OBJ_L1ICACHE,
  HWLOC_OBJ_CORE,
  HWLOC_OBJ_BRIDGE,
  HWLOC_OBJ_PCI_DEVICE,
  HWLOC_OBJ_OS_DEVICE,
  HWLOC_OBJ_PU,
  HWLOC_OBJ_MISC /* Misc is always a leaf */
};
#endif
/***** Make sure you update obj_type_priority[] below as well. *****/

/* priority to be used when merging identical parent/children object
 * (in merge_useless_child), keep the highest priority one.
 *
 * Always keep Machine/NUMANode/PU/PCIDev/OSDev
 * then Core
 * then Package
 * then Die
 * then Cache,
 * then Instruction Caches
 * then always drop Group/Misc/Bridge.
 *
 * Some type won't actually ever be involved in such merging.
 */
/***** Make sure you update this array when changing the list of types. *****/
static const int obj_type_priority[] = {
  /* first entry is HWLOC_OBJ_MACHINE */     90,
  /* next entry is HWLOC_OBJ_PACKAGE */     40,
  /* next entry is HWLOC_OBJ_CORE */        60,
  /* next entry is HWLOC_OBJ_PU */          100,
  /* next entry is HWLOC_OBJ_L1CACHE */     20,
  /* next entry is HWLOC_OBJ_L2CACHE */     20,
  /* next entry is HWLOC_OBJ_L3CACHE */     20,
  /* next entry is HWLOC_OBJ_L4CACHE */     20,
  /* next entry is HWLOC_OBJ_L5CACHE */     20,
  /* next entry is HWLOC_OBJ_L1ICACHE */    19,
  /* next entry is HWLOC_OBJ_L2ICACHE */    19,
  /* next entry is HWLOC_OBJ_L3ICACHE */    19,
  /* next entry is HWLOC_OBJ_GROUP */       0,
  /* next entry is HWLOC_OBJ_NUMANODE */    100,
  /* next entry is HWLOC_OBJ_BRIDGE */      0,
  /* next entry is HWLOC_OBJ_PCI_DEVICE */  100,
  /* next entry is HWLOC_OBJ_OS_DEVICE */   100,
  /* next entry is HWLOC_OBJ_MISC */        0,
  /* next entry is HWLOC_OBJ_MEMCACHE */    19,
  /* next entry is HWLOC_OBJ_DIE */         30
};

int hwloc_compare_types (hwloc_obj_type_t type1, hwloc_obj_type_t type2)
{
  unsigned order1 = obj_type_order[type1];
  unsigned order2 = obj_type_order[type2];

  /* only normal objects are comparable. others are only comparable with machine */
  if (!hwloc__obj_type_is_normal(type1)
      && hwloc__obj_type_is_normal(type2) && type2 != HWLOC_OBJ_MACHINE)
    return HWLOC_TYPE_UNORDERED;
  if (!hwloc__obj_type_is_normal(type2)
      && hwloc__obj_type_is_normal(type1) && type1 != HWLOC_OBJ_MACHINE)
    return HWLOC_TYPE_UNORDERED;

  return order1 - order2;
}

enum hwloc_obj_cmp_e {
  HWLOC_OBJ_EQUAL = HWLOC_BITMAP_EQUAL,			/**< \brief Equal */
  HWLOC_OBJ_INCLUDED = HWLOC_BITMAP_INCLUDED,		/**< \brief Strictly included into */
  HWLOC_OBJ_CONTAINS = HWLOC_BITMAP_CONTAINS,		/**< \brief Strictly contains */
  HWLOC_OBJ_INTERSECTS = HWLOC_BITMAP_INTERSECTS,	/**< \brief Intersects, but no inclusion! */
  HWLOC_OBJ_DIFFERENT = HWLOC_BITMAP_DIFFERENT		/**< \brief No intersection */
};

static enum hwloc_obj_cmp_e
hwloc_type_cmp(hwloc_obj_t obj1, hwloc_obj_t obj2)
{
  hwloc_obj_type_t type1 = obj1->type;
  hwloc_obj_type_t type2 = obj2->type;
  int compare;

  compare = hwloc_compare_types(type1, type2);
  if (compare == HWLOC_TYPE_UNORDERED)
    return HWLOC_OBJ_DIFFERENT; /* we cannot do better */
  if (compare > 0)
    return HWLOC_OBJ_INCLUDED;
  if (compare < 0)
    return HWLOC_OBJ_CONTAINS;

  if (obj1->type == HWLOC_OBJ_GROUP
      && (obj1->attr->group.kind != obj2->attr->group.kind
	  || obj1->attr->group.subkind != obj2->attr->group.subkind))
    return HWLOC_OBJ_DIFFERENT; /* we cannot do better */

  return HWLOC_OBJ_EQUAL;
}

/*
 * How to compare objects based on cpusets.
 */
static int
hwloc_obj_cmp_sets(hwloc_obj_t obj1, hwloc_obj_t obj2)
{
  hwloc_bitmap_t set1, set2;

  assert(!hwloc__obj_type_is_special(obj1->type));
  assert(!hwloc__obj_type_is_special(obj2->type));

  /* compare cpusets first */
  if (obj1->complete_cpuset && obj2->complete_cpuset) {
    set1 = obj1->complete_cpuset;
    set2 = obj2->complete_cpuset;
  } else {
    set1 = obj1->cpuset;
    set2 = obj2->cpuset;
  }
  if (set1 && set2 && !hwloc_bitmap_iszero(set1) && !hwloc_bitmap_iszero(set2))
    return hwloc_bitmap_compare_inclusion(set1, set2);

  return HWLOC_OBJ_DIFFERENT;
}

/* Compare object cpusets based on complete_cpuset if defined (always correctly ordered),
 * or fallback to the main cpusets (only correctly ordered during early insert before disallowed bits are cleared).
 *
 * This is the sane way to compare object among a horizontal level.
 */
int
hwloc__object_cpusets_compare_first(hwloc_obj_t obj1, hwloc_obj_t obj2)
{
  if (obj1->complete_cpuset && obj2->complete_cpuset)
    return hwloc_bitmap_compare_first(obj1->complete_cpuset, obj2->complete_cpuset);
  else if (obj1->cpuset && obj2->cpuset)
    return hwloc_bitmap_compare_first(obj1->cpuset, obj2->cpuset);
  return 0;
}

/*
 * How to insert objects into the topology.
 *
 * Note: during detection, only the first_child and next_sibling pointers are
 * kept up to date.  Others are computed only once topology detection is
 * complete.
 */

/* merge new object attributes in old.
 * use old if defined, otherwise use new.
 */
static void
merge_insert_equal(hwloc_obj_t new, hwloc_obj_t old)
{
  if (old->os_index == HWLOC_UNKNOWN_INDEX)
    old->os_index = new->os_index;

  if (new->infos_count) {
    /* FIXME: dedup */
    hwloc__move_infos(&old->infos, &old->infos_count,
		      &new->infos, &new->infos_count);
  }

  if (new->name && !old->name) {
    old->name = new->name;
    new->name = NULL;
  }
  if (new->subtype && !old->subtype) {
    old->subtype = new->subtype;
    new->subtype = NULL;
  }

  /* Ignore userdata. It will be NULL before load().
   * It may be non-NULL if alloc+insert_group() after load().
   */

  switch(new->type) {
  case HWLOC_OBJ_NUMANODE:
    if (new->attr->numanode.local_memory && !old->attr->numanode.local_memory) {
      /* no memory in old, use new memory */
      old->attr->numanode.local_memory = new->attr->numanode.local_memory;
      free(old->attr->numanode.page_types);
      old->attr->numanode.page_types_len = new->attr->numanode.page_types_len;
      old->attr->numanode.page_types = new->attr->numanode.page_types;
      new->attr->numanode.page_types = NULL;
      new->attr->numanode.page_types_len = 0;
    }
    /* old->attr->numanode.total_memory will be updated by propagate_total_memory() */
    break;
  case HWLOC_OBJ_L1CACHE:
  case HWLOC_OBJ_L2CACHE:
  case HWLOC_OBJ_L3CACHE:
  case HWLOC_OBJ_L4CACHE:
  case HWLOC_OBJ_L5CACHE:
  case HWLOC_OBJ_L1ICACHE:
  case HWLOC_OBJ_L2ICACHE:
  case HWLOC_OBJ_L3ICACHE:
    if (!old->attr->cache.size)
      old->attr->cache.size = new->attr->cache.size;
    if (!old->attr->cache.linesize)
      old->attr->cache.size = new->attr->cache.linesize;
    if (!old->attr->cache.associativity)
      old->attr->cache.size = new->attr->cache.linesize;
    break;
  default:
    break;
  }
}

/* returns the result of merge, or NULL if not merged */
static __hwloc_inline hwloc_obj_t
hwloc__insert_try_merge_group(hwloc_topology_t topology, hwloc_obj_t old, hwloc_obj_t new)
{
  if (new->type == HWLOC_OBJ_GROUP && old->type == HWLOC_OBJ_GROUP) {
    /* which group do we keep? */
    if (new->attr->group.dont_merge) {
      if (old->attr->group.dont_merge)
	/* nobody wants to be merged */
	return NULL;

      /* keep the new one, it doesn't want to be merged */
      hwloc_replace_linked_object(old, new);
      topology->modified = 1;
      return new;

    } else {
      if (old->attr->group.dont_merge)
	/* keep the old one, it doesn't want to be merged */
	return old;

      /* compare subkinds to decide which group to keep */
      if (new->attr->group.kind < old->attr->group.kind) {
        /* keep smaller kind */
	hwloc_replace_linked_object(old, new);
        topology->modified = 1;
      }
      return old;
    }
  }

  if (new->type == HWLOC_OBJ_GROUP && !new->attr->group.dont_merge) {

    if (old->type == HWLOC_OBJ_PU && new->attr->group.kind == HWLOC_GROUP_KIND_MEMORY)
      /* Never merge Memory groups with PU, we don't want to attach Memory under PU */
      return NULL;

    /* Remove the Group now. The normal ignore code path wouldn't tell us whether the Group was removed or not,
     * while some callers need to know (at least hwloc_topology_insert_group()).
     */
    return old;

  } else if (old->type == HWLOC_OBJ_GROUP && !old->attr->group.dont_merge) {

    if (new->type == HWLOC_OBJ_PU && old->attr->group.kind == HWLOC_GROUP_KIND_MEMORY)
      /* Never merge Memory groups with PU, we don't want to attach Memory under PU */
      return NULL;

    /* Replace the Group with the new object contents
     * and let the caller free the new object
     */
    hwloc_replace_linked_object(old, new);
    topology->modified = 1;
    return old;

  } else {
    /* cannot merge */
    return NULL;
  }
}

/*
 * The main insertion routine, only used for CPU-side object (normal types)
 * uisng cpuset only (or complete_cpuset).
 *
 * Try to insert OBJ in CUR, recurse if needed.
 * Returns the object if it was inserted,
 * the remaining object it was merged,
 * NULL if failed to insert.
 */
static struct hwloc_obj *
hwloc___insert_object_by_cpuset(struct hwloc_topology *topology, hwloc_obj_t cur, hwloc_obj_t obj,
			        const char *reason)
{
  hwloc_obj_t child, next_child = NULL, tmp;
  /* These will always point to the pointer to their next last child. */
  hwloc_obj_t *cur_children = &cur->first_child;
  hwloc_obj_t *obj_children = &obj->first_child;
  /* Pointer where OBJ should be put */
  hwloc_obj_t *putp = NULL; /* OBJ position isn't found yet */

  assert(!hwloc__obj_type_is_memory(obj->type));

  /* Iteration with prefetching to be completely safe against CHILD removal.
   * The list is already sorted by cpuset, and there's no intersection between siblings.
   */
  for (child = cur->first_child, child ? next_child = child->next_sibling : NULL;
       child;
       child = next_child, child ? next_child = child->next_sibling : NULL) {

    int res = hwloc_obj_cmp_sets(obj, child);
    int setres = res;

    if (res == HWLOC_OBJ_EQUAL) {
      hwloc_obj_t merged = hwloc__insert_try_merge_group(topology, child, obj);
      if (merged)
	return merged;
      /* otherwise compare actual types to decide of the inclusion */
      res = hwloc_type_cmp(obj, child);
    }

    switch (res) {
      case HWLOC_OBJ_EQUAL:
	/* Two objects with same type.
	 * Groups are handled above.
	 */
	merge_insert_equal(obj, child);
	/* Already present, no need to insert.  */
	return child;

      case HWLOC_OBJ_INCLUDED:
	/* OBJ is strictly contained is some child of CUR, go deeper.  */
	return hwloc___insert_object_by_cpuset(topology, child, obj, reason);

      case HWLOC_OBJ_INTERSECTS:
        report_insert_error(obj, child, "intersection without inclusion", reason);
	goto putback;

      case HWLOC_OBJ_DIFFERENT:
        /* OBJ should be a child of CUR before CHILD, mark its position if not found yet. */
	if (!putp && hwloc__object_cpusets_compare_first(obj, child) < 0)
	  /* Don't insert yet, there could be intersect errors later */
	  putp = cur_children;
	/* Advance cur_children.  */
	cur_children = &child->next_sibling;
	break;

      case HWLOC_OBJ_CONTAINS:
	/* OBJ contains CHILD, remove CHILD from CUR */
	*cur_children = child->next_sibling;
	child->next_sibling = NULL;
	/* Put CHILD in OBJ */
	*obj_children = child;
	obj_children = &child->next_sibling;
	child->parent = obj;
	if (setres == HWLOC_OBJ_EQUAL) {
	  obj->memory_first_child = child->memory_first_child;
	  child->memory_first_child = NULL;
	  for(tmp=obj->memory_first_child; tmp; tmp = tmp->next_sibling)
	    tmp->parent = obj;
	}
	break;
    }
  }
  /* cur/obj_children points to last CUR/OBJ child next_sibling pointer, which must be NULL. */
  assert(!*obj_children);
  assert(!*cur_children);

  /* Put OBJ where it belongs, or in last in CUR's children.  */
  if (!putp)
    putp = cur_children;
  obj->next_sibling = *putp;
  *putp = obj;
  obj->parent = cur;

  topology->modified = 1;
  return obj;

 putback:
  /* OBJ cannot be inserted.
   * Put-back OBJ children in CUR and return an error.
   */
  if (putp)
    cur_children = putp; /* No need to try to insert before where OBJ was supposed to go */
  else
    cur_children = &cur->first_child; /* Start from the beginning */
  /* We can insert in order, but there can be holes in the middle. */
  while ((child = obj->first_child) != NULL) {
    /* Remove from OBJ */
    obj->first_child = child->next_sibling;
    /* Find child position in CUR, and reinsert it. */
    while (*cur_children && hwloc__object_cpusets_compare_first(*cur_children, child) < 0)
      cur_children = &(*cur_children)->next_sibling;
    child->next_sibling = *cur_children;
    *cur_children = child;
    child->parent = cur;
  }
  return NULL;
}

/* this differs from hwloc_get_obj_covering_cpuset() by:
 * - not looking at the parent cpuset first, which means we can insert
 *   below root even if root PU bits are not set yet (PU are inserted later).
 * - returning the first child that exactly matches instead of walking down in case
 *   of identical children.
 */
static struct hwloc_obj *
hwloc__find_obj_covering_memory_cpuset(struct hwloc_topology *topology, hwloc_obj_t parent, hwloc_bitmap_t cpuset)
{
  hwloc_obj_t child = hwloc_get_child_covering_cpuset(topology, cpuset, parent);
  if (!child)
    return parent;
  if (child && hwloc_bitmap_isequal(child->cpuset, cpuset))
    return child;
  return hwloc__find_obj_covering_memory_cpuset(topology, child, cpuset);
}

static struct hwloc_obj *
hwloc__find_insert_memory_parent(struct hwloc_topology *topology, hwloc_obj_t obj,
                                 const char *reason)
{
  hwloc_obj_t parent, group, result;

  if (hwloc_bitmap_iszero(obj->cpuset)) {
    /* CPU-less go in dedicated group below root */
    parent = topology->levels[0][0];

  } else {
    /* find the highest obj covering the cpuset */
    parent = hwloc__find_obj_covering_memory_cpuset(topology, topology->levels[0][0], obj->cpuset);
    if (!parent) {
      /* fallback to root */
      parent = hwloc_get_root_obj(topology);
    }

    if (parent->type == HWLOC_OBJ_PU) {
      /* Never attach to PU, try parent */
      parent = parent->parent;
      assert(parent);
    }

    /* TODO: if root->cpuset was updated earlier, we would be sure whether the group will remain identical to root */
    if (parent != topology->levels[0][0] && hwloc_bitmap_isequal(parent->cpuset, obj->cpuset))
      /* that parent is fine */
      return parent;
  }

  if (!hwloc_filter_check_keep_object_type(topology, HWLOC_OBJ_GROUP))
    /* even if parent isn't perfect, we don't want an intermediate group */
    return parent;

  /* need to insert an intermediate group for attaching the NUMA node */
  group = hwloc_alloc_setup_object(topology, HWLOC_OBJ_GROUP, HWLOC_UNKNOWN_INDEX);
  if (!group)
    /* failed to create the group, fallback to larger parent */
    return parent;

  group->attr->group.kind = HWLOC_GROUP_KIND_MEMORY;
  group->cpuset = hwloc_bitmap_dup(obj->cpuset);
  group->complete_cpuset = hwloc_bitmap_dup(obj->complete_cpuset);
  /* we could duplicate nodesets too but hwloc__insert_object_by_cpuset()
   * doesn't actually need it. and it could prevent future calls from reusing
   * that groups for other NUMA nodes.
   */
  if (!group->cpuset != !obj->cpuset
      || !group->complete_cpuset != !obj->complete_cpuset) {
    /* failed to create the group, fallback to larger parent */
    hwloc_free_unlinked_object(group);
    return parent;
  }

  result = hwloc__insert_object_by_cpuset(topology, parent, group, reason);
  if (!result) {
    /* failed to insert, fallback to larger parent */
    return parent;
  }

  assert(result == group);
  return group;
}

/* only works for MEMCACHE and NUMAnode with a single bit in nodeset */
static hwloc_obj_t
hwloc___attach_memory_object_by_nodeset(struct hwloc_topology *topology, hwloc_obj_t parent,
					hwloc_obj_t obj, const char *reason)
{
  hwloc_obj_t *curp = &parent->memory_first_child;
  unsigned first = hwloc_bitmap_first(obj->nodeset);

  while (*curp) {
    hwloc_obj_t cur = *curp;
    unsigned curfirst = hwloc_bitmap_first(cur->nodeset);

    if (first < curfirst) {
      /* insert before cur */
      obj->next_sibling = cur;
      *curp = obj;
      obj->memory_first_child = NULL;
      obj->parent = parent;
      topology->modified = 1;
      return obj;
    }

    if (first == curfirst) {
      /* identical nodeset */
      if (obj->type == HWLOC_OBJ_NUMANODE) {
	if (cur->type == HWLOC_OBJ_NUMANODE) {
	  /* identical NUMA nodes? ignore the new one */
          report_insert_error(obj, cur, "NUMAnodes with identical nodesets", reason);
	  return NULL;
	}
	assert(cur->type == HWLOC_OBJ_MEMCACHE);
	/* insert the new NUMA node below that existing memcache */
	return hwloc___attach_memory_object_by_nodeset(topology, cur, obj, reason);

      } else {
	assert(obj->type == HWLOC_OBJ_MEMCACHE);
	if (cur->type == HWLOC_OBJ_MEMCACHE) {
	  if (cur->attr->cache.depth == obj->attr->cache.depth)
	    /* memcache with same nodeset and depth, ignore the new one */
	    return NULL;
	  if (cur->attr->cache.depth > obj->attr->cache.depth)
	    /* memcache with higher cache depth is actually *higher* in the hierarchy
	     * (depth starts from the NUMA node).
	     * insert the new memcache below the existing one
	     */
	    return hwloc___attach_memory_object_by_nodeset(topology, cur, obj, reason);
	}
	/* insert the memcache above the existing memcache or numa node */
	obj->next_sibling = cur->next_sibling;
	cur->next_sibling = NULL;
	obj->memory_first_child = cur;
	cur->parent = obj;
	*curp = obj;
	obj->parent = parent;
	topology->modified = 1;
	return obj;
      }
    }

    curp = &cur->next_sibling;
  }

  /* append to the end of the list */
  obj->next_sibling = NULL;
  *curp = obj;
  obj->memory_first_child = NULL;
  obj->parent = parent;
  topology->modified = 1;
  return obj;
}

/* Attach the given memory object below the given normal parent.
 *
 * Only the nodeset is used to find the location inside memory children below parent.
 *
 * Nodeset inclusion inside the given memory hierarchy is guaranteed by this function,
 * but nodesets are not propagated to CPU-side parent yet. It will be done by
 * propagate_nodeset() later.
 */
struct hwloc_obj *
hwloc__attach_memory_object(struct hwloc_topology *topology, hwloc_obj_t parent,
			    hwloc_obj_t obj, const char *reason)
{
  hwloc_obj_t result;

  assert(parent);
  assert(hwloc__obj_type_is_normal(parent->type));

  /* Check the nodeset */
  if (!obj->nodeset || hwloc_bitmap_iszero(obj->nodeset))
    return NULL;
  /* Initialize or check the complete nodeset */
  if (!obj->complete_nodeset) {
    obj->complete_nodeset = hwloc_bitmap_dup(obj->nodeset);
  } else if (!hwloc_bitmap_isincluded(obj->nodeset, obj->complete_nodeset)) {
    return NULL;
  }
  /* Neither ACPI nor Linux support multinode mscache */
  assert(hwloc_bitmap_weight(obj->nodeset) == 1);

#if 0
  /* TODO: enable this instead of hack in fixup_sets once NUMA nodes are inserted late */
  /* copy the parent cpuset in case it's larger than expected.
   * we could also keep the cpuset smaller than the parent and say that a normal-parent
   * can have multiple memory children with smaller cpusets.
   * However, the user decided the ignore Groups, so hierarchy/locality loss is expected.
   */
  hwloc_bitmap_copy(obj->cpuset, parent->cpuset);
  hwloc_bitmap_copy(obj->complete_cpuset, parent->complete_cpuset);
#endif

  result = hwloc___attach_memory_object_by_nodeset(topology, parent, obj, reason);
  if (result == obj) {
    /* Add the bit to the top sets, and to the parent CPU-side object */
    if (obj->type == HWLOC_OBJ_NUMANODE) {
      hwloc_bitmap_set(topology->levels[0][0]->nodeset, obj->os_index);
      hwloc_bitmap_set(topology->levels[0][0]->complete_nodeset, obj->os_index);
    }
  }
  if (result != obj) {
    /* either failed to insert, or got merged, free the original object */
    hwloc_free_unlinked_object(obj);
  }
  return result;
}

/* insertion routine that lets you change the error reporting callback */
struct hwloc_obj *
hwloc__insert_object_by_cpuset(struct hwloc_topology *topology, hwloc_obj_t root,
			       hwloc_obj_t obj, const char *reason)
{
  struct hwloc_obj *result;

#ifdef HWLOC_DEBUG
  assert(!hwloc__obj_type_is_special(obj->type));

  /* we need at least one non-NULL set (normal or complete, cpuset or nodeset) */
  assert(obj->cpuset || obj->complete_cpuset || obj->nodeset || obj->complete_nodeset);
  /* we support the case where all of them are empty.
   * it may happen when hwloc__find_insert_memory_parent()
   * inserts a Group for a CPU-less NUMA-node.
   */
#endif

  if (hwloc__obj_type_is_memory(obj->type)) {
    if (!root) {
      root = hwloc__find_insert_memory_parent(topology, obj, reason);
      if (!root) {
	hwloc_free_unlinked_object(obj);
	return NULL;
      }
    }
    return hwloc__attach_memory_object(topology, root, obj, reason);
  }

  if (!root)
    /* Start at the top. */
    root = topology->levels[0][0];

  result = hwloc___insert_object_by_cpuset(topology, root, obj, reason);
  if (result && result->type == HWLOC_OBJ_PU) {
      /* Add the bit to the top sets */
      if (hwloc_bitmap_isset(result->cpuset, result->os_index))
	hwloc_bitmap_set(topology->levels[0][0]->cpuset, result->os_index);
      hwloc_bitmap_set(topology->levels[0][0]->complete_cpuset, result->os_index);
  }
  if (result != obj) {
    /* either failed to insert, or got merged, free the original object */
    hwloc_free_unlinked_object(obj);
  }
  return result;
}

/* the default insertion routine warns in case of error.
 * it's used by most backends */
void
hwloc_insert_object_by_parent(struct hwloc_topology *topology, hwloc_obj_t parent, hwloc_obj_t obj)
{
  hwloc_obj_t *current;

  if (obj->type == HWLOC_OBJ_MISC) {
    /* Append to the end of the Misc list */
    for (current = &parent->misc_first_child; *current; current = &(*current)->next_sibling);
  } else if (hwloc__obj_type_is_io(obj->type)) {
    /* Append to the end of the I/O list */
    for (current = &parent->io_first_child; *current; current = &(*current)->next_sibling);
  } else if (hwloc__obj_type_is_memory(obj->type)) {
    /* Append to the end of the memory list */
    for (current = &parent->memory_first_child; *current; current = &(*current)->next_sibling);
    /* Add the bit to the top sets */
    if (obj->type == HWLOC_OBJ_NUMANODE) {
      if (hwloc_bitmap_isset(obj->nodeset, obj->os_index))
	hwloc_bitmap_set(topology->levels[0][0]->nodeset, obj->os_index);
      hwloc_bitmap_set(topology->levels[0][0]->complete_nodeset, obj->os_index);
    }
  } else {
    /* Append to the end of the list.
     * The caller takes care of inserting children in the right cpuset order, without intersection between them.
     * Duplicating doesn't need to check the order since the source topology is supposed to be OK already.
     * XML reorders if needed, and fails on intersecting siblings.
     * Other callers just insert random objects such as I/O or Misc, no cpuset issue there.
     */
    for (current = &parent->first_child; *current; current = &(*current)->next_sibling);
    /* Add the bit to the top sets */
    if (obj->type == HWLOC_OBJ_PU) {
      if (hwloc_bitmap_isset(obj->cpuset, obj->os_index))
	hwloc_bitmap_set(topology->levels[0][0]->cpuset, obj->os_index);
      hwloc_bitmap_set(topology->levels[0][0]->complete_cpuset, obj->os_index);
    }
  }

  *current = obj;
  obj->parent = parent;
  obj->next_sibling = NULL;
  topology->modified = 1;
}

hwloc_obj_t
hwloc_alloc_setup_object(hwloc_topology_t topology,
			 hwloc_obj_type_t type, unsigned os_index)
{
  struct hwloc_obj *obj = hwloc_tma_malloc(topology->tma, sizeof(*obj));
  if (!obj)
    return NULL;
  memset(obj, 0, sizeof(*obj));
  obj->type = type;
  obj->os_index = os_index;
  obj->gp_index = topology->next_gp_index++;
  obj->attr = hwloc_tma_malloc(topology->tma, sizeof(*obj->attr));
  if (!obj->attr) {
    assert(!topology->tma || !topology->tma->dontfree); /* this tma cannot fail to allocate */
    free(obj);
    return NULL;
  }
  memset(obj->attr, 0, sizeof(*obj->attr));
  /* do not allocate the cpuset here, let the caller do it */
  return obj;
}

hwloc_obj_t
hwloc_topology_alloc_group_object(struct hwloc_topology *topology)
{
  if (!topology->is_loaded) {
    /* this could actually work, see insert() below */
    errno = EINVAL;
    return NULL;
  }
  if (topology->adopted_shmem_addr) {
    errno = EPERM;
    return NULL;
  }
  return hwloc_alloc_setup_object(topology, HWLOC_OBJ_GROUP, HWLOC_UNKNOWN_INDEX);
}

int
hwloc_topology_free_group_object(struct hwloc_topology *topology, hwloc_obj_t obj)
{
  if (!topology->is_loaded) {
    /* this could actually work, see insert() below */
    errno = EINVAL;
    return -1;
  }
  if (topology->adopted_shmem_addr) {
    errno = EPERM;
    return -1;
  }
  hwloc_free_unlinked_object(obj);
  return 0;
}

static void hwloc_propagate_symmetric_subtree(hwloc_topology_t topology, hwloc_obj_t root);
static void propagate_total_memory(hwloc_obj_t obj);
static void hwloc_set_group_depth(hwloc_topology_t topology);
static void hwloc_connect_children(hwloc_obj_t parent);
static int hwloc_connect_levels(hwloc_topology_t topology);
static int hwloc_connect_special_levels(hwloc_topology_t topology);

hwloc_obj_t
hwloc_topology_insert_group_object(struct hwloc_topology *topology, hwloc_obj_t obj)
{
  hwloc_obj_t res, root, child;
  int cmp;

  if (!topology->is_loaded) {
    /* this could actually work, we would just need to disable connect_children/levels below */
    hwloc_free_unlinked_object(obj);
    errno = EINVAL;
    return NULL;
  }
  if (topology->adopted_shmem_addr) {
    hwloc_free_unlinked_object(obj);
    errno = EPERM;
    return NULL;
  }

  if (topology->type_filter[HWLOC_OBJ_GROUP] == HWLOC_TYPE_FILTER_KEEP_NONE) {
    hwloc_free_unlinked_object(obj);
    errno = EINVAL;
    return NULL;
  }

  root = hwloc_get_root_obj(topology);
  if (obj->cpuset)
    hwloc_bitmap_and(obj->cpuset, obj->cpuset, root->cpuset);
  if (obj->complete_cpuset)
    hwloc_bitmap_and(obj->complete_cpuset, obj->complete_cpuset, root->complete_cpuset);
  if (obj->nodeset)
    hwloc_bitmap_and(obj->nodeset, obj->nodeset, root->nodeset);
  if (obj->complete_nodeset)
    hwloc_bitmap_and(obj->complete_nodeset, obj->complete_nodeset, root->complete_nodeset);

  if ((!obj->cpuset || hwloc_bitmap_iszero(obj->cpuset))
      && (!obj->complete_cpuset || hwloc_bitmap_iszero(obj->complete_cpuset))) {
    /* we'll insert by cpuset, so build cpuset from the nodeset */
    hwloc_const_bitmap_t nodeset = obj->nodeset ? obj->nodeset : obj->complete_nodeset;
    hwloc_obj_t numa;

    if ((!obj->nodeset || hwloc_bitmap_iszero(obj->nodeset))
	&& (!obj->complete_nodeset || hwloc_bitmap_iszero(obj->complete_nodeset))) {
      hwloc_free_unlinked_object(obj);
      errno = EINVAL;
      return NULL;
    }

    if (!obj->cpuset) {
      obj->cpuset = hwloc_bitmap_alloc();
      if (!obj->cpuset) {
	hwloc_free_unlinked_object(obj);
	return NULL;
      }
    }

    numa = NULL;
    while ((numa = hwloc_get_next_obj_by_type(topology, HWLOC_OBJ_NUMANODE, numa)) != NULL)
      if (hwloc_bitmap_isset(nodeset, numa->os_index))
	hwloc_bitmap_or(obj->cpuset, obj->cpuset, numa->cpuset);
  }
  /* FIXME insert by nodeset to group NUMAs even if CPUless? */

  cmp = hwloc_obj_cmp_sets(obj, root);
  if (cmp == HWLOC_OBJ_INCLUDED) {
    res = hwloc__insert_object_by_cpuset(topology, NULL, obj, NULL /* do not show errors on stdout */);
  } else {
    /* just merge root */
    hwloc_free_unlinked_object(obj);
    res = root;
  }

  if (!res)
    return NULL;

  if (res != obj && res->type != HWLOC_OBJ_GROUP)
    /* merged, not into a Group, nothing to update */
    return res;

  /* res == obj means that the object was inserted.
   * We need to reconnect levels, fill all its cpu/node sets,
   * compute its total memory, group depth, etc.
   *
   * res != obj usually means that our new group was merged into an
   * existing object, no need to recompute anything.
   * However, if merging with an existing group, depending on their kinds,
   * the contents of obj may overwrite the contents of the old group.
   * This requires reconnecting levels, filling sets, recomputing total memory, etc.
   */

  /* properly inserted */
  hwloc_obj_add_children_sets(res);
  if (hwloc_topology_reconnect(topology, 0) < 0)
    return NULL;

  /* Compute group total_memory. */
  res->total_memory = 0;
  for_each_child(child, res)
    res->total_memory += child->total_memory;
  for_each_memory_child(child, res)
    res->total_memory += child->total_memory;

  hwloc_propagate_symmetric_subtree(topology, topology->levels[0][0]);
  hwloc_set_group_depth(topology);

#ifndef HWLOC_DEBUG
  if (getenv("HWLOC_DEBUG_CHECK"))
#endif
    hwloc_topology_check(topology);

  return res;
}

hwloc_obj_t
hwloc_topology_insert_misc_object(struct hwloc_topology *topology, hwloc_obj_t parent, const char *name)
{
  hwloc_obj_t obj;

  if (topology->type_filter[HWLOC_OBJ_MISC] == HWLOC_TYPE_FILTER_KEEP_NONE) {
    errno = EINVAL;
    return NULL;
  }

  if (!topology->is_loaded) {
    errno = EINVAL;
    return NULL;
  }
  if (topology->adopted_shmem_addr) {
    errno = EPERM;
    return NULL;
  }

  obj = hwloc_alloc_setup_object(topology, HWLOC_OBJ_MISC, HWLOC_UNKNOWN_INDEX);
  if (name)
    obj->name = strdup(name);

  hwloc_insert_object_by_parent(topology, parent, obj);

  /* FIXME: only connect misc parent children and misc level,
   * but this API is likely not performance critical anyway
   */
  hwloc_topology_reconnect(topology, 0);

#ifndef HWLOC_DEBUG
  if (getenv("HWLOC_DEBUG_CHECK"))
#endif
    hwloc_topology_check(topology);

  return obj;
}

/* assuming set is included in the topology complete_cpuset
 * and all objects have a proper complete_cpuset,
 * return the best one containing set.
 * if some object are equivalent (same complete_cpuset), return the highest one.
 */
static hwloc_obj_t
hwloc_get_highest_obj_covering_complete_cpuset (hwloc_topology_t topology, hwloc_const_cpuset_t set)
{
  hwloc_obj_t current = hwloc_get_root_obj(topology);
  hwloc_obj_t child;

  if (hwloc_bitmap_isequal(set, current->complete_cpuset))
    /* root cpuset is exactly what we want, no need to look at children, we want the highest */
    return current;

 recurse:
  /* find the right child */
  for_each_child(child, current) {
    if (hwloc_bitmap_isequal(set, child->complete_cpuset))
      /* child puset is exactly what we want, no need to look at children, we want the highest */
      return child;
    if (!hwloc_bitmap_iszero(child->complete_cpuset) && hwloc_bitmap_isincluded(set, child->complete_cpuset))
      break;
  }

  if (child) {
    current = child;
    goto recurse;
  }

  /* no better child */
  return current;
}

hwloc_obj_t
hwloc_find_insert_io_parent_by_complete_cpuset(struct hwloc_topology *topology, hwloc_cpuset_t cpuset)
{
  hwloc_obj_t group_obj, largeparent, parent;

  /* restrict to the existing complete cpuset to avoid errors later */
  hwloc_bitmap_and(cpuset, cpuset, hwloc_topology_get_complete_cpuset(topology));
  if (hwloc_bitmap_iszero(cpuset))
    /* remaining cpuset is empty, invalid */
    return NULL;

  largeparent = hwloc_get_highest_obj_covering_complete_cpuset(topology, cpuset);
  if (hwloc_bitmap_isequal(largeparent->complete_cpuset, cpuset)
      || !hwloc_filter_check_keep_object_type(topology, HWLOC_OBJ_GROUP))
    /* Found a valid object (normal case) */
    return largeparent;

  /* we need to insert an intermediate group */
  group_obj = hwloc_alloc_setup_object(topology, HWLOC_OBJ_GROUP, HWLOC_UNKNOWN_INDEX);
  if (!group_obj)
    /* Failed to insert the exact Group, fallback to largeparent */
    return largeparent;

  group_obj->complete_cpuset = hwloc_bitmap_dup(cpuset);
  hwloc_bitmap_and(cpuset, cpuset, hwloc_topology_get_topology_cpuset(topology));
  group_obj->cpuset = hwloc_bitmap_dup(cpuset);
  group_obj->attr->group.kind = HWLOC_GROUP_KIND_IO;
  parent = hwloc__insert_object_by_cpuset(topology, largeparent, group_obj, "topology:io_parent");
  if (!parent)
    /* Failed to insert the Group, maybe a conflicting cpuset */
    return largeparent;

  /* Group couldn't get merged or we would have gotten the right largeparent earlier */
  assert(parent == group_obj);

  /* Group inserted without being merged, everything OK, setup its sets */
  hwloc_obj_add_children_sets(group_obj);

  return parent;
}

static int hwloc_memory_page_type_compare(const void *_a, const void *_b)
{
  const struct hwloc_memory_page_type_s *a = _a;
  const struct hwloc_memory_page_type_s *b = _b;
  /* consider 0 as larger so that 0-size page_type go to the end */
  if (!b->size)
    return -1;
  /* don't cast a-b in int since those are ullongs */
  if (b->size == a->size)
    return 0;
  return a->size < b->size ? -1 : 1;
}

/* Propagate memory counts */
static void
propagate_total_memory(hwloc_obj_t obj)
{
  hwloc_obj_t child;
  unsigned i;

  /* reset total before counting local and children memory */
  obj->total_memory = 0;

  /* Propagate memory up. */
  for_each_child(child, obj) {
    propagate_total_memory(child);
    obj->total_memory += child->total_memory;
  }
  for_each_memory_child(child, obj) {
    propagate_total_memory(child);
    obj->total_memory += child->total_memory;
  }
  /* No memory under I/O or Misc */

  if (obj->type == HWLOC_OBJ_NUMANODE) {
    obj->total_memory += obj->attr->numanode.local_memory;

    if (obj->attr->numanode.page_types_len) {
      /* By the way, sort the page_type array.
       * Cannot do it on insert since some backends (e.g. XML) add page_types after inserting the object.
       */
      qsort(obj->attr->numanode.page_types, obj->attr->numanode.page_types_len, sizeof(*obj->attr->numanode.page_types), hwloc_memory_page_type_compare);
      /* Ignore 0-size page_types, they are at the end */
      for(i=obj->attr->numanode.page_types_len; i>=1; i--)
	if (obj->attr->numanode.page_types[i-1].size)
	  break;
      obj->attr->numanode.page_types_len = i;
    }
  }
}

/* Now that root sets are ready, propagate them to children
 * by allocating missing sets and restricting existing ones.
 */
static void
fixup_sets(hwloc_obj_t obj)
{
  int in_memory_list;
  hwloc_obj_t child;

  child = obj->first_child;
  in_memory_list = 0;
  /* iterate over normal children first, we'll come back for memory children later */

  /* FIXME: if memory objects are inserted late, we should update their cpuset and complete_cpuset at insertion instead of here */
 iterate:
  while (child) {
    /* our cpuset must be included in our parent's one */
    hwloc_bitmap_and(child->cpuset, child->cpuset, obj->cpuset);
    hwloc_bitmap_and(child->nodeset, child->nodeset, obj->nodeset);
    /* our complete_cpuset must be included in our parent's one, but can be larger than our cpuset */
    if (child->complete_cpuset) {
      hwloc_bitmap_and(child->complete_cpuset, child->complete_cpuset, obj->complete_cpuset);
    } else {
      child->complete_cpuset = hwloc_bitmap_dup(child->cpuset);
    }
    if (child->complete_nodeset) {
      hwloc_bitmap_and(child->complete_nodeset, child->complete_nodeset, obj->complete_nodeset);
    } else {
      child->complete_nodeset = hwloc_bitmap_dup(child->nodeset);
    }

    if (hwloc_obj_type_is_memory(child->type)) {
      /* update memory children cpusets in case some CPU-side parent was removed */
      hwloc_bitmap_copy(child->cpuset, obj->cpuset);
      hwloc_bitmap_copy(child->complete_cpuset, obj->complete_cpuset);
    }

    fixup_sets(child);
    child = child->next_sibling;
  }

  /* switch to memory children list if any */
  if (!in_memory_list && obj->memory_first_child) {
    child = obj->memory_first_child;
    in_memory_list = 1;
    goto iterate;
  }

  /* No sets in I/O or Misc */
}

/* Setup object cpusets/nodesets by OR'ing its children. */
int
hwloc_obj_add_other_obj_sets(hwloc_obj_t dst, hwloc_obj_t src)
{
#define ADD_OTHER_OBJ_SET(_dst, _src, _set)					\
  if ((_src)->_set) {								\
    if (!(_dst)->_set)								\
      (_dst)->_set = hwloc_bitmap_alloc();					\
    if (!(_dst)->_set								\
        || hwloc_bitmap_or((_dst)->_set, (_dst)->_set, (_src)->_set) < 0)	\
      return -1;								\
  }
  ADD_OTHER_OBJ_SET(dst, src, cpuset);
  ADD_OTHER_OBJ_SET(dst, src, complete_cpuset);
  ADD_OTHER_OBJ_SET(dst, src, nodeset);
  ADD_OTHER_OBJ_SET(dst, src, complete_nodeset);
  return 0;
}

int
hwloc_obj_add_children_sets(hwloc_obj_t obj)
{
  hwloc_obj_t child;
  for_each_child(child, obj) {
    hwloc_obj_add_other_obj_sets(obj, child);
  }
  /* No need to look at Misc children, they contain no PU. */
  return 0;
}

/* CPU objects are inserted by cpusets, we know their cpusets are properly included.
 * We just need fixup_sets() to make sure they aren't too wide.
 *
 * Within each memory hierarchy, nodeset are consistent as well.
 * However they must be propagated to their CPU-side parents.
 *
 * A memory object nodeset consists of NUMA nodes below it.
 * A normal object nodeset consists in NUMA nodes attached to any
 * of its children or parents.
 */
static void
propagate_nodeset(hwloc_obj_t obj)
{
  hwloc_obj_t child;

  /* Start our nodeset from the parent one.
   * It was emptied at root, and it's being filled with local nodes
   * in that branch of the tree as we recurse down.
   */
  if (!obj->nodeset)
    obj->nodeset = hwloc_bitmap_alloc();
  if (obj->parent)
    hwloc_bitmap_copy(obj->nodeset, obj->parent->nodeset);
  else
    hwloc_bitmap_zero(obj->nodeset);

  /* Don't clear complete_nodeset, just make sure it contains nodeset.
   * We cannot clear the complete_nodeset at root and rebuild it down because
   * some bits may correspond to offline/disallowed NUMA nodes missing in the topology.
   */
  if (!obj->complete_nodeset)
    obj->complete_nodeset = hwloc_bitmap_dup(obj->nodeset);
  else
    hwloc_bitmap_or(obj->complete_nodeset, obj->complete_nodeset, obj->nodeset);

  /* now add our local nodeset */
  for_each_memory_child(child, obj) {
    /* add memory children nodesets to ours */
    hwloc_bitmap_or(obj->nodeset, obj->nodeset, child->nodeset);
    hwloc_bitmap_or(obj->complete_nodeset, obj->complete_nodeset, child->complete_nodeset);
    /* no need to recurse because hwloc__attach_memory_object()
     * makes sure nodesets are consistent within each memory hierarchy.
     */
  }

  /* Propagate our nodeset to CPU children. */
  for_each_child(child, obj) {
    propagate_nodeset(child);
  }

  /* Propagate CPU children specific nodesets back to us.
   *
   * We cannot merge these two loops because we don't want to first child
   * nodeset to be propagated back to us and then down to the second child.
   * Each child may have its own local nodeset,
   * each of them is propagated to us, but not to other children.
   */
  for_each_child(child, obj) {
    hwloc_bitmap_or(obj->nodeset, obj->nodeset, child->nodeset);
    hwloc_bitmap_or(obj->complete_nodeset, obj->complete_nodeset, child->complete_nodeset);
  }

  /* No nodeset under I/O or Misc */

}

static void
remove_unused_sets(hwloc_topology_t topology, hwloc_obj_t obj)
{
  hwloc_obj_t child;

  hwloc_bitmap_and(obj->cpuset, obj->cpuset, topology->allowed_cpuset);
  hwloc_bitmap_and(obj->nodeset, obj->nodeset, topology->allowed_nodeset);

  for_each_child(child, obj)
    remove_unused_sets(topology, child);
  for_each_memory_child(child, obj)
    remove_unused_sets(topology, child);
  /* No cpuset under I/O or Misc */
}

static void
hwloc__filter_bridges(hwloc_topology_t topology, hwloc_obj_t root, unsigned depth)
{
  hwloc_obj_t child, *pchild;

  /* filter I/O children and recurse */
  for_each_io_child_safe(child, root, pchild) {
    enum hwloc_type_filter_e filter = topology->type_filter[child->type];

    /* recurse into grand-children */
    hwloc__filter_bridges(topology, child, depth+1);

    child->attr->bridge.depth = depth;

    /* remove bridges that have no child,
     * and pci-to-non-pci bridges (pcidev) that no child either.
     * keep NVSwitch since they may be used in NVLink matrices.
     */
    if (filter == HWLOC_TYPE_FILTER_KEEP_IMPORTANT
	&& !child->io_first_child
        && (child->type == HWLOC_OBJ_BRIDGE
            || (child->type == HWLOC_OBJ_PCI_DEVICE && (child->attr->pcidev.class_id >> 8) == 0x06
                && (!child->subtype || strcmp(child->subtype, "NVSwitch"))))) {
      unlink_and_free_single_object(pchild);
      topology->modified = 1;
    }
  }
}

static void
hwloc_filter_bridges(hwloc_topology_t topology, hwloc_obj_t parent)
{
  hwloc_obj_t child = parent->first_child;
  while (child) {
    hwloc_filter_bridges(topology, child);
    child = child->next_sibling;
  }

  hwloc__filter_bridges(topology, parent, 0);
}

void
hwloc__reorder_children(hwloc_obj_t parent)
{
  /* move the children list on the side */
  hwloc_obj_t *prev, child, children = parent->first_child;
  parent->first_child = NULL;
  while (children) {
    /* dequeue child */
    child = children;
    children = child->next_sibling;
    /* find where to enqueue it */
    prev = &parent->first_child;
    while (*prev && hwloc__object_cpusets_compare_first(child, *prev) > 0)
      prev = &((*prev)->next_sibling);
    /* enqueue */
    child->next_sibling = *prev;
    *prev = child;
  }
  /* No ordering to enforce for Misc or I/O children. */
}

/* Remove all normal children whose cpuset is empty,
 * and memory children whose nodeset is empty.
 * Also don't remove objects that have I/O children, but ignore Misc.
 */
static void
remove_empty(hwloc_topology_t topology, hwloc_obj_t *pobj)
{
  hwloc_obj_t obj = *pobj, child, *pchild;

  for_each_child_safe(child, obj, pchild)
    remove_empty(topology, pchild);
  for_each_memory_child_safe(child, obj, pchild)
    remove_empty(topology, pchild);
  /* No cpuset under I/O or Misc */

  if (obj->first_child /* only remove if all children were removed above, so that we don't remove parents of NUMAnode */
      || obj->memory_first_child /* only remove if no memory attached there */
      || obj->io_first_child /* only remove if no I/O is attached there */)
    /* ignore Misc */
    return;

  if (hwloc__obj_type_is_normal(obj->type)) {
    if (!hwloc_bitmap_iszero(obj->cpuset))
      return;
  } else {
    assert(hwloc__obj_type_is_memory(obj->type));
    if (!hwloc_bitmap_iszero(obj->nodeset))
      return;
  }

  hwloc_debug("%s", "\nRemoving empty object ");
  hwloc_debug_print_object(0, obj);
  unlink_and_free_single_object(pobj);
  topology->modified = 1;
}

/* reset type depth before modifying levels (either reconnecting or filtering/keep_structure) */
static void
hwloc_reset_normal_type_depths(hwloc_topology_t topology)
{
  unsigned i;
  for (i=HWLOC_OBJ_TYPE_MIN; i<=HWLOC_OBJ_GROUP; i++)
    topology->type_depth[i] = HWLOC_TYPE_DEPTH_UNKNOWN;
  /* type contiguity is asserted in topology_check() */
  topology->type_depth[HWLOC_OBJ_DIE] = HWLOC_TYPE_DEPTH_UNKNOWN;
}

static int
hwloc_dont_merge_group_level(hwloc_topology_t topology, unsigned i)
{
  unsigned j;

  /* Don't merge some groups in that level? */
  for(j=0; j<topology->level_nbobjects[i]; j++)
    if (topology->levels[i][j]->attr->group.dont_merge)
      return 1;

  return 0;
}

/* compare i-th and i-1-th levels structure */
static int
hwloc_compare_levels_structure(hwloc_topology_t topology, unsigned i)
{
  int checkmemory = (topology->levels[i][0]->type == HWLOC_OBJ_PU);
  unsigned j;

  if (topology->level_nbobjects[i-1] != topology->level_nbobjects[i])
    return -1;

  for(j=0; j<topology->level_nbobjects[i]; j++) {
    if (topology->levels[i-1][j] != topology->levels[i][j]->parent)
      return -1;
    if (topology->levels[i-1][j]->arity != 1)
      return -1;
    if (checkmemory && topology->levels[i-1][j]->memory_arity)
      /* don't merge PUs if there's memory above */
      return -1;
  }
  /* same number of objects with arity 1 above, no problem */
  return 0;
}

/* return > 0 if any level was removed.
 * performs its own reconnect internally if needed
 */
static int
hwloc_filter_levels_keep_structure(hwloc_topology_t topology)
{
  unsigned i, j;
  int res = 0;

  if (topology->modified) {
    /* WARNING: hwloc_topology_reconnect() is duplicated partially here
     * and at the end of this function:
     * - we need normal levels before merging.
     * - and we'll need to update special levels after merging.
     */
    hwloc_connect_children(topology->levels[0][0]);
    if (hwloc_connect_levels(topology) < 0)
      return -1;
  }

  /* start from the bottom since we'll remove intermediate levels */
  for(i=topology->nb_levels-1; i>0; i--) {
    int replacechild = 0, replaceparent = 0;
    hwloc_obj_t obj1 = topology->levels[i-1][0];
    hwloc_obj_t obj2 = topology->levels[i][0];
    hwloc_obj_type_t type1 = obj1->type;
    hwloc_obj_type_t type2 = obj2->type;

    /* Check whether parents and/or children can be replaced */
    if (topology->type_filter[type1] == HWLOC_TYPE_FILTER_KEEP_STRUCTURE) {
      /* Parents can be ignored in favor of children.  */
      replaceparent = 1;
      if (type1 == HWLOC_OBJ_GROUP && hwloc_dont_merge_group_level(topology, i-1))
	replaceparent = 0;
    }
    if (topology->type_filter[type2] == HWLOC_TYPE_FILTER_KEEP_STRUCTURE) {
      /* Children can be ignored in favor of parents.  */
      replacechild = 1;
      if (type1 == HWLOC_OBJ_GROUP && hwloc_dont_merge_group_level(topology, i))
	replacechild = 0;
    }
    if (!replacechild && !replaceparent)
      /* no ignoring */
      continue;
    /* Decide which one to actually replace */
    if (replaceparent && replacechild) {
      /* If both may be replaced, look at obj_type_priority */
      if (obj_type_priority[type1] >= obj_type_priority[type2])
	replaceparent = 0;
      else
	replacechild = 0;
    }
    /* Are these levels actually identical? */
    if (hwloc_compare_levels_structure(topology, i) < 0)
      continue;
    hwloc_debug("may merge levels #%u=%s and #%u=%s\n",
		i-1, hwloc_obj_type_string(type1), i, hwloc_obj_type_string(type2));

    /* OK, remove intermediate objects from the tree. */
    for(j=0; j<topology->level_nbobjects[i]; j++) {
      hwloc_obj_t parent = topology->levels[i-1][j];
      hwloc_obj_t child = topology->levels[i][j];
      unsigned k;
      if (replacechild) {
	/* move child's children to parent */
	parent->first_child = child->first_child;
	parent->last_child = child->last_child;
	parent->arity = child->arity;
	free(parent->children);
	parent->children = child->children;
	child->children = NULL;
	/* update children parent */
	for(k=0; k<parent->arity; k++)
	  parent->children[k]->parent = parent;
	/* append child memory/io/misc children to parent */
	if (child->memory_first_child) {
	  append_siblings_list(&parent->memory_first_child, child->memory_first_child, parent);
	  parent->memory_arity += child->memory_arity;
	}
	if (child->io_first_child) {
	  append_siblings_list(&parent->io_first_child, child->io_first_child, parent);
	  parent->io_arity += child->io_arity;
	}
	if (child->misc_first_child) {
	  append_siblings_list(&parent->misc_first_child, child->misc_first_child, parent);
	  parent->misc_arity += child->misc_arity;
	}
	hwloc_free_unlinked_object(child);
      } else {
	/* replace parent with child in grand-parent */
	if (parent->parent) {
	  parent->parent->children[parent->sibling_rank] = child;
	  child->sibling_rank = parent->sibling_rank;
	  if (!parent->sibling_rank) {
	    parent->parent->first_child = child;
	    /* child->prev_sibling was already NULL, child was single */
	  } else {
	    child->prev_sibling = parent->parent->children[parent->sibling_rank-1];
	    child->prev_sibling->next_sibling = child;
	  }
	  if (parent->sibling_rank == parent->parent->arity-1) {
	    parent->parent->last_child = child;
	    /* child->next_sibling was already NULL, child was single */
	  } else {
	    child->next_sibling = parent->parent->children[parent->sibling_rank+1];
	    child->next_sibling->prev_sibling = child;
	  }
	  /* update child parent */
	  child->parent = parent->parent;
	} else {
	  /* make child the new root */
	  topology->levels[0][0] = child;
	  child->parent = NULL;
	}
	/* prepend parent memory/io/misc children to child */
	if (parent->memory_first_child) {
	  prepend_siblings_list(&child->memory_first_child, parent->memory_first_child, child);
	  child->memory_arity += parent->memory_arity;
	}
	if (parent->io_first_child) {
	  prepend_siblings_list(&child->io_first_child, parent->io_first_child, child);
	  child->io_arity += parent->io_arity;
	}
	if (parent->misc_first_child) {
	  prepend_siblings_list(&child->misc_first_child, parent->misc_first_child, child);
	  child->misc_arity += parent->misc_arity;
	}
	hwloc_free_unlinked_object(parent);
	/* prev/next_sibling will be updated below in another loop */
      }
    }
    if (replaceparent && i>1) {
      /* Update sibling list within modified parent->parent arrays */
      for(j=0; j<topology->level_nbobjects[i]; j++) {
	hwloc_obj_t child = topology->levels[i][j];
	unsigned rank = child->sibling_rank;
	child->prev_sibling = rank > 0 ? child->parent->children[rank-1] : NULL;
	child->next_sibling = rank < child->parent->arity-1 ? child->parent->children[rank+1] : NULL;
      }
    }

    /* Update levels so that the next reconnect isn't confused */
    if (replaceparent) {
      /* Removing level i-1, so move levels [i..nb_levels-1] to [i-1..] */
      free(topology->levels[i-1]);
      memmove(&topology->levels[i-1],
	      &topology->levels[i],
	      (topology->nb_levels-i)*sizeof(topology->levels[i]));
      memmove(&topology->level_nbobjects[i-1],
	      &topology->level_nbobjects[i],
	      (topology->nb_levels-i)*sizeof(topology->level_nbobjects[i]));
      hwloc_debug("removed parent level %s at depth %u\n",
		  hwloc_obj_type_string(type1), i-1);
    } else {
      /* Removing level i, so move levels [i+1..nb_levels-1] and later to [i..] */
      free(topology->levels[i]);
      memmove(&topology->levels[i],
	      &topology->levels[i+1],
	      (topology->nb_levels-1-i)*sizeof(topology->levels[i]));
      memmove(&topology->level_nbobjects[i],
	      &topology->level_nbobjects[i+1],
	      (topology->nb_levels-1-i)*sizeof(topology->level_nbobjects[i]));
      hwloc_debug("removed child level %s at depth %u\n",
		  hwloc_obj_type_string(type2), i);
    }
    topology->level_nbobjects[topology->nb_levels-1] = 0;
    topology->levels[topology->nb_levels-1] = NULL;
    topology->nb_levels--;

    res++;
  }

  if (res > 0) {
    /* Update object and type depths if some levels were removed */
    hwloc_reset_normal_type_depths(topology);
    for(i=0; i<topology->nb_levels; i++) {
      hwloc_obj_type_t type = topology->levels[i][0]->type;
      for(j=0; j<topology->level_nbobjects[i]; j++)
	topology->levels[i][j]->depth = (int)i;
      if (topology->type_depth[type] == HWLOC_TYPE_DEPTH_UNKNOWN)
	topology->type_depth[type] = (int)i;
      else
	topology->type_depth[type] = HWLOC_TYPE_DEPTH_MULTIPLE;
    }
  }


  if (res > 0 || topology-> modified) {
    /* WARNING: hwloc_topology_reconnect() is duplicated partially here
     * and at the beginning of this function.
     * If we merged some levels, some child+parent special children lisst
     * may have been merged, hence specials level might need reordering,
     * So reconnect special levels only here at the end
     * (it's not needed at the beginning of this function).
     */
    if (hwloc_connect_special_levels(topology) < 0)
      return -1;
    topology->modified = 0;
  }

  return 0;
}

static void
hwloc_propagate_symmetric_subtree(hwloc_topology_t topology, hwloc_obj_t root)
{
  hwloc_obj_t child;
  unsigned arity = root->arity;
  hwloc_obj_t *array;
  int ok;

  /* assume we're not symmetric by default */
  root->symmetric_subtree = 0;

  /* if no child, we are symmetric */
  if (!arity)
    goto good;

  /* FIXME ignore memory just like I/O and Misc? */

  /* look at normal children only, I/O and Misc are ignored.
   * return if any child is not symmetric.
   */
  ok = 1;
  for_each_child(child, root) {
    hwloc_propagate_symmetric_subtree(topology, child);
    if (!child->symmetric_subtree)
      ok = 0;
  }
  if (!ok)
    return;
  /* Misc and I/O children do not care about symmetric_subtree */

  /* if single child is symmetric, we're good */
  if (arity == 1)
    goto good;

  /* now check that children subtrees are identical.
   * just walk down the first child in each tree and compare their depth and arities
   */
  array = malloc(arity * sizeof(*array));
  if (!array)
    return;
  memcpy(array, root->children, arity * sizeof(*array));
  while (1) {
    unsigned i;
    /* check current level arities and depth */
    for(i=1; i<arity; i++)
      if (array[i]->depth != array[0]->depth
	  || array[i]->arity != array[0]->arity) {
	free(array);
	return;
      }
    if (!array[0]->arity)
      /* no more children level, we're ok */
      break;
    /* look at first child of each element now */
    for(i=0; i<arity; i++)
      array[i] = array[i]->first_child;
  }
  free(array);

  /* everything went fine, we're symmetric */
 good:
  root->symmetric_subtree = 1;
}

static void hwloc_set_group_depth(hwloc_topology_t topology)
{
  unsigned groupdepth = 0;
  unsigned i, j;
  for(i=0; i<topology->nb_levels; i++)
    if (topology->levels[i][0]->type == HWLOC_OBJ_GROUP) {
      for (j = 0; j < topology->level_nbobjects[i]; j++)
	topology->levels[i][j]->attr->group.depth = groupdepth;
      groupdepth++;
    }
}

/*
 * Initialize handy pointers in the whole topology.
 * The topology only had first_child and next_sibling pointers.
 * When this funtions return, all parent/children pointers are initialized.
 * The remaining fields (levels, cousins, logical_index, depth, ...) will
 * be setup later in hwloc_connect_levels().
 *
 * Can be called several times, so may have to update the array.
 */
static void
hwloc_connect_children(hwloc_obj_t parent)
{
  unsigned n, oldn = parent->arity;
  hwloc_obj_t child, prev_child;
  int ok;

  /* Main children list */

  ok = 1;
  prev_child = NULL;
  for (n = 0, child = parent->first_child;
       child;
       n++,   prev_child = child, child = child->next_sibling) {
    child->sibling_rank = n;
    child->prev_sibling = prev_child;
    /* already OK in the array? */
    if (n >= oldn || parent->children[n] != child)
      ok = 0;
    /* recurse */
    hwloc_connect_children(child);
  }
  parent->last_child = prev_child;
  parent->arity = n;
  if (!n) {
    /* no need for an array anymore */
    free(parent->children);
    parent->children = NULL;
    goto memory;
  }
  if (ok)
    /* array is already OK (even if too large) */
    goto memory;

  /* alloc a larger array if needed */
  if (oldn < n) {
    free(parent->children);
    parent->children = malloc(n * sizeof(*parent->children));
  }
  /* refill */
  for (n = 0, child = parent->first_child;
       child;
       n++,   child = child->next_sibling) {
    parent->children[n] = child;
  }



 memory:
  /* Memory children list */

  prev_child = NULL;
  for (n = 0, child = parent->memory_first_child;
       child;
       n++,   prev_child = child, child = child->next_sibling) {
    child->parent = parent;
    child->sibling_rank = n;
    child->prev_sibling = prev_child;
    hwloc_connect_children(child);
  }
  parent->memory_arity = n;

  /* I/O children list */

  prev_child = NULL;
  for (n = 0, child = parent->io_first_child;
       child;
       n++,   prev_child = child, child = child->next_sibling) {
    child->parent = parent;
    child->sibling_rank = n;
    child->prev_sibling = prev_child;
    hwloc_connect_children(child);
  }
  parent->io_arity = n;

  /* Misc children list */

  prev_child = NULL;
  for (n = 0, child = parent->misc_first_child;
       child;
       n++,   prev_child = child, child = child->next_sibling) {
    child->parent = parent;
    child->sibling_rank = n;
    child->prev_sibling = prev_child;
    hwloc_connect_children(child);
  }
  parent->misc_arity = n;
}

/*
 * Check whether there is an object strictly below ROOT that has the same type as OBJ
 */
static int
find_same_type(hwloc_obj_t root, hwloc_obj_t obj)
{
  hwloc_obj_t child;

  for_each_child (child, root) {
    if (hwloc_type_cmp(child, obj) == HWLOC_OBJ_EQUAL)
      return 1;
    if (find_same_type(child, obj))
      return 1;
  }

  return 0;
}

static int
hwloc_build_level_from_list(struct hwloc_special_level_s *slevel)
{
  unsigned i, nb;
  struct hwloc_obj * obj;

  /* count */
  obj = slevel->first;
  i = 0;
  while (obj) {
    i++;
    obj = obj->next_cousin;
  }
  nb = i;

  if (nb) {
    /* allocate and fill level */
    slevel->objs = malloc(nb * sizeof(struct hwloc_obj *));
    if (!slevel->objs)
      return -1;

    obj = slevel->first;
    i = 0;
    while (obj) {
      obj->logical_index = i;
      slevel->objs[i] = obj;
      i++;
      obj = obj->next_cousin;
    }
  }

  slevel->nbobjs = nb;
  return 0;
}

static void
hwloc_append_special_object(struct hwloc_special_level_s *level, hwloc_obj_t obj)
{
  if (level->first) {
    obj->prev_cousin = level->last;
    obj->prev_cousin->next_cousin = obj;
    level->last = obj;
  } else {
    obj->prev_cousin = NULL;
    level->first = level->last = obj;
  }
}

/* Append special objects to their lists */
static void
hwloc_list_special_objects(hwloc_topology_t topology, hwloc_obj_t obj)
{
  hwloc_obj_t child;

  if (obj->type == HWLOC_OBJ_NUMANODE) {
    obj->next_cousin = NULL;
    obj->depth = HWLOC_TYPE_DEPTH_NUMANODE;
    /* Insert the main NUMA node list */
    hwloc_append_special_object(&topology->slevels[HWLOC_SLEVEL_NUMANODE], obj);

    /* Recurse, NUMA nodes only have Misc children */
    for_each_misc_child(child, obj)
      hwloc_list_special_objects(topology, child);

  } else if (obj->type == HWLOC_OBJ_MEMCACHE) {
    obj->next_cousin = NULL;
    obj->depth = HWLOC_TYPE_DEPTH_MEMCACHE;
    /* Insert the main MemCache list */
    hwloc_append_special_object(&topology->slevels[HWLOC_SLEVEL_MEMCACHE], obj);

    /* Recurse, MemCaches have NUMA nodes or Misc children */
    for_each_memory_child(child, obj)
      hwloc_list_special_objects(topology, child);
    for_each_misc_child(child, obj)
      hwloc_list_special_objects(topology, child);

  } else if (obj->type == HWLOC_OBJ_MISC) {
    obj->next_cousin = NULL;
    obj->depth = HWLOC_TYPE_DEPTH_MISC;
    /* Insert the main Misc list */
    hwloc_append_special_object(&topology->slevels[HWLOC_SLEVEL_MISC], obj);
    /* Recurse, Misc only have Misc children */
    for_each_misc_child(child, obj)
      hwloc_list_special_objects(topology, child);

  } else if (hwloc__obj_type_is_io(obj->type)) {
    obj->next_cousin = NULL;

    if (obj->type == HWLOC_OBJ_BRIDGE) {
      obj->depth = HWLOC_TYPE_DEPTH_BRIDGE;
      /* Insert in the main bridge list */
      hwloc_append_special_object(&topology->slevels[HWLOC_SLEVEL_BRIDGE], obj);

    } else if (obj->type == HWLOC_OBJ_PCI_DEVICE) {
      obj->depth = HWLOC_TYPE_DEPTH_PCI_DEVICE;
      /* Insert in the main pcidev list */
      hwloc_append_special_object(&topology->slevels[HWLOC_SLEVEL_PCIDEV], obj);

    } else if (obj->type == HWLOC_OBJ_OS_DEVICE) {
      obj->depth = HWLOC_TYPE_DEPTH_OS_DEVICE;
      /* Insert in the main osdev list */
      hwloc_append_special_object(&topology->slevels[HWLOC_SLEVEL_OSDEV], obj);
    }

    /* Recurse, I/O only have I/O and Misc children */
    for_each_io_child(child, obj)
      hwloc_list_special_objects(topology, child);
    for_each_misc_child(child, obj)
      hwloc_list_special_objects(topology, child);

  } else {
    /* Recurse */
    for_each_child(child, obj)
      hwloc_list_special_objects(topology, child);
    for_each_memory_child(child, obj)
      hwloc_list_special_objects(topology, child);
    for_each_io_child(child, obj)
      hwloc_list_special_objects(topology, child);
    for_each_misc_child(child, obj)
      hwloc_list_special_objects(topology, child);
  }
}

/* Build Memory, I/O and Misc levels */
static int
hwloc_connect_special_levels(hwloc_topology_t topology)
{
  unsigned i;

  for(i=0; i<HWLOC_NR_SLEVELS; i++)
    free(topology->slevels[i].objs);
  memset(&topology->slevels, 0, sizeof(topology->slevels));

  hwloc_list_special_objects(topology, topology->levels[0][0]);

  for(i=0; i<HWLOC_NR_SLEVELS; i++) {
    if (hwloc_build_level_from_list(&topology->slevels[i]) < 0)
      return -1;
  }

  return 0;
}

/*
 * Do the remaining work that hwloc_connect_children() did not do earlier.
 * Requires object arity and children list to be properly initialized (by hwloc_connect_children()).
 */
static int
hwloc_connect_levels(hwloc_topology_t topology)
{
  unsigned l, i=0;
  hwloc_obj_t *objs, *taken_objs, *new_objs, top_obj, root;
  unsigned n_objs, n_taken_objs, n_new_objs;

  /* reset non-root levels (root was initialized during init and will not change here) */
  for(l=1; l<topology->nb_levels; l++)
    free(topology->levels[l]);
  memset(topology->levels+1, 0, (topology->nb_levels-1)*sizeof(*topology->levels));
  memset(topology->level_nbobjects+1, 0, (topology->nb_levels-1)*sizeof(*topology->level_nbobjects));
  topology->nb_levels = 1;

  /* initialize all non-IO/non-Misc depths to unknown */
  hwloc_reset_normal_type_depths(topology);

  /* initialize root type depth */
  root = topology->levels[0][0];
  root->depth = 0;
  topology->type_depth[root->type] = 0;
  /* root level */
  root->logical_index = 0;
  root->prev_cousin = NULL;
  root->next_cousin = NULL;
  /* root as a child of nothing */
  root->parent = NULL;
  root->sibling_rank = 0;
  root->prev_sibling = NULL;
  root->next_sibling = NULL;

  /* Start with children of the whole system.  */
  n_objs = topology->levels[0][0]->arity;
  objs = malloc(n_objs * sizeof(objs[0]));
  if (!objs) {
    errno = ENOMEM;
    return -1;
  }
  memcpy(objs, topology->levels[0][0]->children, n_objs*sizeof(objs[0]));

  /* Keep building levels while there are objects left in OBJS.  */
  while (n_objs) {
    /* At this point, the objs array contains only objects that may go into levels */

    /* First find which type of object is the topmost.
     * Don't use PU if there are other types since we want to keep PU at the bottom.
     */

    /* Look for the first non-PU object, and use the first PU if we really find nothing else */
    for (i = 0; i < n_objs; i++)
      if (objs[i]->type != HWLOC_OBJ_PU)
        break;
    top_obj = i == n_objs ? objs[0] : objs[i];

    /* See if this is actually the topmost object */
    for (i = 0; i < n_objs; i++) {
      if (hwloc_type_cmp(top_obj, objs[i]) != HWLOC_OBJ_EQUAL) {
	if (find_same_type(objs[i], top_obj)) {
	  /* OBJS[i] is strictly above an object of the same type as TOP_OBJ, so it
	   * is above TOP_OBJ.  */
	  top_obj = objs[i];
	}
      }
    }

    /* Now peek all objects of the same type, build a level with that and
     * replace them with their children.  */

    /* allocate enough to take all current objects and an ending NULL */
    taken_objs = malloc((n_objs+1) * sizeof(taken_objs[0]));
    if (!taken_objs) {
      free(objs);
      errno = ENOMEM;
      return -1;
    }

    /* allocate enough to keep all current objects or their children */
    n_new_objs = 0;
    for (i = 0; i < n_objs; i++) {
      if (objs[i]->arity)
	n_new_objs += objs[i]->arity;
      else
	n_new_objs++;
    }
    new_objs = malloc(n_new_objs * sizeof(new_objs[0]));
    if (!new_objs) {
      free(objs);
      free(taken_objs);
      errno = ENOMEM;
      return -1;
    }

    /* now actually take these objects */
    n_new_objs = 0;
    n_taken_objs = 0;
    for (i = 0; i < n_objs; i++)
      if (hwloc_type_cmp(top_obj, objs[i]) == HWLOC_OBJ_EQUAL) {
	/* Take it, add main children.  */
	taken_objs[n_taken_objs++] = objs[i];
	if (objs[i]->arity)
	  memcpy(&new_objs[n_new_objs], objs[i]->children, objs[i]->arity * sizeof(new_objs[0]));
	n_new_objs += objs[i]->arity;
      } else {
	/* Leave it.  */
	new_objs[n_new_objs++] = objs[i];
      }

    if (!n_new_objs) {
      free(new_objs);
      new_objs = NULL;
    }

    /* Ok, put numbers in the level and link cousins.  */
    for (i = 0; i < n_taken_objs; i++) {
      taken_objs[i]->depth = (int) topology->nb_levels;
      taken_objs[i]->logical_index = i;
      if (i) {
	taken_objs[i]->prev_cousin = taken_objs[i-1];
	taken_objs[i-1]->next_cousin = taken_objs[i];
      }
    }
    taken_objs[0]->prev_cousin = NULL;
    taken_objs[n_taken_objs-1]->next_cousin = NULL;

    /* One more level!  */
    hwloc_debug("--- %s level", hwloc_obj_type_string(top_obj->type));
    hwloc_debug(" has number %u\n\n", topology->nb_levels);

    if (topology->type_depth[top_obj->type] == HWLOC_TYPE_DEPTH_UNKNOWN)
      topology->type_depth[top_obj->type] = (int) topology->nb_levels;
    else
      topology->type_depth[top_obj->type] = HWLOC_TYPE_DEPTH_MULTIPLE; /* mark as unknown */

    taken_objs[n_taken_objs] = NULL;

    if (topology->nb_levels == topology->nb_levels_allocated) {
      /* extend the arrays of levels */
      void *tmplevels, *tmpnbobjs;
      tmplevels = realloc(topology->levels,
			  2 * topology->nb_levels_allocated * sizeof(*topology->levels));
      tmpnbobjs = realloc(topology->level_nbobjects,
			  2 * topology->nb_levels_allocated * sizeof(*topology->level_nbobjects));
      if (!tmplevels || !tmpnbobjs) {
        if (HWLOC_SHOW_CRITICAL_ERRORS())
          fprintf(stderr, "hwloc: failed to realloc level arrays to %u\n", topology->nb_levels_allocated * 2);

	/* if one realloc succeeded, make sure the caller will free the new buffer */
	if (tmplevels)
	  topology->levels = tmplevels;
	if (tmpnbobjs)
	  topology->level_nbobjects = tmpnbobjs;
	/* the realloc that failed left topology->level_foo untouched, will be freed by the caller */

	free(objs);
	free(taken_objs);
	free(new_objs);
	errno = ENOMEM;
	return -1;
      }
      topology->levels = tmplevels;
      topology->level_nbobjects = tmpnbobjs;
      memset(topology->levels + topology->nb_levels_allocated,
	     0, topology->nb_levels_allocated * sizeof(*topology->levels));
      memset(topology->level_nbobjects + topology->nb_levels_allocated,
	     0, topology->nb_levels_allocated * sizeof(*topology->level_nbobjects));
      topology->nb_levels_allocated *= 2;
    }
    /* add the new level */
    topology->level_nbobjects[topology->nb_levels] = n_taken_objs;
    topology->levels[topology->nb_levels] = taken_objs;

    topology->nb_levels++;

    free(objs);

    /* Switch to new_objs */
    objs = new_objs;
    n_objs = n_new_objs;
  }

  /* It's empty now.  */
  free(objs);

  return 0;
}

int
hwloc_topology_reconnect(struct hwloc_topology *topology, unsigned long flags)
{
  /* WARNING: when updating this function, the replicated code must
   * also be updated inside hwloc_filter_levels_keep_structure()
   */

  if (flags) {
    errno = EINVAL;
    return -1;
  }
  if (!topology->modified)
    return 0;

  hwloc_connect_children(topology->levels[0][0]);

  if (hwloc_connect_levels(topology) < 0)
    return -1;

  if (hwloc_connect_special_levels(topology) < 0)
    return -1;

  topology->modified = 0;

  return 0;
}

/* for regression testing, make sure the order of io devices
 * doesn't change with the dentry order in the filesystem
 *
 * Only needed for OSDev for now.
 */
static hwloc_obj_t
hwloc_debug_insert_osdev_sorted(hwloc_obj_t queue, hwloc_obj_t obj)
{
  hwloc_obj_t *pcur = &queue;
  while (*pcur && strcmp((*pcur)->name, obj->name) < 0)
    pcur = &((*pcur)->next_sibling);
  obj->next_sibling = *pcur;
  *pcur = obj;
  return queue;
}

static void
hwloc_debug_sort_children(hwloc_obj_t root)
{
  hwloc_obj_t child;

  if (root->io_first_child) {
    hwloc_obj_t osdevqueue, *pchild;

    pchild = &root->io_first_child;
    osdevqueue = NULL;
    while ((child = *pchild) != NULL) {
      if (child->type != HWLOC_OBJ_OS_DEVICE) {
	/* keep non-osdev untouched */
	pchild = &child->next_sibling;
	continue;
      }

      /* dequeue this child */
      *pchild = child->next_sibling;
      child->next_sibling = NULL;

      /* insert in osdev queue in order */
      osdevqueue = hwloc_debug_insert_osdev_sorted(osdevqueue, child);
    }

    /* requeue the now-sorted osdev queue */
    *pchild = osdevqueue;
  }

  /* Recurse */
  for_each_child(child, root)
    hwloc_debug_sort_children(child);
  for_each_memory_child(child, root)
    hwloc_debug_sort_children(child);
  for_each_io_child(child, root)
    hwloc_debug_sort_children(child);
  /* no I/O under Misc */
}

void hwloc_alloc_root_sets(hwloc_obj_t root)
{
  /*
   * All sets are initially NULL.
   *
   * At least one backend should call this function to initialize all sets at once.
   * XML uses it lazily in case only some sets were given in the XML import.
   *
   * Other backends can check root->cpuset != NULL to see if somebody
   * discovered things before them.
   */
  if (!root->cpuset)
     root->cpuset = hwloc_bitmap_alloc();
  if (!root->complete_cpuset)
     root->complete_cpuset = hwloc_bitmap_alloc();
  if (!root->nodeset)
    root->nodeset = hwloc_bitmap_alloc();
  if (!root->complete_nodeset)
    root->complete_nodeset = hwloc_bitmap_alloc();
}

static void
hwloc_discover_by_phase(struct hwloc_topology *topology,
			struct hwloc_disc_status *dstatus,
			const char *phasename __hwloc_attribute_unused)
{
  struct hwloc_backend *backend;
  hwloc_debug("%s phase discovery...\n", phasename);
  for(backend = topology->backends; backend; backend = backend->next) {
    if (dstatus->phase & dstatus->excluded_phases)
      break;
    if (!(backend->phases & dstatus->phase))
      continue;
    if (!backend->discover)
      continue;
    hwloc_debug("%s phase discovery in component %s...\n", phasename, backend->component->name);
    backend->discover(backend, dstatus);
    hwloc_debug_print_objects(0, topology->levels[0][0]);
  }
}

/* Main discovery loop */
static int
hwloc_discover(struct hwloc_topology *topology,
	       struct hwloc_disc_status *dstatus)
{
  const char *env;

  topology->modified = 0; /* no need to reconnect yet */

  topology->allowed_cpuset = hwloc_bitmap_alloc_full();
  topology->allowed_nodeset = hwloc_bitmap_alloc_full();

  /* discover() callbacks should use hwloc_insert to add objects initialized
   * through hwloc_alloc_setup_object.
   * For node levels, nodeset and memory must be initialized.
   * For cache levels, memory and type/depth must be initialized.
   * For group levels, depth must be initialized.
   */

  /* There must be at least a PU object for each logical processor, at worse
   * produced by hwloc_setup_pu_level()
   */

  /* To be able to just use hwloc__insert_object_by_cpuset to insert the object
   * in the topology according to the cpuset, the cpuset field must be
   * initialized.
   */

  /* A priori, All processors are visible in the topology, and allowed
   * for the application.
   *
   * - If some processors exist but topology information is unknown for them
   *   (and thus the backend couldn't create objects for them), they should be
   *   added to the complete_cpuset field of the lowest object where the object
   *   could reside.
   *
   * - If some processors are not allowed for the application (e.g. for
   *   administration reasons), they should be dropped from the allowed_cpuset
   *   field.
   *
   * The same applies to the node sets complete_nodeset and allowed_cpuset.
   *
   * If such field doesn't exist yet, it can be allocated, and initialized to
   * zero (for complete), or to full (for allowed). The values are
   * automatically propagated to the whole tree after detection.
   */

  if (topology->backend_phases & HWLOC_DISC_PHASE_GLOBAL) {
    /* usually, GLOBAL is alone.
     * but HWLOC_ANNOTATE_GLOBAL_COMPONENTS=1 allows optional ANNOTATE steps.
     */
    struct hwloc_backend *global_backend = topology->backends;
    assert(global_backend);
    assert(global_backend->phases == HWLOC_DISC_PHASE_GLOBAL);

    /*
     * Perform the single-component-based GLOBAL discovery
     */
    hwloc_debug("GLOBAL phase discovery...\n");
    hwloc_debug("GLOBAL phase discovery with component %s...\n", global_backend->component->name);
    dstatus->phase = HWLOC_DISC_PHASE_GLOBAL;
    global_backend->discover(global_backend, dstatus);
    hwloc_debug_print_objects(0, topology->levels[0][0]);
  }
  /* Don't explicitly ignore other phases, in case there's ever
   * a need to bring them back.
   * The component with usually exclude them by default anyway.
   * Except if annotating global components is explicitly requested.
   */

  if (topology->backend_phases & HWLOC_DISC_PHASE_CPU) {
    /*
     * Discover CPUs first
     */
    dstatus->phase = HWLOC_DISC_PHASE_CPU;
    hwloc_discover_by_phase(topology, dstatus, "CPU");
  }

  if (!(topology->backend_phases & (HWLOC_DISC_PHASE_GLOBAL|HWLOC_DISC_PHASE_CPU))) {
    hwloc_debug("No GLOBAL or CPU component phase found\n");
    /* we'll fail below */
  }

  /* One backend should have called hwloc_alloc_root_sets()
   * and set bits during PU and NUMA insert.
   */
  if (!topology->levels[0][0]->cpuset || hwloc_bitmap_iszero(topology->levels[0][0]->cpuset)) {
    hwloc_debug("%s", "No PU added by any CPU or GLOBAL component phase\n");
    errno = EINVAL;
    return -1;
  }

  /*
   * Memory-specific discovery
   */
  if (topology->backend_phases & HWLOC_DISC_PHASE_MEMORY) {
    dstatus->phase = HWLOC_DISC_PHASE_MEMORY;
    hwloc_discover_by_phase(topology, dstatus, "MEMORY");
  }

  if (/* check if getting the sets of locally allowed resources is possible */
      topology->binding_hooks.get_allowed_resources
      && topology->is_thissystem
      /* check whether it has been done already */
      && !(dstatus->flags & HWLOC_DISC_STATUS_FLAG_GOT_ALLOWED_RESOURCES)
      /* check whether it was explicitly requested */
      && ((topology->flags & HWLOC_TOPOLOGY_FLAG_THISSYSTEM_ALLOWED_RESOURCES) != 0
	  || ((env = getenv("HWLOC_THISSYSTEM_ALLOWED_RESOURCES")) != NULL && atoi(env)))) {
    /* OK, get the sets of locally allowed resources */
    topology->binding_hooks.get_allowed_resources(topology);
    dstatus->flags |= HWLOC_DISC_STATUS_FLAG_GOT_ALLOWED_RESOURCES;
  }

  /* If there's no NUMA node, add one with all the memory.
   * root->complete_nodeset wouldn't be empty if any NUMA was ever added:
   * - insert_by_cpuset() adds bits whe PU/NUMA are added.
   * - XML takes care of sanitizing nodesets.
   */
  if (hwloc_bitmap_iszero(topology->levels[0][0]->complete_nodeset)) {
    hwloc_obj_t node;
    hwloc_debug("%s", "\nAdd missing single NUMA node\n");
    node = hwloc_alloc_setup_object(topology, HWLOC_OBJ_NUMANODE, 0);
    node->cpuset = hwloc_bitmap_dup(topology->levels[0][0]->cpuset);
    node->nodeset = hwloc_bitmap_alloc();
    /* other nodesets will be filled below */
    hwloc_bitmap_set(node->nodeset, 0);
    memcpy(&node->attr->numanode, &topology->machine_memory, sizeof(topology->machine_memory));
    memset(&topology->machine_memory, 0, sizeof(topology->machine_memory));
    hwloc__insert_object_by_cpuset(topology, NULL, node, "core:defaultnumanode");
  } else {
    /* if we're sure we found all NUMA nodes without their sizes (x86 backend?),
     * we could split topology->total_memory in all of them.
     */
    free(topology->machine_memory.page_types);
    memset(&topology->machine_memory, 0, sizeof(topology->machine_memory));
  }

  hwloc_debug("%s", "\nFixup root sets\n");
  hwloc_bitmap_and(topology->levels[0][0]->cpuset, topology->levels[0][0]->cpuset, topology->levels[0][0]->complete_cpuset);
  hwloc_bitmap_and(topology->levels[0][0]->nodeset, topology->levels[0][0]->nodeset, topology->levels[0][0]->complete_nodeset);

  hwloc_bitmap_and(topology->allowed_cpuset, topology->allowed_cpuset, topology->levels[0][0]->cpuset);
  hwloc_bitmap_and(topology->allowed_nodeset, topology->allowed_nodeset, topology->levels[0][0]->nodeset);

  hwloc_debug("%s", "\nPropagate sets\n");
  /* cpuset are already there thanks to the _by_cpuset insertion,
   * but nodeset have to be propagated below and above NUMA nodes
   */
  propagate_nodeset(topology->levels[0][0]);
  /* now fixup parent/children sets */
  fixup_sets(topology->levels[0][0]);

  hwloc_debug_print_objects(0, topology->levels[0][0]);

  if (!(topology->flags & HWLOC_TOPOLOGY_FLAG_INCLUDE_DISALLOWED)) {
    hwloc_debug("%s", "\nRemoving unauthorized sets from all sets\n");
    remove_unused_sets(topology, topology->levels[0][0]);
    hwloc_debug_print_objects(0, topology->levels[0][0]);
  }

  /* see if we should ignore the root now that we know how many children it has */
  if (!hwloc_filter_check_keep_object(topology, topology->levels[0][0])
      && topology->levels[0][0]->first_child && !topology->levels[0][0]->first_child->next_sibling) {
    hwloc_obj_t oldroot = topology->levels[0][0];
    hwloc_obj_t newroot = oldroot->first_child;
    /* switch to the new root */
    newroot->parent = NULL;
    topology->levels[0][0] = newroot;
    /* move oldroot memory/io/misc children before newroot children */
    if (oldroot->memory_first_child)
      prepend_siblings_list(&newroot->memory_first_child, oldroot->memory_first_child, newroot);
    if (oldroot->io_first_child)
      prepend_siblings_list(&newroot->io_first_child, oldroot->io_first_child, newroot);
    if (oldroot->misc_first_child)
      prepend_siblings_list(&newroot->misc_first_child, oldroot->misc_first_child, newroot);
    /* destroy oldroot and use the new one */
    hwloc_free_unlinked_object(oldroot);
  }

  /*
   * All object cpusets and nodesets are properly set now.
   */

  /* Now connect handy pointers to make remaining discovery easier. */
  hwloc_debug("%s", "\nOk, finished tweaking, now connect\n");
  if (hwloc_topology_reconnect(topology, 0) < 0)
    return -1;
  hwloc_debug_print_objects(0, topology->levels[0][0]);

  /*
   * Additional discovery
   */
  hwloc_pci_discovery_prepare(topology);

  if (topology->backend_phases & HWLOC_DISC_PHASE_PCI) {
    dstatus->phase = HWLOC_DISC_PHASE_PCI;
    hwloc_discover_by_phase(topology, dstatus, "PCI");
  }
  if (topology->backend_phases & HWLOC_DISC_PHASE_IO) {
    dstatus->phase = HWLOC_DISC_PHASE_IO;
    hwloc_discover_by_phase(topology, dstatus, "IO");
  }
  if (topology->backend_phases & HWLOC_DISC_PHASE_MISC) {
    dstatus->phase = HWLOC_DISC_PHASE_MISC;
    hwloc_discover_by_phase(topology, dstatus, "MISC");
  }
  if (topology->backend_phases & HWLOC_DISC_PHASE_ANNOTATE) {
    dstatus->phase = HWLOC_DISC_PHASE_ANNOTATE;
    hwloc_discover_by_phase(topology, dstatus, "ANNOTATE");
  }

  hwloc_pci_discovery_exit(topology); /* pci needed up to annotate */

  if (getenv("HWLOC_DEBUG_SORT_CHILDREN"))
    hwloc_debug_sort_children(topology->levels[0][0]);

  /* Remove some stuff */

  hwloc_debug("%s", "\nRemoving bridge objects if needed\n");
  hwloc_filter_bridges(topology, topology->levels[0][0]);
  hwloc_debug_print_objects(0, topology->levels[0][0]);

  hwloc_debug("%s", "\nRemoving empty objects\n");
  remove_empty(topology, &topology->levels[0][0]);
  if (!topology->levels[0][0]) {
    if (HWLOC_SHOW_CRITICAL_ERRORS())
      fprintf(stderr, "hwloc: Topology became empty, aborting!\n");
    return -1;
  }
  if (hwloc_bitmap_iszero(topology->levels[0][0]->cpuset)) {
    if (HWLOC_SHOW_CRITICAL_ERRORS())
      fprintf(stderr, "hwloc: Topology does not contain any PU, aborting!\n");
    return -1;
  }
  if (hwloc_bitmap_iszero(topology->levels[0][0]->nodeset)) {
    if (HWLOC_SHOW_CRITICAL_ERRORS())
      fprintf(stderr, "hwloc: Topology does not contain any NUMA node, aborting!\n");
    return -1;
  }
  hwloc_debug_print_objects(0, topology->levels[0][0]);

  hwloc_debug("%s", "\nRemoving levels with HWLOC_TYPE_FILTER_KEEP_STRUCTURE\n");
  if (hwloc_filter_levels_keep_structure(topology) < 0)
    return -1;
  /* takes care of reconnecting children/levels internally,
   * because it needs normal levels.
   * and it's often needed below because of Groups inserted for I/Os anyway */
  hwloc_debug_print_objects(0, topology->levels[0][0]);

  /* accumulate children memory in total_memory fields (only once parent is set) */
  hwloc_debug("%s", "\nPropagate total memory up\n");
  propagate_total_memory(topology->levels[0][0]);

  /* setup the symmetric_subtree attribute */
  hwloc_propagate_symmetric_subtree(topology, topology->levels[0][0]);

  /* apply group depths */
  hwloc_set_group_depth(topology);

  /* add some identification attributes if not loading from XML */
  if (topology->backends
      && strcmp(topology->backends->component->name, "xml")
      && !getenv("HWLOC_DONT_ADD_VERSION_INFO")) {
    char *value;
    /* add a hwlocVersion */
    hwloc_obj_add_info(topology->levels[0][0], "hwlocVersion", HWLOC_VERSION);
    /* add a ProcessName */
    value = hwloc_progname(topology);
    if (value) {
      hwloc_obj_add_info(topology->levels[0][0], "ProcessName", value);
      free(value);
    }
  }

  return 0;
}

/* To be called before discovery is actually launched,
 * Resets everything in case a previous load initialized some stuff.
 */
void
hwloc_topology_setup_defaults(struct hwloc_topology *topology)
{
  struct hwloc_obj *root_obj;

  /* reset support */
  memset(&topology->binding_hooks, 0, sizeof(topology->binding_hooks));
  memset(topology->support.discovery, 0, sizeof(*topology->support.discovery));
  memset(topology->support.cpubind, 0, sizeof(*topology->support.cpubind));
  memset(topology->support.membind, 0, sizeof(*topology->support.membind));
  memset(topology->support.misc, 0, sizeof(*topology->support.misc));

  /* Only the System object on top by default */
  topology->next_gp_index = 1; /* keep 0 as an invalid value */
  topology->nb_levels = 1; /* there's at least SYSTEM */
  topology->levels[0] = hwloc_tma_malloc (topology->tma, sizeof (hwloc_obj_t));
  topology->level_nbobjects[0] = 1;

  /* Machine-wide memory */
  topology->machine_memory.local_memory = 0;
  topology->machine_memory.page_types_len = 0;
  topology->machine_memory.page_types = NULL;

  /* Allowed stuff */
  topology->allowed_cpuset = NULL;
  topology->allowed_nodeset = NULL;

  /* NULLify other special levels */
  memset(&topology->slevels, 0, sizeof(topology->slevels));
  /* assert the indexes of special levels */
  HWLOC_BUILD_ASSERT(HWLOC_SLEVEL_NUMANODE == HWLOC_SLEVEL_FROM_DEPTH(HWLOC_TYPE_DEPTH_NUMANODE));
  HWLOC_BUILD_ASSERT(HWLOC_SLEVEL_MISC == HWLOC_SLEVEL_FROM_DEPTH(HWLOC_TYPE_DEPTH_MISC));
  HWLOC_BUILD_ASSERT(HWLOC_SLEVEL_BRIDGE == HWLOC_SLEVEL_FROM_DEPTH(HWLOC_TYPE_DEPTH_BRIDGE));
  HWLOC_BUILD_ASSERT(HWLOC_SLEVEL_PCIDEV == HWLOC_SLEVEL_FROM_DEPTH(HWLOC_TYPE_DEPTH_PCI_DEVICE));
  HWLOC_BUILD_ASSERT(HWLOC_SLEVEL_OSDEV == HWLOC_SLEVEL_FROM_DEPTH(HWLOC_TYPE_DEPTH_OS_DEVICE));
  HWLOC_BUILD_ASSERT(HWLOC_SLEVEL_MEMCACHE == HWLOC_SLEVEL_FROM_DEPTH(HWLOC_TYPE_DEPTH_MEMCACHE));

  /* sane values to type_depth */
  hwloc_reset_normal_type_depths(topology);
  topology->type_depth[HWLOC_OBJ_NUMANODE] = HWLOC_TYPE_DEPTH_NUMANODE;
  topology->type_depth[HWLOC_OBJ_MISC] = HWLOC_TYPE_DEPTH_MISC;
  topology->type_depth[HWLOC_OBJ_BRIDGE] = HWLOC_TYPE_DEPTH_BRIDGE;
  topology->type_depth[HWLOC_OBJ_PCI_DEVICE] = HWLOC_TYPE_DEPTH_PCI_DEVICE;
  topology->type_depth[HWLOC_OBJ_OS_DEVICE] = HWLOC_TYPE_DEPTH_OS_DEVICE;
  topology->type_depth[HWLOC_OBJ_MEMCACHE] = HWLOC_TYPE_DEPTH_MEMCACHE;

  /* Create the actual machine object, but don't touch its attributes yet
   * since the OS backend may still change the object into something else
   * (for instance System)
   */
  root_obj = hwloc_alloc_setup_object(topology, HWLOC_OBJ_MACHINE, 0);
  topology->levels[0][0] = root_obj;
}

static void hwloc__topology_filter_init(struct hwloc_topology *topology);

/* This function may use a tma, it cannot free() or realloc() */
static int
hwloc__topology_init (struct hwloc_topology **topologyp,
		      unsigned nblevels,
		      struct hwloc_tma *tma)
{
  struct hwloc_topology *topology;

  topology = hwloc_tma_malloc (tma, sizeof (struct hwloc_topology));
  if(!topology)
    return -1;

  topology->tma = tma;

  hwloc_components_init(); /* uses malloc without tma, but won't need it since dup() caller already took a reference */
  hwloc_topology_components_init(topology);
  hwloc_pci_discovery_init(topology); /* make sure both dup() and load() get sane variables */

  /* Setup topology context */
  topology->is_loaded = 0;
  topology->flags = 0;
  topology->is_thissystem = 1;
  topology->pid = 0;
  topology->userdata = NULL;
  topology->topology_abi = HWLOC_TOPOLOGY_ABI;
  topology->adopted_shmem_addr = NULL;
  topology->adopted_shmem_length = 0;

  topology->support.discovery = hwloc_tma_malloc(tma, sizeof(*topology->support.discovery));
  topology->support.cpubind = hwloc_tma_malloc(tma, sizeof(*topology->support.cpubind));
  topology->support.membind = hwloc_tma_malloc(tma, sizeof(*topology->support.membind));
  topology->support.misc = hwloc_tma_malloc(tma, sizeof(*topology->support.misc));

  topology->nb_levels_allocated = nblevels; /* enough for default 10 levels = Mach+Pack+Die+NUMA+L3+L2+L1d+L1i+Co+PU */
  topology->levels = hwloc_tma_calloc(tma, topology->nb_levels_allocated * sizeof(*topology->levels));
  topology->level_nbobjects = hwloc_tma_calloc(tma, topology->nb_levels_allocated * sizeof(*topology->level_nbobjects));

  hwloc__topology_filter_init(topology);

  /* always initialize since we don't know flags to disable those yet */
  hwloc_internal_distances_init(topology);
  hwloc_internal_memattrs_init(topology);
  hwloc_internal_cpukinds_init(topology);

  topology->userdata_export_cb = NULL;
  topology->userdata_import_cb = NULL;
  topology->userdata_not_decoded = 0;

  /* Make the topology look like something coherent but empty */
  hwloc_topology_setup_defaults(topology);

  *topologyp = topology;
  return 0;
}

int
hwloc_topology_init (struct hwloc_topology **topologyp)
{
  return hwloc__topology_init(topologyp,
			      16, /* 16 is enough for default 10 levels = Mach+Pack+Die+NUMA+L3+L2+L1d+L1i+Co+PU */
			      NULL); /* no TMA for normal topologies, too many allocations to fix */
}

int
hwloc_topology_set_pid(struct hwloc_topology *topology __hwloc_attribute_unused,
                       hwloc_pid_t pid __hwloc_attribute_unused)
{
  if (topology->is_loaded) {
    errno = EBUSY;
    return -1;
  }

  /* this does *not* change the backend */
#ifdef HWLOC_LINUX_SYS
  topology->pid = pid;
  return 0;
#else /* HWLOC_LINUX_SYS */
  errno = ENOSYS;
  return -1;
#endif /* HWLOC_LINUX_SYS */
}

int
hwloc_topology_set_synthetic(struct hwloc_topology *topology, const char *description)
{
  if (topology->is_loaded) {
    errno = EBUSY;
    return -1;
  }

  return hwloc_disc_component_force_enable(topology,
					   0 /* api */,
					   "synthetic",
					   description, NULL, NULL);
}

int
hwloc_topology_set_xml(struct hwloc_topology *topology,
		       const char *xmlpath)
{
  if (topology->is_loaded) {
    errno = EBUSY;
    return -1;
  }

  return hwloc_disc_component_force_enable(topology,
					   0 /* api */,
					   "xml",
					   xmlpath, NULL, NULL);
}

int
hwloc_topology_set_xmlbuffer(struct hwloc_topology *topology,
                             const char *xmlbuffer,
                             int size)
{
  if (topology->is_loaded) {
    errno = EBUSY;
    return -1;
  }

  return hwloc_disc_component_force_enable(topology,
					   0 /* api */,
					   "xml", NULL,
					   xmlbuffer, (void*) (uintptr_t) size);
}

int
hwloc_topology_set_flags (struct hwloc_topology *topology, unsigned long flags)
{
  if (topology->is_loaded) {
    /* actually harmless */
    errno = EBUSY;
    return -1;
  }

  if (flags & ~(HWLOC_TOPOLOGY_FLAG_INCLUDE_DISALLOWED
                |HWLOC_TOPOLOGY_FLAG_IS_THISSYSTEM
                |HWLOC_TOPOLOGY_FLAG_THISSYSTEM_ALLOWED_RESOURCES
                |HWLOC_TOPOLOGY_FLAG_IMPORT_SUPPORT
                |HWLOC_TOPOLOGY_FLAG_RESTRICT_TO_CPUBINDING
                |HWLOC_TOPOLOGY_FLAG_RESTRICT_TO_MEMBINDING
                |HWLOC_TOPOLOGY_FLAG_DONT_CHANGE_BINDING
                |HWLOC_TOPOLOGY_FLAG_NO_DISTANCES
                |HWLOC_TOPOLOGY_FLAG_NO_MEMATTRS
                |HWLOC_TOPOLOGY_FLAG_NO_CPUKINDS)) {
    errno = EINVAL;
    return -1;
  }

  if ((flags & (HWLOC_TOPOLOGY_FLAG_RESTRICT_TO_CPUBINDING|HWLOC_TOPOLOGY_FLAG_IS_THISSYSTEM)) == HWLOC_TOPOLOGY_FLAG_RESTRICT_TO_CPUBINDING) {
    /* RESTRICT_TO_CPUBINDING requires THISSYSTEM for binding */
    errno = EINVAL;
    return -1;
  }
  if ((flags & (HWLOC_TOPOLOGY_FLAG_RESTRICT_TO_MEMBINDING|HWLOC_TOPOLOGY_FLAG_IS_THISSYSTEM)) == HWLOC_TOPOLOGY_FLAG_RESTRICT_TO_MEMBINDING) {
    /* RESTRICT_TO_MEMBINDING requires THISSYSTEM for binding */
    errno = EINVAL;
    return -1;
  }

  topology->flags = flags;
  return 0;
}

unsigned long
hwloc_topology_get_flags (struct hwloc_topology *topology)
{
  return topology->flags;
}

static void
hwloc__topology_filter_init(struct hwloc_topology *topology)
{
  hwloc_obj_type_t type;
  /* Only ignore useless cruft by default */
  for(type = HWLOC_OBJ_TYPE_MIN; type < HWLOC_OBJ_TYPE_MAX; type++)
    topology->type_filter[type] = HWLOC_TYPE_FILTER_KEEP_ALL;
  topology->type_filter[HWLOC_OBJ_L1ICACHE] = HWLOC_TYPE_FILTER_KEEP_NONE;
  topology->type_filter[HWLOC_OBJ_L2ICACHE] = HWLOC_TYPE_FILTER_KEEP_NONE;
  topology->type_filter[HWLOC_OBJ_L3ICACHE] = HWLOC_TYPE_FILTER_KEEP_NONE;
  topology->type_filter[HWLOC_OBJ_MEMCACHE] = HWLOC_TYPE_FILTER_KEEP_NONE;
  topology->type_filter[HWLOC_OBJ_GROUP] = HWLOC_TYPE_FILTER_KEEP_STRUCTURE;
  topology->type_filter[HWLOC_OBJ_MISC] = HWLOC_TYPE_FILTER_KEEP_NONE;
  topology->type_filter[HWLOC_OBJ_BRIDGE] = HWLOC_TYPE_FILTER_KEEP_NONE;
  topology->type_filter[HWLOC_OBJ_PCI_DEVICE] = HWLOC_TYPE_FILTER_KEEP_NONE;
  topology->type_filter[HWLOC_OBJ_OS_DEVICE] = HWLOC_TYPE_FILTER_KEEP_NONE;
}

static int
hwloc__topology_set_type_filter(struct hwloc_topology *topology, hwloc_obj_type_t type, enum hwloc_type_filter_e filter)
{
  if (type == HWLOC_OBJ_PU || type == HWLOC_OBJ_NUMANODE || type == HWLOC_OBJ_MACHINE) {
    if (filter != HWLOC_TYPE_FILTER_KEEP_ALL) {
      /* we need the Machine, PU and NUMA levels */
      errno = EINVAL;
      return -1;
    }
  } else if (hwloc__obj_type_is_special(type)) {
    if (filter == HWLOC_TYPE_FILTER_KEEP_STRUCTURE) {
      /* I/O and Misc are outside of the main topology structure, makes no sense. */
      errno = EINVAL;
      return -1;
    }
  } else if (type == HWLOC_OBJ_GROUP) {
    if (filter == HWLOC_TYPE_FILTER_KEEP_ALL) {
      /* Groups are always ignored, at least keep_structure */
      errno = EINVAL;
      return -1;
    }
  }

  /* "important" just means "all" for non-I/O non-Misc */
  if (!hwloc__obj_type_is_special(type) && filter == HWLOC_TYPE_FILTER_KEEP_IMPORTANT)
    filter = HWLOC_TYPE_FILTER_KEEP_ALL;

  topology->type_filter[type] = filter;
  return 0;
}

int
hwloc_topology_set_type_filter(struct hwloc_topology *topology, hwloc_obj_type_t type, enum hwloc_type_filter_e filter)
{
  HWLOC_BUILD_ASSERT(HWLOC_OBJ_TYPE_MIN == 0);
  if ((unsigned) type >= HWLOC_OBJ_TYPE_MAX) {
    errno = EINVAL;
    return -1;
  }
  if (topology->is_loaded) {
    errno = EBUSY;
    return -1;
  }
  return hwloc__topology_set_type_filter(topology, type, filter);
}

int
hwloc_topology_set_all_types_filter(struct hwloc_topology *topology, enum hwloc_type_filter_e filter)
{
  hwloc_obj_type_t type;
  if (topology->is_loaded) {
    errno = EBUSY;
    return -1;
  }
  for(type = HWLOC_OBJ_TYPE_MIN; type < HWLOC_OBJ_TYPE_MAX; type++)
    hwloc__topology_set_type_filter(topology, type, filter);
  return 0;
}

int
hwloc_topology_set_cache_types_filter(hwloc_topology_t topology, enum hwloc_type_filter_e filter)
{
  unsigned i;
  if (topology->is_loaded) {
    errno = EBUSY;
    return -1;
  }
  for(i=HWLOC_OBJ_L1CACHE; i<=HWLOC_OBJ_L3ICACHE; i++)
    hwloc__topology_set_type_filter(topology, (hwloc_obj_type_t) i, filter);
  return 0;
}

int
hwloc_topology_set_icache_types_filter(hwloc_topology_t topology, enum hwloc_type_filter_e filter)
{
  unsigned i;
  if (topology->is_loaded) {
    errno = EBUSY;
    return -1;
  }
  for(i=HWLOC_OBJ_L1ICACHE; i<=HWLOC_OBJ_L3ICACHE; i++)
    hwloc__topology_set_type_filter(topology, (hwloc_obj_type_t) i, filter);
  return 0;
}

int
hwloc_topology_set_io_types_filter(hwloc_topology_t topology, enum hwloc_type_filter_e filter)
{
  if (topology->is_loaded) {
    errno = EBUSY;
    return -1;
  }
  hwloc__topology_set_type_filter(topology, HWLOC_OBJ_BRIDGE, filter);
  hwloc__topology_set_type_filter(topology, HWLOC_OBJ_PCI_DEVICE, filter);
  hwloc__topology_set_type_filter(topology, HWLOC_OBJ_OS_DEVICE, filter);
  return 0;
}

int
hwloc_topology_get_type_filter(struct hwloc_topology *topology, hwloc_obj_type_t type, enum hwloc_type_filter_e *filterp)
{
  HWLOC_BUILD_ASSERT(HWLOC_OBJ_TYPE_MIN == 0);
  if ((unsigned) type >= HWLOC_OBJ_TYPE_MAX) {
    errno = EINVAL;
    return -1;
  }
  *filterp = topology->type_filter[type];
  return 0;
}

void
hwloc_topology_clear (struct hwloc_topology *topology)
{
  /* no need to set to NULL after free() since callers will call setup_defaults() or just destroy the rest of the topology */
  unsigned l;

  /* always destroy cpukinds/distances/memattrs since there are always initialized during init() */
  hwloc_internal_cpukinds_destroy(topology);
  hwloc_internal_distances_destroy(topology);
  hwloc_internal_memattrs_destroy(topology);

  hwloc_free_object_and_children(topology->levels[0][0]);
  hwloc_bitmap_free(topology->allowed_cpuset);
  hwloc_bitmap_free(topology->allowed_nodeset);
  for (l=0; l<topology->nb_levels; l++)
    free(topology->levels[l]);
  for(l=0; l<HWLOC_NR_SLEVELS; l++)
    free(topology->slevels[l].objs);
  free(topology->machine_memory.page_types);
}

void
hwloc_topology_destroy (struct hwloc_topology *topology)
{
  if (topology->adopted_shmem_addr) {
    hwloc__topology_disadopt(topology);
    return;
  }

  hwloc_backends_disable_all(topology);
  hwloc_topology_components_fini(topology);
  hwloc_components_fini();

  hwloc_topology_clear(topology);

  free(topology->levels);
  free(topology->level_nbobjects);

  free(topology->support.discovery);
  free(topology->support.cpubind);
  free(topology->support.membind);
  free(topology->support.misc);
  free(topology);
}

int
hwloc_topology_load (struct hwloc_topology *topology)
{
  struct hwloc_disc_status dstatus;
  const char *env;
  unsigned i;
  int err;

  if (topology->is_loaded) {
    errno = EBUSY;
    return -1;
  }

  /* initialize envvar-related things */
  if (!(topology->flags & HWLOC_TOPOLOGY_FLAG_NO_DISTANCES))
    hwloc_internal_distances_prepare(topology);
  if (!(topology->flags & HWLOC_TOPOLOGY_FLAG_NO_MEMATTRS))
    hwloc_internal_memattrs_prepare(topology);

  /* check if any cpu cache filter is not NONE */
  topology->want_some_cpu_caches = 0;
  for(i=HWLOC_OBJ_L1CACHE; i<=HWLOC_OBJ_L3ICACHE; i++)
    if (topology->type_filter[i] != HWLOC_TYPE_FILTER_KEEP_NONE) {
      topology->want_some_cpu_caches = 1;
      break;
    }

  if (getenv("HWLOC_XML_USERDATA_NOT_DECODED"))
    topology->userdata_not_decoded = 1;

  /* Ignore variables if HWLOC_COMPONENTS is set. It will be processed later */
  if (!getenv("HWLOC_COMPONENTS")) {
    /* Only apply variables if we have not changed the backend yet.
     * Only the first one will be kept.
     * Check for FSROOT first since it's for debugging so likely needs to override everything else.
     * Check for XML last (that's the one that may be set system-wide by administrators)
     * so that it's only used if other variables are not set,
     * to allow users to override easily.
     */
    if (!topology->backends) {
      const char *fsroot_path_env = getenv("HWLOC_FSROOT");
      if (fsroot_path_env)
	hwloc_disc_component_force_enable(topology,
					  1 /* env force */,
					  "linux",
					  NULL /* backend will getenv again */, NULL, NULL);
    }
    if (!topology->backends) {
      const char *cpuid_path_env = getenv("HWLOC_CPUID_PATH");
      if (cpuid_path_env)
	hwloc_disc_component_force_enable(topology,
					  1 /* env force */,
					  "x86",
					  NULL /* backend will getenv again */, NULL, NULL);
    }
    if (!topology->backends) {
      const char *synthetic_env = getenv("HWLOC_SYNTHETIC");
      if (synthetic_env)
	hwloc_disc_component_force_enable(topology,
					  1 /* env force */,
					  "synthetic",
					  synthetic_env, NULL, NULL);
    }
    if (!topology->backends) {
      const char *xmlpath_env = getenv("HWLOC_XMLFILE");
      if (xmlpath_env)
	hwloc_disc_component_force_enable(topology,
					  1 /* env force */,
					  "xml",
					  xmlpath_env, NULL, NULL);
    }
  }

  dstatus.excluded_phases = 0;
  dstatus.flags = 0; /* did nothing yet */

  env = getenv("HWLOC_ALLOW");
  if (env && !strcmp(env, "all"))
    /* don't retrieve the sets of allowed resources */
    dstatus.flags |= HWLOC_DISC_STATUS_FLAG_GOT_ALLOWED_RESOURCES;

  /* instantiate all possible other backends now */
  hwloc_disc_components_enable_others(topology);
  /* now that backends are enabled, update the thissystem flag and some callbacks */
  hwloc_backends_is_thissystem(topology);
  hwloc_backends_find_callbacks(topology);
  /*
   * Now set binding hooks according to topology->is_thissystem
   * and what the native OS backend offers.
   */
  hwloc_set_binding_hooks(topology);

  /* actual topology discovery */
  err = hwloc_discover(topology, &dstatus);
  if (err < 0)
    goto out;

#ifndef HWLOC_DEBUG
  if (getenv("HWLOC_DEBUG_CHECK"))
#endif
    hwloc_topology_check(topology);

  if (!(topology->flags & HWLOC_TOPOLOGY_FLAG_NO_CPUKINDS)) {
    /* Rank cpukinds */
    hwloc_internal_cpukinds_rank(topology);
  }

  if (!(topology->flags & HWLOC_TOPOLOGY_FLAG_NO_DISTANCES)) {
    /* Mark distances objs arrays as invalid since we may have removed objects
     * from the topology after adding the distances (remove_empty, etc).
     * It would be hard to actually verify whether it's needed.
     */
    hwloc_internal_distances_invalidate_cached_objs(topology);
    /* And refresh distances so that multithreaded concurrent distances_get()
     * don't refresh() concurrently (disallowed).
     */
    hwloc_internal_distances_refresh(topology);
  }

  if (!(topology->flags & HWLOC_TOPOLOGY_FLAG_NO_MEMATTRS)) {
    int force_memtiers = (getenv("HWLOC_MEMTIERS_REFRESH") != NULL);
    /* Same for memattrs */
    hwloc_internal_memattrs_need_refresh(topology);
    hwloc_internal_memattrs_refresh(topology);
    /* update memtiers unless XML */
    if (force_memtiers || strcmp(topology->backends->component->name, "xml"))
      hwloc_internal_memattrs_guess_memory_tiers(topology, force_memtiers);
  }

  topology->is_loaded = 1;

  if (topology->flags & HWLOC_TOPOLOGY_FLAG_RESTRICT_TO_CPUBINDING) {
    /* FIXME: filter directly in backends during the discovery.
     * Only x86 does it because binding may cause issues on Windows.
     */
    hwloc_bitmap_t set = hwloc_bitmap_alloc();
    if (set) {
      err = hwloc_get_cpubind(topology, set, HWLOC_CPUBIND_STRICT);
      if (!err)
        hwloc_topology_restrict(topology, set, 0);
      hwloc_bitmap_free(set);
    }
  }
  if (topology->flags & HWLOC_TOPOLOGY_FLAG_RESTRICT_TO_MEMBINDING) {
    /* FIXME: filter directly in backends during the discovery.
     */
    hwloc_bitmap_t set = hwloc_bitmap_alloc();
    hwloc_membind_policy_t policy;
    if (set) {
      err = hwloc_get_membind(topology, set, &policy, HWLOC_MEMBIND_STRICT | HWLOC_MEMBIND_BYNODESET);
      if (!err)
        hwloc_topology_restrict(topology, set, HWLOC_RESTRICT_FLAG_BYNODESET);
      hwloc_bitmap_free(set);
    }
  }

  if (topology->backend_phases & HWLOC_DISC_PHASE_TWEAK) {
    dstatus.phase = HWLOC_DISC_PHASE_TWEAK;
    hwloc_discover_by_phase(topology, &dstatus, "TWEAK");
  }

  return 0;

 out:
  hwloc_pci_discovery_exit(topology);
  hwloc_topology_clear(topology);
  hwloc_topology_setup_defaults(topology);
  hwloc_backends_disable_all(topology);
  return -1;
}

/* adjust object cpusets according the given droppedcpuset,
 * drop object whose cpuset becomes empty and that have no children,
 * and propagate NUMA node removal as nodeset changes in parents.
 */
static void
restrict_object_by_cpuset(hwloc_topology_t topology, unsigned long flags, hwloc_obj_t *pobj,
			  hwloc_bitmap_t droppedcpuset, hwloc_bitmap_t droppednodeset)
{
  hwloc_obj_t obj = *pobj, child, *pchild;
  int modified = 0;

  if (hwloc_bitmap_intersects(obj->complete_cpuset, droppedcpuset)) {
    hwloc_bitmap_andnot(obj->cpuset, obj->cpuset, droppedcpuset);
    hwloc_bitmap_andnot(obj->complete_cpuset, obj->complete_cpuset, droppedcpuset);
    modified = 1;
  }
  if (droppednodeset && hwloc_bitmap_intersects(obj->complete_nodeset, droppednodeset)) {
    hwloc_bitmap_andnot(obj->nodeset, obj->nodeset, droppednodeset);
    hwloc_bitmap_andnot(obj->complete_nodeset, obj->complete_nodeset, droppednodeset);
    modified = 1;
  }

  if (modified) {
    for_each_child_safe(child, obj, pchild)
      restrict_object_by_cpuset(topology, flags, pchild, droppedcpuset, droppednodeset);
    /* if some hwloc_bitmap_first(child->complete_cpuset) changed, children might need to be reordered */
    hwloc__reorder_children(obj);

    for_each_memory_child_safe(child, obj, pchild)
      restrict_object_by_cpuset(topology, flags, pchild, droppedcpuset, droppednodeset);
    /* local NUMA nodes have the same cpusets, no need to reorder them */

    /* Nothing to restrict under I/O or Misc */
  }

  if (!obj->first_child && !obj->memory_first_child /* arity not updated before connect_children() */
      && hwloc_bitmap_iszero(obj->cpuset)
      && (obj->type != HWLOC_OBJ_NUMANODE || (flags & HWLOC_RESTRICT_FLAG_REMOVE_CPULESS))) {
    /* remove object */
    hwloc_debug("%s", "\nRemoving object during restrict by cpuset");
    hwloc_debug_print_object(0, obj);

    if (!(flags & HWLOC_RESTRICT_FLAG_ADAPT_IO)) {
      hwloc_free_object_siblings_and_children(obj->io_first_child);
      obj->io_first_child = NULL;
    }
    if (!(flags & HWLOC_RESTRICT_FLAG_ADAPT_MISC)) {
      hwloc_free_object_siblings_and_children(obj->misc_first_child);
      obj->misc_first_child = NULL;
    }
    assert(!obj->first_child);
    assert(!obj->memory_first_child);
    unlink_and_free_single_object(pobj);
    topology->modified = 1;
  }
}

/* adjust object nodesets according the given droppednodeset,
 * drop object whose nodeset becomes empty and that have no children,
 * and propagate PU removal as cpuset changes in parents.
 */
static void
restrict_object_by_nodeset(hwloc_topology_t topology, unsigned long flags, hwloc_obj_t *pobj,
			   hwloc_bitmap_t droppedcpuset, hwloc_bitmap_t droppednodeset)
{
  hwloc_obj_t obj = *pobj, child, *pchild;
  int modified = 0;

  if (hwloc_bitmap_intersects(obj->complete_nodeset, droppednodeset)) {
    hwloc_bitmap_andnot(obj->nodeset, obj->nodeset, droppednodeset);
    hwloc_bitmap_andnot(obj->complete_nodeset, obj->complete_nodeset, droppednodeset);
    modified = 1;
  }
  if (droppedcpuset && hwloc_bitmap_intersects(obj->complete_cpuset, droppedcpuset)) {
    hwloc_bitmap_andnot(obj->cpuset, obj->cpuset, droppedcpuset);
    hwloc_bitmap_andnot(obj->complete_cpuset, obj->complete_cpuset, droppedcpuset);
    modified = 1;
  }

  if (modified) {
    for_each_child_safe(child, obj, pchild)
      restrict_object_by_nodeset(topology, flags, pchild, droppedcpuset, droppednodeset);
    if (flags & HWLOC_RESTRICT_FLAG_REMOVE_MEMLESS)
      /* cpuset may have changed above where some NUMA nodes were removed.
       * if some hwloc_bitmap_first(child->complete_cpuset) changed, children might need to be reordered */
      hwloc__reorder_children(obj);

    for_each_memory_child_safe(child, obj, pchild)
      restrict_object_by_nodeset(topology, flags, pchild, droppedcpuset, droppednodeset);
    /* FIXME: we may have to reorder CPU-less groups of NUMA nodes if some of their nodes were removed */

    /* Nothing to restrict under I/O or Misc */
  }

  if (!obj->first_child && !obj->memory_first_child /* arity not updated before connect_children() */
      && hwloc_bitmap_iszero(obj->nodeset)
      && (obj->type != HWLOC_OBJ_PU || (flags & HWLOC_RESTRICT_FLAG_REMOVE_MEMLESS))) {
    /* remove object */
    hwloc_debug("%s", "\nRemoving object during restrict by nodeset");
    hwloc_debug_print_object(0, obj);

    if (!(flags & HWLOC_RESTRICT_FLAG_ADAPT_IO)) {
      hwloc_free_object_siblings_and_children(obj->io_first_child);
      obj->io_first_child = NULL;
    }
    if (!(flags & HWLOC_RESTRICT_FLAG_ADAPT_MISC)) {
      hwloc_free_object_siblings_and_children(obj->misc_first_child);
      obj->misc_first_child = NULL;
    }
    assert(!obj->first_child);
    assert(!obj->memory_first_child);
    unlink_and_free_single_object(pobj);
    topology->modified = 1;
  }
}

int
hwloc_topology_restrict(struct hwloc_topology *topology, hwloc_const_bitmap_t set, unsigned long flags)
{
  hwloc_bitmap_t droppedcpuset, droppednodeset;

  if (!topology->is_loaded) {
    errno = EINVAL;
    return -1;
  }
  if (topology->adopted_shmem_addr) {
    errno = EPERM;
    return -1;
  }

  if (flags & ~(HWLOC_RESTRICT_FLAG_REMOVE_CPULESS
		|HWLOC_RESTRICT_FLAG_ADAPT_MISC|HWLOC_RESTRICT_FLAG_ADAPT_IO
		|HWLOC_RESTRICT_FLAG_BYNODESET|HWLOC_RESTRICT_FLAG_REMOVE_MEMLESS)) {
    errno = EINVAL;
    return -1;
  }

  if (flags & HWLOC_RESTRICT_FLAG_BYNODESET) {
    /* cannot use CPULESS with BYNODESET */
    if (flags & HWLOC_RESTRICT_FLAG_REMOVE_CPULESS) {
      errno = EINVAL;
      return -1;
    }
  } else {
    /* cannot use MEMLESS without BYNODESET */
    if (flags & HWLOC_RESTRICT_FLAG_REMOVE_MEMLESS) {
      errno = EINVAL;
      return -1;
    }
  }

  /* make sure we'll keep something in the topology */
  if (((flags & HWLOC_RESTRICT_FLAG_BYNODESET) && !hwloc_bitmap_intersects(set, topology->allowed_nodeset))
      || (!(flags & HWLOC_RESTRICT_FLAG_BYNODESET) && !hwloc_bitmap_intersects(set, topology->allowed_cpuset))) {
    errno = EINVAL; /* easy failure, just don't touch the topology */
    return -1;
  }

  droppedcpuset = hwloc_bitmap_alloc();
  droppednodeset = hwloc_bitmap_alloc();
  if (!droppedcpuset || !droppednodeset) {
    hwloc_bitmap_free(droppedcpuset);
    hwloc_bitmap_free(droppednodeset);
    return -1;
  }

  if (flags & HWLOC_RESTRICT_FLAG_BYNODESET) {
    /* nodeset to clear */
    hwloc_bitmap_not(droppednodeset, set);
    /* cpuset to clear */
    if (flags & HWLOC_RESTRICT_FLAG_REMOVE_MEMLESS) {
      hwloc_obj_t pu = hwloc_get_obj_by_type(topology, HWLOC_OBJ_PU, 0);
      assert(pu);
      do {
	/* PU will be removed if cpuset gets or was empty */
	if (hwloc_bitmap_iszero(pu->cpuset)
	    || hwloc_bitmap_isincluded(pu->nodeset, droppednodeset))
	  hwloc_bitmap_set(droppedcpuset, pu->os_index);
	pu = pu->next_cousin;
      } while (pu);

      /* check we're not removing all PUs */
      if (hwloc_bitmap_isincluded(topology->allowed_cpuset, droppedcpuset)) {
	errno = EINVAL; /* easy failure, just don't touch the topology */
	hwloc_bitmap_free(droppedcpuset);
	hwloc_bitmap_free(droppednodeset);
	return -1;
      }
    }
    /* remove cpuset if empty */
    if (!(flags & HWLOC_RESTRICT_FLAG_REMOVE_MEMLESS)
	|| hwloc_bitmap_iszero(droppedcpuset)) {
      hwloc_bitmap_free(droppedcpuset);
      droppedcpuset = NULL;
    }

    /* now recurse to filter sets and drop things */
    restrict_object_by_nodeset(topology, flags, &topology->levels[0][0], droppedcpuset, droppednodeset);
    hwloc_bitmap_andnot(topology->allowed_nodeset, topology->allowed_nodeset, droppednodeset);
    if (droppedcpuset)
      hwloc_bitmap_andnot(topology->allowed_cpuset, topology->allowed_cpuset, droppedcpuset);

  } else {
    /* cpuset to clear */
    hwloc_bitmap_not(droppedcpuset, set);
    /* nodeset to clear */
    if (flags & HWLOC_RESTRICT_FLAG_REMOVE_CPULESS) {
      hwloc_obj_t node = hwloc_get_obj_by_type(topology, HWLOC_OBJ_NUMANODE, 0);
      assert(node);
      do {
	/* node will be removed if nodeset gets or was empty */
	if (hwloc_bitmap_iszero(node->cpuset)
	    || hwloc_bitmap_isincluded(node->cpuset, droppedcpuset))
	  hwloc_bitmap_set(droppednodeset, node->os_index);
	node = node->next_cousin;
      } while (node);

      /* check we're not removing all NUMA nodes */
      if (hwloc_bitmap_isincluded(topology->allowed_nodeset, droppednodeset)) {
	errno = EINVAL; /* easy failure, just don't touch the topology */
	hwloc_bitmap_free(droppedcpuset);
	hwloc_bitmap_free(droppednodeset);
	return -1;
      }
    }
    /* remove nodeset if empty */
    if (!(flags & HWLOC_RESTRICT_FLAG_REMOVE_CPULESS)
	|| hwloc_bitmap_iszero(droppednodeset)) {
      hwloc_bitmap_free(droppednodeset);
      droppednodeset = NULL;
    }

    /* now recurse to filter sets and drop things */
    restrict_object_by_cpuset(topology, flags, &topology->levels[0][0], droppedcpuset, droppednodeset);
    hwloc_bitmap_andnot(topology->allowed_cpuset, topology->allowed_cpuset, droppedcpuset);
    if (droppednodeset)
      hwloc_bitmap_andnot(topology->allowed_nodeset, topology->allowed_nodeset, droppednodeset);
  }

  hwloc_bitmap_free(droppedcpuset);
  hwloc_bitmap_free(droppednodeset);

  if (hwloc_filter_levels_keep_structure(topology) < 0) /* takes care of reconnecting internally */
    goto out;

  /* some objects may have disappeared and sets were modified,
   * we need to update distances, etc */
  if (!(topology->flags & HWLOC_TOPOLOGY_FLAG_NO_DISTANCES))
    hwloc_internal_distances_invalidate_cached_objs(topology);
  if (!(topology->flags & HWLOC_TOPOLOGY_FLAG_NO_MEMATTRS))
    hwloc_internal_memattrs_need_refresh(topology);
  if (!(topology->flags & HWLOC_TOPOLOGY_FLAG_NO_CPUKINDS))
    hwloc_internal_cpukinds_restrict(topology);


  hwloc_propagate_symmetric_subtree(topology, topology->levels[0][0]);
  propagate_total_memory(topology->levels[0][0]);

#ifndef HWLOC_DEBUG
  if (getenv("HWLOC_DEBUG_CHECK"))
#endif
    hwloc_topology_check(topology);

  return 0;

 out:
  /* unrecoverable failure, re-init the topology */
   hwloc_topology_clear(topology);
   hwloc_topology_setup_defaults(topology);
   return -1;
}

int
hwloc_topology_allow(struct hwloc_topology *topology,
		     hwloc_const_cpuset_t cpuset, hwloc_const_nodeset_t nodeset,
		     unsigned long flags)
{
  if (!topology->is_loaded)
    goto einval;

  if (topology->adopted_shmem_addr) {
    errno = EPERM;
    goto error;
  }

  if (!(topology->flags & HWLOC_TOPOLOGY_FLAG_INCLUDE_DISALLOWED))
    goto einval;

  if (flags & ~(HWLOC_ALLOW_FLAG_ALL|HWLOC_ALLOW_FLAG_LOCAL_RESTRICTIONS|HWLOC_ALLOW_FLAG_CUSTOM))
    goto einval;

  switch (flags) {
  case HWLOC_ALLOW_FLAG_ALL: {
    if (cpuset || nodeset)
      goto einval;
    hwloc_bitmap_copy(topology->allowed_cpuset, hwloc_get_root_obj(topology)->complete_cpuset);
    hwloc_bitmap_copy(topology->allowed_nodeset, hwloc_get_root_obj(topology)->complete_nodeset);
    break;
  }
  case HWLOC_ALLOW_FLAG_LOCAL_RESTRICTIONS: {
    if (cpuset || nodeset)
      goto einval;
    if (!topology->is_thissystem)
      goto einval;
    if (!topology->binding_hooks.get_allowed_resources) {
      errno = ENOSYS;
      goto error;
    }
    topology->binding_hooks.get_allowed_resources(topology);
    /* make sure the backend returned something sane (Linux cpusets may return offline PUs in some cases) */
    hwloc_bitmap_and(topology->allowed_cpuset, topology->allowed_cpuset, hwloc_get_root_obj(topology)->cpuset);
    hwloc_bitmap_and(topology->allowed_nodeset, topology->allowed_nodeset, hwloc_get_root_obj(topology)->nodeset);
    break;
  }
  case HWLOC_ALLOW_FLAG_CUSTOM: {
    if (cpuset) {
      /* keep the intersection with the full topology cpuset, if not empty */
      if (!hwloc_bitmap_intersects(hwloc_get_root_obj(topology)->cpuset, cpuset))
	goto einval;
      hwloc_bitmap_and(topology->allowed_cpuset, hwloc_get_root_obj(topology)->cpuset, cpuset);
    }
    if (nodeset) {
      /* keep the intersection with the full topology nodeset, if not empty */
      if (!hwloc_bitmap_intersects(hwloc_get_root_obj(topology)->nodeset, nodeset))
	goto einval;
      hwloc_bitmap_and(topology->allowed_nodeset, hwloc_get_root_obj(topology)->nodeset, nodeset);
    }
    break;
  }
  default:
    goto einval;
  }

  return 0;

 einval:
  errno = EINVAL;
 error:
  return -1;
}

int
hwloc_topology_refresh(struct hwloc_topology *topology)
{
  if (!(topology->flags & HWLOC_TOPOLOGY_FLAG_NO_CPUKINDS))
    hwloc_internal_cpukinds_rank(topology);
  if (!(topology->flags & HWLOC_TOPOLOGY_FLAG_NO_DISTANCES))
    hwloc_internal_distances_refresh(topology);
  if (!(topology->flags & HWLOC_TOPOLOGY_FLAG_NO_MEMATTRS))
    hwloc_internal_memattrs_refresh(topology);
  return 0;
}

int
hwloc_topology_is_thissystem(struct hwloc_topology *topology)
{
  return topology->is_thissystem;
}

int
hwloc_topology_get_depth(struct hwloc_topology *topology)
{
  return (int) topology->nb_levels;
}

const struct hwloc_topology_support *
hwloc_topology_get_support(struct hwloc_topology * topology)
{
  return &topology->support;
}

void hwloc_topology_set_userdata(struct hwloc_topology * topology, const void *userdata)
{
  topology->userdata = (void *) userdata;
}

void * hwloc_topology_get_userdata(struct hwloc_topology * topology)
{
  return topology->userdata;
}

hwloc_const_cpuset_t
hwloc_topology_get_complete_cpuset(hwloc_topology_t topology)
{
  return hwloc_get_root_obj(topology)->complete_cpuset;
}

hwloc_const_cpuset_t
hwloc_topology_get_topology_cpuset(hwloc_topology_t topology)
{
  return hwloc_get_root_obj(topology)->cpuset;
}

hwloc_const_cpuset_t
hwloc_topology_get_allowed_cpuset(hwloc_topology_t topology)
{
  return topology->allowed_cpuset;
}

hwloc_const_nodeset_t
hwloc_topology_get_complete_nodeset(hwloc_topology_t topology)
{
  return hwloc_get_root_obj(topology)->complete_nodeset;
}

hwloc_const_nodeset_t
hwloc_topology_get_topology_nodeset(hwloc_topology_t topology)
{
  return hwloc_get_root_obj(topology)->nodeset;
}

hwloc_const_nodeset_t
hwloc_topology_get_allowed_nodeset(hwloc_topology_t topology)
{
  return topology->allowed_nodeset;
}


/****************
 * Debug Checks *
 ****************/

#ifndef NDEBUG /* assert only enabled if !NDEBUG */

static void
hwloc__check_child_siblings(hwloc_obj_t parent, hwloc_obj_t *array,
			    unsigned arity, unsigned i,
			    hwloc_obj_t child, hwloc_obj_t prev)
{
  assert(child->parent == parent);

  assert(child->sibling_rank == i);
  if (array)
    assert(child == array[i]);

  if (prev)
    assert(prev->next_sibling == child);
  assert(child->prev_sibling == prev);

  if (!i)
    assert(child->prev_sibling == NULL);
  else
    assert(child->prev_sibling != NULL);

  if (i == arity-1)
    assert(child->next_sibling == NULL);
  else
    assert(child->next_sibling != NULL);
}

static void
hwloc__check_object(hwloc_topology_t topology, hwloc_bitmap_t gp_indexes, hwloc_obj_t obj);

/* check children between a parent object */
static void
hwloc__check_normal_children(hwloc_topology_t topology, hwloc_bitmap_t gp_indexes, hwloc_obj_t parent)
{
  hwloc_obj_t child, prev;
  unsigned j;

  if (!parent->arity) {
    /* check whether that parent has no children for real */
    assert(!parent->children);
    assert(!parent->first_child);
    assert(!parent->last_child);
    return;
  }
  /* check whether that parent has children for real */
  assert(parent->children);
  assert(parent->first_child);
  assert(parent->last_child);

  /* sibling checks */
  for(prev = NULL, child = parent->first_child, j = 0;
      child;
      prev = child, child = child->next_sibling, j++) {
    /* normal child */
    assert(hwloc__obj_type_is_normal(child->type));
    /* check depth */
    assert(child->depth > parent->depth);
    /* check siblings */
    hwloc__check_child_siblings(parent, parent->children, parent->arity, j, child, prev);
    /* recurse */
    hwloc__check_object(topology, gp_indexes, child);
  }
  /* check arity */
  assert(j == parent->arity);

  assert(parent->first_child == parent->children[0]);
  assert(parent->last_child == parent->children[parent->arity-1]);

  /* no normal children below a PU */
  if (parent->type == HWLOC_OBJ_PU)
    assert(!parent->arity);
}

static void
hwloc__check_children_cpusets(hwloc_topology_t topology __hwloc_attribute_unused, hwloc_obj_t obj)
{
  /* we already checked in the caller that objects have either all sets or none */
  hwloc_obj_t child;
  int prev_first, prev_empty;

  if (obj->type == HWLOC_OBJ_PU) {
    /* PU cpuset is just itself, with no normal children */
    assert(hwloc_bitmap_weight(obj->cpuset) == 1);
    assert(hwloc_bitmap_first(obj->cpuset) == (int) obj->os_index);
    assert(hwloc_bitmap_weight(obj->complete_cpuset) == 1);
    assert(hwloc_bitmap_first(obj->complete_cpuset) == (int) obj->os_index);
    if (!(topology->flags & HWLOC_TOPOLOGY_FLAG_INCLUDE_DISALLOWED)) {
      assert(hwloc_bitmap_isset(topology->allowed_cpuset, (int) obj->os_index));
    }
    assert(!obj->arity);
  } else if (hwloc__obj_type_is_memory(obj->type)) {
    /* memory object cpuset is equal to its parent */
    assert(hwloc_bitmap_isequal(obj->parent->cpuset, obj->cpuset));
    assert(!obj->arity);
  } else if (!hwloc__obj_type_is_special(obj->type)) {
    hwloc_bitmap_t set;
    /* other obj cpuset is an exclusive OR of normal children, except for PUs */
    set = hwloc_bitmap_alloc();
    for_each_child(child, obj) {
      assert(!hwloc_bitmap_intersects(set, child->cpuset));
      hwloc_bitmap_or(set, set, child->cpuset);
    }
    assert(hwloc_bitmap_isequal(set, obj->cpuset));
    hwloc_bitmap_free(set);
  }

  /* check that memory children have same cpuset */
  for_each_memory_child(child, obj)
    assert(hwloc_bitmap_isequal(obj->cpuset, child->cpuset));

  /* check that children complete_cpusets are properly ordered, empty ones may be anywhere
   * (can be wrong for main cpuset since removed PUs can break the ordering).
   */
  prev_first = -1; /* -1 works fine with first comparisons below */
  prev_empty = 0; /* no empty cpuset in previous children */
  for_each_child(child, obj) {
    int first = hwloc_bitmap_first(child->complete_cpuset);
    if (first >= 0) {
      assert(!prev_empty); /* no objects with CPU after objects without CPU */
      assert(prev_first < first);
    } else {
      prev_empty = 1;
    }
    prev_first = first;
  }
}

static void
hwloc__check_memory_children(hwloc_topology_t topology, hwloc_bitmap_t gp_indexes, hwloc_obj_t parent)
{
  unsigned j;
  hwloc_obj_t child, prev;

  if (!parent->memory_arity) {
    /* check whether that parent has no children for real */
    assert(!parent->memory_first_child);
    return;
  }
  /* check whether that parent has children for real */
  assert(parent->memory_first_child);

  for(prev = NULL, child = parent->memory_first_child, j = 0;
      child;
      prev = child, child = child->next_sibling, j++) {
    assert(hwloc__obj_type_is_memory(child->type));
    /* check siblings */
    hwloc__check_child_siblings(parent, NULL, parent->memory_arity, j, child, prev);
    /* only Memory and Misc children, recurse */
    assert(!child->first_child);
    assert(!child->io_first_child);
    hwloc__check_object(topology, gp_indexes, child);
  }
  /* check arity */
  assert(j == parent->memory_arity);

  /* no memory children below a NUMA node */
  if (parent->type == HWLOC_OBJ_NUMANODE)
    assert(!parent->memory_arity);
}

static void
hwloc__check_io_children(hwloc_topology_t topology, hwloc_bitmap_t gp_indexes, hwloc_obj_t parent)
{
  unsigned j;
  hwloc_obj_t child, prev;

  if (!parent->io_arity) {
    /* check whether that parent has no children for real */
    assert(!parent->io_first_child);
    return;
  }
  /* check whether that parent has children for real */
  assert(parent->io_first_child);

  for(prev = NULL, child = parent->io_first_child, j = 0;
      child;
      prev = child, child = child->next_sibling, j++) {
    /* all children must be I/O */
    assert(hwloc__obj_type_is_io(child->type));
    /* check siblings */
    hwloc__check_child_siblings(parent, NULL, parent->io_arity, j, child, prev);
    /* only I/O and Misc children, recurse */
    assert(!child->first_child);
    assert(!child->memory_first_child);
    hwloc__check_object(topology, gp_indexes, child);
  }
  /* check arity */
  assert(j == parent->io_arity);
}

static void
hwloc__check_misc_children(hwloc_topology_t topology, hwloc_bitmap_t gp_indexes, hwloc_obj_t parent)
{
  unsigned j;
  hwloc_obj_t child, prev;

  if (!parent->misc_arity) {
    /* check whether that parent has no children for real */
    assert(!parent->misc_first_child);
    return;
  }
  /* check whether that parent has children for real */
  assert(parent->misc_first_child);

  for(prev = NULL, child = parent->misc_first_child, j = 0;
      child;
      prev = child, child = child->next_sibling, j++) {
    /* all children must be Misc */
    assert(child->type == HWLOC_OBJ_MISC);
    /* check siblings */
    hwloc__check_child_siblings(parent, NULL, parent->misc_arity, j, child, prev);
    /* only Misc children, recurse */
    assert(!child->first_child);
    assert(!child->memory_first_child);
    assert(!child->io_first_child);
    hwloc__check_object(topology, gp_indexes, child);
  }
  /* check arity */
  assert(j == parent->misc_arity);
}

static void
hwloc__check_object(hwloc_topology_t topology, hwloc_bitmap_t gp_indexes, hwloc_obj_t obj)
{
  hwloc_uint64_t total_memory;
  hwloc_obj_t child;

  assert(!hwloc_bitmap_isset(gp_indexes, obj->gp_index));
  hwloc_bitmap_set(gp_indexes, obj->gp_index);

  HWLOC_BUILD_ASSERT(HWLOC_OBJ_TYPE_MIN == 0);
  assert((unsigned) obj->type < HWLOC_OBJ_TYPE_MAX);

  assert(hwloc_filter_check_keep_object(topology, obj));

  /* check that sets and depth */
  if (hwloc__obj_type_is_special(obj->type)) {
    assert(!obj->cpuset);
    if (obj->type == HWLOC_OBJ_BRIDGE)
      assert(obj->depth == HWLOC_TYPE_DEPTH_BRIDGE);
    else if (obj->type == HWLOC_OBJ_PCI_DEVICE)
      assert(obj->depth == HWLOC_TYPE_DEPTH_PCI_DEVICE);
    else if (obj->type == HWLOC_OBJ_OS_DEVICE)
      assert(obj->depth == HWLOC_TYPE_DEPTH_OS_DEVICE);
    else if (obj->type == HWLOC_OBJ_MISC)
      assert(obj->depth == HWLOC_TYPE_DEPTH_MISC);
  } else {
    assert(obj->cpuset);
    if (obj->type == HWLOC_OBJ_NUMANODE)
      assert(obj->depth == HWLOC_TYPE_DEPTH_NUMANODE);
    else if (obj->type == HWLOC_OBJ_MEMCACHE)
      assert(obj->depth == HWLOC_TYPE_DEPTH_MEMCACHE);
    else
      assert(obj->depth >= 0);
  }

  /* group depth cannot be -1 anymore in v2.0+ */
  if (obj->type == HWLOC_OBJ_GROUP) {
    assert(obj->attr->group.depth != (unsigned) -1);
  }

  /* there's other cpusets and nodesets if and only if there's a main cpuset */
  assert(!!obj->cpuset == !!obj->complete_cpuset);
  assert(!!obj->cpuset == !!obj->nodeset);
  assert(!!obj->nodeset == !!obj->complete_nodeset);

  /* check that complete/inline sets are larger than the main sets */
  if (obj->cpuset) {
    assert(hwloc_bitmap_isincluded(obj->cpuset, obj->complete_cpuset));
    assert(hwloc_bitmap_isincluded(obj->nodeset, obj->complete_nodeset));
  }

  /* check cache type/depth vs type */
  if (hwloc__obj_type_is_cache(obj->type)) {
    if (hwloc__obj_type_is_icache(obj->type))
      assert(obj->attr->cache.type == HWLOC_OBJ_CACHE_INSTRUCTION);
    else if (hwloc__obj_type_is_dcache(obj->type))
      assert(obj->attr->cache.type == HWLOC_OBJ_CACHE_DATA
	     || obj->attr->cache.type == HWLOC_OBJ_CACHE_UNIFIED);
    else
      assert(0);
    assert(hwloc_cache_type_by_depth_type(obj->attr->cache.depth, obj->attr->cache.type) == obj->type);
  }

  /* check total memory */
  total_memory = 0;
  if (obj->type == HWLOC_OBJ_NUMANODE)
    total_memory += obj->attr->numanode.local_memory;
  for_each_child(child, obj) {
    total_memory += child->total_memory;
  }
  for_each_memory_child(child, obj) {
    total_memory += child->total_memory;
  }
  assert(total_memory == obj->total_memory);

  /* check children */
  hwloc__check_normal_children(topology, gp_indexes, obj);
  hwloc__check_memory_children(topology, gp_indexes, obj);
  hwloc__check_io_children(topology, gp_indexes, obj);
  hwloc__check_misc_children(topology, gp_indexes, obj);
  hwloc__check_children_cpusets(topology, obj);
  /* nodesets are checked during another recursion with state below */
}

static void
hwloc__check_nodesets(hwloc_topology_t topology, hwloc_obj_t obj, hwloc_bitmap_t parentset)
{
  hwloc_obj_t child;
  int prev_first;

  if (obj->type == HWLOC_OBJ_NUMANODE) {
    /* NUMANODE nodeset is just itself, with no memory/normal children */
    assert(hwloc_bitmap_weight(obj->nodeset) == 1);
    assert(hwloc_bitmap_first(obj->nodeset) == (int) obj->os_index);
    assert(hwloc_bitmap_weight(obj->complete_nodeset) == 1);
    assert(hwloc_bitmap_first(obj->complete_nodeset) == (int) obj->os_index);
    if (!(topology->flags & HWLOC_TOPOLOGY_FLAG_INCLUDE_DISALLOWED)) {
      assert(hwloc_bitmap_isset(topology->allowed_nodeset, (int) obj->os_index));
    }
    assert(!obj->arity);
    assert(!obj->memory_arity);
    assert(hwloc_bitmap_isincluded(obj->nodeset, parentset));
  } else {
    hwloc_bitmap_t myset;
    hwloc_bitmap_t childset;

    /* the local nodeset is an exclusive OR of memory children */
    myset = hwloc_bitmap_alloc();
    for_each_memory_child(child, obj) {
      assert(!hwloc_bitmap_intersects(myset, child->nodeset));
      hwloc_bitmap_or(myset, myset, child->nodeset);
    }
    /* the local nodeset cannot intersect with parents' local nodeset */
    assert(!hwloc_bitmap_intersects(myset, parentset));
    hwloc_bitmap_or(parentset, parentset, myset);
    hwloc_bitmap_free(myset);
    /* parentset now contains parent+local contribution */

    /* for each children, recurse to check/get its contribution */
    childset = hwloc_bitmap_alloc();
    for_each_child(child, obj) {
      hwloc_bitmap_t set = hwloc_bitmap_dup(parentset); /* don't touch parentset, we don't want to propagate the first child contribution to other children */
      hwloc__check_nodesets(topology, child, set);
      /* extract this child contribution */
      hwloc_bitmap_andnot(set, set, parentset);
      /* save it */
      assert(!hwloc_bitmap_intersects(childset, set));
      hwloc_bitmap_or(childset, childset, set);
      hwloc_bitmap_free(set);
    }
    /* combine child contribution into parentset */
    assert(!hwloc_bitmap_intersects(parentset, childset));
    hwloc_bitmap_or(parentset, parentset, childset);
    hwloc_bitmap_free(childset);
    /* now check that our nodeset is combination of parent, local and children */
    assert(hwloc_bitmap_isequal(obj->nodeset, parentset));
  }

  /* check that children complete_nodesets are properly ordered, empty ones may be anywhere
   * (can be wrong for main nodeset since removed PUs can break the ordering).
   */
  prev_first = -1; /* -1 works fine with first comparisons below */
  for_each_memory_child(child, obj) {
    int first = hwloc_bitmap_first(child->complete_nodeset);
    assert(prev_first < first);
    prev_first = first;
  }
}

static void
hwloc__check_level(struct hwloc_topology *topology, int depth,
		   hwloc_obj_t first, hwloc_obj_t last)
{
  unsigned width = hwloc_get_nbobjs_by_depth(topology, depth);
  struct hwloc_obj *prev = NULL;
  hwloc_obj_t obj;
  unsigned j;

  /* check each object of the level */
  for(j=0; j<width; j++) {
    obj = hwloc_get_obj_by_depth(topology, depth, j);
    /* check that the object is corrected placed horizontally and vertically */
    assert(obj);
    assert(obj->depth == depth);
    assert(obj->logical_index == j);
    /* check that all objects in the level have the same type */
    if (prev) {
      assert(hwloc_type_cmp(obj, prev) == HWLOC_OBJ_EQUAL);
      assert(prev->next_cousin == obj);
    }
    assert(obj->prev_cousin == prev);

    /* check that PUs and NUMA nodes have correct cpuset/nodeset */
    if (obj->type == HWLOC_OBJ_NUMANODE) {
      assert(hwloc_bitmap_weight(obj->complete_nodeset) == 1);
      assert(hwloc_bitmap_first(obj->complete_nodeset) == (int) obj->os_index);
    }
    prev = obj;
  }
  if (prev)
    assert(prev->next_cousin == NULL);

  if (width) {
    /* check first object of the level */
    obj = hwloc_get_obj_by_depth(topology, depth, 0);
    assert(obj);
    assert(!obj->prev_cousin);
    /* check type */
    assert(hwloc_get_depth_type(topology, depth) == obj->type);
    assert(depth == hwloc_get_type_depth(topology, obj->type)
	   || HWLOC_TYPE_DEPTH_MULTIPLE == hwloc_get_type_depth(topology, obj->type));
    /* check last object of the level */
    obj = hwloc_get_obj_by_depth(topology, depth, width-1);
    assert(obj);
    assert(!obj->next_cousin);
  }

  if (depth < 0) {
    assert(first == hwloc_get_obj_by_depth(topology, depth, 0));
    assert(last == hwloc_get_obj_by_depth(topology, depth, width-1));
  } else {
    assert(!first);
    assert(!last);
  }

  /* check last+1 object of the level */
  obj = hwloc_get_obj_by_depth(topology, depth, width);
  assert(!obj);
}

/* check a whole topology structure */
void
hwloc_topology_check(struct hwloc_topology *topology)
{
  struct hwloc_obj *obj;
  hwloc_bitmap_t gp_indexes, set;
  hwloc_obj_type_t type;
  unsigned i;
  int j, depth;

  /* make sure we can use ranges to check types */

  /* hwloc__obj_type_is_{,d,i}cache() want cache types to be ordered like this */
  HWLOC_BUILD_ASSERT(HWLOC_OBJ_L2CACHE == HWLOC_OBJ_L1CACHE + 1);
  HWLOC_BUILD_ASSERT(HWLOC_OBJ_L3CACHE == HWLOC_OBJ_L2CACHE + 1);
  HWLOC_BUILD_ASSERT(HWLOC_OBJ_L4CACHE == HWLOC_OBJ_L3CACHE + 1);
  HWLOC_BUILD_ASSERT(HWLOC_OBJ_L5CACHE == HWLOC_OBJ_L4CACHE + 1);
  HWLOC_BUILD_ASSERT(HWLOC_OBJ_L1ICACHE == HWLOC_OBJ_L5CACHE + 1);
  HWLOC_BUILD_ASSERT(HWLOC_OBJ_L2ICACHE == HWLOC_OBJ_L1ICACHE + 1);
  HWLOC_BUILD_ASSERT(HWLOC_OBJ_L3ICACHE == HWLOC_OBJ_L2ICACHE + 1);

  /* hwloc__obj_type_is_normal(), hwloc__obj_type_is_memory(), hwloc__obj_type_is_io(), hwloc__obj_type_is_special()
   * and hwloc_reset_normal_type_depths()
   * want special types to be ordered like this, after all normal types.
   */
  HWLOC_BUILD_ASSERT(HWLOC_OBJ_NUMANODE   + 1 == HWLOC_OBJ_BRIDGE);
  HWLOC_BUILD_ASSERT(HWLOC_OBJ_BRIDGE     + 1 == HWLOC_OBJ_PCI_DEVICE);
  HWLOC_BUILD_ASSERT(HWLOC_OBJ_PCI_DEVICE + 1 == HWLOC_OBJ_OS_DEVICE);
  HWLOC_BUILD_ASSERT(HWLOC_OBJ_OS_DEVICE  + 1 == HWLOC_OBJ_MISC);
  HWLOC_BUILD_ASSERT(HWLOC_OBJ_MISC       + 1 == HWLOC_OBJ_MEMCACHE);
  HWLOC_BUILD_ASSERT(HWLOC_OBJ_MEMCACHE   + 1 == HWLOC_OBJ_DIE);
  HWLOC_BUILD_ASSERT(HWLOC_OBJ_DIE        + 1 == HWLOC_OBJ_TYPE_MAX);

  /* make sure order and priority arrays have the right size */
  HWLOC_BUILD_ASSERT(sizeof(obj_type_order)/sizeof(*obj_type_order) == HWLOC_OBJ_TYPE_MAX);
  HWLOC_BUILD_ASSERT(sizeof(obj_order_type)/sizeof(*obj_order_type) == HWLOC_OBJ_TYPE_MAX);
  HWLOC_BUILD_ASSERT(sizeof(obj_type_priority)/sizeof(*obj_type_priority) == HWLOC_OBJ_TYPE_MAX);

  /* make sure group are not entirely ignored */
  assert(topology->type_filter[HWLOC_OBJ_GROUP] != HWLOC_TYPE_FILTER_KEEP_ALL);

  /* make sure order arrays are coherent */
  for(type=HWLOC_OBJ_TYPE_MIN; type<HWLOC_OBJ_TYPE_MAX; type++)
    assert(obj_order_type[obj_type_order[type]] == type);
  for(i=HWLOC_OBJ_TYPE_MIN; i<HWLOC_OBJ_TYPE_MAX; i++)
    assert(obj_type_order[obj_order_type[i]] == i);

  if (!topology->is_loaded)
    return;

  depth = hwloc_topology_get_depth(topology);

  assert(!topology->modified);

  /* check that first level is Machine.
   * Root object cannot be ignored. And Machine can only be merged into PU,
   * but there must be a NUMA node below Machine, and it cannot be below PU.
   */
  assert(hwloc_get_depth_type(topology, 0) == HWLOC_OBJ_MACHINE);

  /* check that last level is PU and that it doesn't have memory */
  assert(hwloc_get_depth_type(topology, depth-1) == HWLOC_OBJ_PU);
  assert(hwloc_get_nbobjs_by_depth(topology, depth-1) > 0);
  for(i=0; i<hwloc_get_nbobjs_by_depth(topology, depth-1); i++) {
    obj = hwloc_get_obj_by_depth(topology, depth-1, i);
    assert(obj);
    assert(obj->type == HWLOC_OBJ_PU);
    assert(!obj->memory_first_child);
  }
  /* check that other levels are not PU or Machine */
  for(j=1; j<depth-1; j++) {
    assert(hwloc_get_depth_type(topology, j) != HWLOC_OBJ_PU);
    assert(hwloc_get_depth_type(topology, j) != HWLOC_OBJ_MACHINE);
  }

  /* check normal levels */
  for(j=0; j<depth; j++) {
    int d;
    type = hwloc_get_depth_type(topology, j);
    assert(type != HWLOC_OBJ_NUMANODE);
    assert(type != HWLOC_OBJ_MEMCACHE);
    assert(type != HWLOC_OBJ_PCI_DEVICE);
    assert(type != HWLOC_OBJ_BRIDGE);
    assert(type != HWLOC_OBJ_OS_DEVICE);
    assert(type != HWLOC_OBJ_MISC);
    d = hwloc_get_type_depth(topology, type);
    assert(d == j || d == HWLOC_TYPE_DEPTH_MULTIPLE);
  }

  /* check type depths, even if there's no such level */
  for(type=HWLOC_OBJ_TYPE_MIN; type<HWLOC_OBJ_TYPE_MAX; type++) {
    int d;
    d = hwloc_get_type_depth(topology, type);
    if (type == HWLOC_OBJ_NUMANODE) {
      assert(d == HWLOC_TYPE_DEPTH_NUMANODE);
      assert(hwloc_get_depth_type(topology, d) == HWLOC_OBJ_NUMANODE);
    } else if (type == HWLOC_OBJ_MEMCACHE) {
      assert(d == HWLOC_TYPE_DEPTH_MEMCACHE);
      assert(hwloc_get_depth_type(topology, d) == HWLOC_OBJ_MEMCACHE);
    } else if (type == HWLOC_OBJ_BRIDGE) {
      assert(d == HWLOC_TYPE_DEPTH_BRIDGE);
      assert(hwloc_get_depth_type(topology, d) == HWLOC_OBJ_BRIDGE);
    } else if (type == HWLOC_OBJ_PCI_DEVICE) {
      assert(d == HWLOC_TYPE_DEPTH_PCI_DEVICE);
      assert(hwloc_get_depth_type(topology, d) == HWLOC_OBJ_PCI_DEVICE);
    } else if (type == HWLOC_OBJ_OS_DEVICE) {
      assert(d == HWLOC_TYPE_DEPTH_OS_DEVICE);
      assert(hwloc_get_depth_type(topology, d) == HWLOC_OBJ_OS_DEVICE);
    } else if (type == HWLOC_OBJ_MISC) {
      assert(d == HWLOC_TYPE_DEPTH_MISC);
      assert(hwloc_get_depth_type(topology, d) == HWLOC_OBJ_MISC);
    } else {
      assert(d >=0 || d == HWLOC_TYPE_DEPTH_UNKNOWN || d == HWLOC_TYPE_DEPTH_MULTIPLE);
    }
  }

  /* top-level specific checks */
  assert(hwloc_get_nbobjs_by_depth(topology, 0) == 1);
  obj = hwloc_get_root_obj(topology);
  assert(obj);
  assert(!obj->parent);
  assert(obj->cpuset);
  assert(!obj->depth);

  /* check that allowed sets are larger than the main sets */
  if (topology->flags & HWLOC_TOPOLOGY_FLAG_INCLUDE_DISALLOWED) {
    assert(hwloc_bitmap_isincluded(topology->allowed_cpuset, obj->cpuset));
    assert(hwloc_bitmap_isincluded(topology->allowed_nodeset, obj->nodeset));
  } else {
    assert(hwloc_bitmap_isequal(topology->allowed_cpuset, obj->cpuset));
    assert(hwloc_bitmap_isequal(topology->allowed_nodeset, obj->nodeset));
  }

  /* check each level */
  for(j=0; j<depth; j++)
    hwloc__check_level(topology, j, NULL, NULL);
  for(j=0; j<HWLOC_NR_SLEVELS; j++)
    hwloc__check_level(topology, HWLOC_SLEVEL_TO_DEPTH(j), topology->slevels[j].first, topology->slevels[j].last);

  /* recurse and check the tree of children, and type-specific checks */
  gp_indexes = hwloc_bitmap_alloc(); /* TODO prealloc to topology->next_gp_index */
  hwloc__check_object(topology, gp_indexes, obj);
  hwloc_bitmap_free(gp_indexes);

  /* recurse and check the nodesets of children */
  set = hwloc_bitmap_alloc();
  hwloc__check_nodesets(topology, obj, set);
  hwloc_bitmap_free(set);
}

#else /* NDEBUG */

void
hwloc_topology_check(struct hwloc_topology *topology __hwloc_attribute_unused)
{
}

#endif /* NDEBUG */
