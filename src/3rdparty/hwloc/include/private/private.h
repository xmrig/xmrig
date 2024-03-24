/*
 * Copyright © 2009      CNRS
 * Copyright © 2009-2023 Inria.  All rights reserved.
 * Copyright © 2009-2012, 2020 Université Bordeaux
 * Copyright © 2009-2011 Cisco Systems, Inc.  All rights reserved.
 *
 * See COPYING in top-level directory.
 */

/* Internal types and helpers. */


#ifdef HWLOC_INSIDE_PLUGIN
/*
 * these declarations are internal only, they are not available to plugins
 * (many functions below are internal static symbols).
 */
#error This file should not be used in plugins
#endif


#ifndef HWLOC_PRIVATE_H
#define HWLOC_PRIVATE_H

#include "private/autogen/config.h"
#include "hwloc.h"
#include "hwloc/bitmap.h"
#include "private/components.h"
#include "private/misc.h"

#include <sys/types.h>
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif
#ifdef HAVE_STDINT_H
#include <stdint.h>
#endif
#ifdef HAVE_SYS_UTSNAME_H
#include <sys/utsname.h>
#endif
#include <string.h>

#define HWLOC_TOPOLOGY_ABI 0x20400 /* version of the layout of struct topology */

struct hwloc_internal_location_s {
  enum hwloc_location_type_e type;
  union {
    struct {
      hwloc_obj_t obj; /* cached between refreshes */
      uint64_t gp_index;
      hwloc_obj_type_t type;
    } object; /* if type == HWLOC_LOCATION_TYPE_OBJECT */
    hwloc_cpuset_t cpuset; /* if type == HWLOC_LOCATION_TYPE_CPUSET */
  } location;
};

/*****************************************************
 * WARNING:
 * changes below in this structure (and its children)
 * should cause a bump of HWLOC_TOPOLOGY_ABI.
 *****************************************************/

struct hwloc_topology {
  unsigned topology_abi;

  unsigned nb_levels;					/* Number of horizontal levels */
  unsigned nb_levels_allocated;				/* Number of levels allocated and zeroed in level_nbobjects and levels below */
  unsigned *level_nbobjects; 				/* Number of objects on each horizontal level */
  struct hwloc_obj ***levels;				/* Direct access to levels, levels[l = 0 .. nblevels-1][0..level_nbobjects[l]] */
  unsigned long flags;
  int type_depth[HWLOC_OBJ_TYPE_MAX];
  enum hwloc_type_filter_e type_filter[HWLOC_OBJ_TYPE_MAX];
  int is_thissystem;
  int is_loaded;
  int modified;                                         /* >0 if objects were added/removed recently, which means a reconnect is needed */
  hwloc_pid_t pid;                                      /* Process ID the topology is view from, 0 for self */
  void *userdata;
  uint64_t next_gp_index;

  void *adopted_shmem_addr;
  size_t adopted_shmem_length;

#define HWLOC_NR_SLEVELS 6
#define HWLOC_SLEVEL_NUMANODE 0
#define HWLOC_SLEVEL_BRIDGE 1
#define HWLOC_SLEVEL_PCIDEV 2
#define HWLOC_SLEVEL_OSDEV 3
#define HWLOC_SLEVEL_MISC 4
#define HWLOC_SLEVEL_MEMCACHE 5
  /* order must match negative depth, it's asserted in setup_defaults() */
#define HWLOC_SLEVEL_FROM_DEPTH(x) (HWLOC_TYPE_DEPTH_NUMANODE-(x))
#define HWLOC_SLEVEL_TO_DEPTH(x) (HWLOC_TYPE_DEPTH_NUMANODE-(x))
  struct hwloc_special_level_s {
    unsigned nbobjs;
    struct hwloc_obj **objs;
    struct hwloc_obj *first, *last; /* Temporarily used while listing object before building the objs array */
  } slevels[HWLOC_NR_SLEVELS];

  hwloc_bitmap_t allowed_cpuset;
  hwloc_bitmap_t allowed_nodeset;

  struct hwloc_binding_hooks {
    /* These are actually rather OS hooks since some of them are not about binding */
    int (*set_thisproc_cpubind)(hwloc_topology_t topology, hwloc_const_cpuset_t set, int flags);
    int (*get_thisproc_cpubind)(hwloc_topology_t topology, hwloc_cpuset_t set, int flags);
    int (*set_thisthread_cpubind)(hwloc_topology_t topology, hwloc_const_cpuset_t set, int flags);
    int (*get_thisthread_cpubind)(hwloc_topology_t topology, hwloc_cpuset_t set, int flags);
    int (*set_proc_cpubind)(hwloc_topology_t topology, hwloc_pid_t pid, hwloc_const_cpuset_t set, int flags);
    int (*get_proc_cpubind)(hwloc_topology_t topology, hwloc_pid_t pid, hwloc_cpuset_t set, int flags);
#ifdef hwloc_thread_t
    int (*set_thread_cpubind)(hwloc_topology_t topology, hwloc_thread_t tid, hwloc_const_cpuset_t set, int flags);
    int (*get_thread_cpubind)(hwloc_topology_t topology, hwloc_thread_t tid, hwloc_cpuset_t set, int flags);
#endif

    int (*get_thisproc_last_cpu_location)(hwloc_topology_t topology, hwloc_cpuset_t set, int flags);
    int (*get_thisthread_last_cpu_location)(hwloc_topology_t topology, hwloc_cpuset_t set, int flags);
    int (*get_proc_last_cpu_location)(hwloc_topology_t topology, hwloc_pid_t pid, hwloc_cpuset_t set, int flags);

    int (*set_thisproc_membind)(hwloc_topology_t topology, hwloc_const_nodeset_t nodeset, hwloc_membind_policy_t policy, int flags);
    int (*get_thisproc_membind)(hwloc_topology_t topology, hwloc_nodeset_t nodeset, hwloc_membind_policy_t * policy, int flags);
    int (*set_thisthread_membind)(hwloc_topology_t topology, hwloc_const_nodeset_t nodeset, hwloc_membind_policy_t policy, int flags);
    int (*get_thisthread_membind)(hwloc_topology_t topology, hwloc_nodeset_t nodeset, hwloc_membind_policy_t * policy, int flags);
    int (*set_proc_membind)(hwloc_topology_t topology, hwloc_pid_t pid, hwloc_const_nodeset_t nodeset, hwloc_membind_policy_t policy, int flags);
    int (*get_proc_membind)(hwloc_topology_t topology, hwloc_pid_t pid, hwloc_nodeset_t nodeset, hwloc_membind_policy_t * policy, int flags);
    int (*set_area_membind)(hwloc_topology_t topology, const void *addr, size_t len, hwloc_const_nodeset_t nodeset, hwloc_membind_policy_t policy, int flags);
    int (*get_area_membind)(hwloc_topology_t topology, const void *addr, size_t len, hwloc_nodeset_t nodeset, hwloc_membind_policy_t * policy, int flags);
    int (*get_area_memlocation)(hwloc_topology_t topology, const void *addr, size_t len, hwloc_nodeset_t nodeset, int flags);
    /* This has to return the same kind of pointer as alloc_membind, so that free_membind can be used on it */
    void *(*alloc)(hwloc_topology_t topology, size_t len);
    /* alloc_membind has to always succeed if !(flags & HWLOC_MEMBIND_STRICT).
     * see hwloc_alloc_or_fail which is convenient for that.  */
    void *(*alloc_membind)(hwloc_topology_t topology, size_t len, hwloc_const_nodeset_t nodeset, hwloc_membind_policy_t policy, int flags);
    int (*free_membind)(hwloc_topology_t topology, void *addr, size_t len);

    int (*get_allowed_resources)(hwloc_topology_t topology);
  } binding_hooks;

  struct hwloc_topology_support support;

  void (*userdata_export_cb)(void *reserved, struct hwloc_topology *topology, struct hwloc_obj *obj);
  void (*userdata_import_cb)(struct hwloc_topology *topology, struct hwloc_obj *obj, const char *name, const void *buffer, size_t length);
  int userdata_not_decoded;

  struct hwloc_internal_distances_s {
    char *name; /* FIXME: needs an API to set it from user */

    unsigned id; /* to match the container id field of public distances structure
		  * not exported to XML, regenerated during _add()
		  */

    /* if all objects have the same type, different_types is NULL and unique_type is valid.
     * otherwise unique_type is HWLOC_OBJ_TYPE_NONE and different_types contains individual objects types.
     */
    hwloc_obj_type_t unique_type;
    hwloc_obj_type_t *different_types;

    /* add union hwloc_obj_attr_u if we ever support groups */
    unsigned nbobjs;
    uint64_t *indexes; /* array of OS or GP indexes before we can convert them into objs.
			* OS indexes for distances covering only PUs or only NUMAnodes.
			*/
#define HWLOC_DIST_TYPE_USE_OS_INDEX(_type) ((_type) == HWLOC_OBJ_PU || (_type == HWLOC_OBJ_NUMANODE))
    uint64_t *values; /* distance matrices, ordered according to the above indexes/objs array.
		       * distance from i to j is stored in slot i*nbnodes+j.
		       */
    unsigned long kind;

#define HWLOC_INTERNAL_DIST_FLAG_OBJS_VALID (1U<<0) /* if the objs array is valid below */
#define HWLOC_INTERNAL_DIST_FLAG_NOT_COMMITTED (1U<<1) /* if the distances isn't in the list yet */
    unsigned iflags;

    /* objects are currently stored in physical_index order */
    hwloc_obj_t *objs; /* array of objects */

    struct hwloc_internal_distances_s *prev, *next;
  } *first_dist, *last_dist;
  unsigned next_dist_id;

  /* memory attributes */
  unsigned nr_memattrs;
  struct hwloc_internal_memattr_s {
    /* memattr info */
    char *name; /* TODO unit is implicit, in the documentation of standard attributes, or in the name? */
    unsigned long flags;
#define HWLOC_IMATTR_FLAG_STATIC_NAME (1U<<0) /* no need to free name */
#define HWLOC_IMATTR_FLAG_CACHE_VALID (1U<<1) /* target and initiator are valid */
#define HWLOC_IMATTR_FLAG_CONVENIENCE (1U<<2) /* convenience attribute reporting values from non-memattr attributes (R/O and no actual targets stored) */
    unsigned iflags;

    /* array of values */
    unsigned nr_targets;
    struct hwloc_internal_memattr_target_s {
      /* target object */
      hwloc_obj_t obj; /* cached between refreshes */
      hwloc_obj_type_t type;
      unsigned os_index; /* only used temporarily during discovery when there's no obj/gp_index yet */
      hwloc_uint64_t gp_index;

      /* value if there are no initiator for this attr */
      hwloc_uint64_t noinitiator_value;
      /* initiators otherwise */
      unsigned nr_initiators;
      struct hwloc_internal_memattr_initiator_s {
        struct hwloc_internal_location_s initiator;
        hwloc_uint64_t value;
      } *initiators;
    } *targets;
  } *memattrs;

  /* hybridcpus */
  unsigned nr_cpukinds;
  unsigned nr_cpukinds_allocated;
  struct hwloc_internal_cpukind_s {
    hwloc_cpuset_t cpuset;
#define HWLOC_CPUKIND_EFFICIENCY_UNKNOWN -1
    int efficiency;
    int forced_efficiency; /* returned by the hardware or OS if any */
    hwloc_uint64_t ranking_value; /* internal value for ranking */
    unsigned nr_infos;
    struct hwloc_info_s *infos;
  } *cpukinds;

  int grouping;
  int grouping_verbose;
  unsigned grouping_nbaccuracies;
  float grouping_accuracies[5];
  unsigned grouping_next_subkind;

  /* list of enabled backends. */
  struct hwloc_backend * backends;
  struct hwloc_backend * get_pci_busid_cpuset_backend; /* first backend that provides get_pci_busid_cpuset() callback */
  unsigned backend_phases;
  unsigned backend_excluded_phases;

  /* memory allocator for topology objects */
  struct hwloc_tma * tma;

/*****************************************************
 * WARNING:
 * changes above in this structure (and its children)
 * should cause a bump of HWLOC_TOPOLOGY_ABI.
 *****************************************************/

  /*
   * temporary variables during discovery
   */

  /* set to 1 at the beginning of load() if the filter of any cpu cache type (L1 to L3i) is not NONE,
   * may be checked by backends before querying caches
   * (when they don't know the level of caches they are querying).
   */
  int want_some_cpu_caches;

  /* machine-wide memory.
   * temporarily stored there by OSes that only provide this without NUMA information,
   * and actually used later by the core.
   */
  struct hwloc_numanode_attr_s machine_memory;

  /* pci stuff */
  int pci_has_forced_locality;
  unsigned pci_forced_locality_nr;
  struct hwloc_pci_forced_locality_s {
    unsigned domain;
    unsigned bus_first, bus_last;
    hwloc_bitmap_t cpuset;
  } * pci_forced_locality;
  hwloc_uint64_t pci_locality_quirks;

  /* component blacklisting */
  unsigned nr_blacklisted_components;
  struct hwloc_topology_forced_component_s {
    struct hwloc_disc_component *component;
    unsigned phases;
  } *blacklisted_components;

  /* FIXME: keep until topo destroy and reuse for finding specific buses */
  struct hwloc_pci_locality_s {
    unsigned domain;
    unsigned bus_min;
    unsigned bus_max;
    hwloc_bitmap_t cpuset;
    hwloc_obj_t parent;
    struct hwloc_pci_locality_s *prev, *next;
  } *first_pci_locality, *last_pci_locality;
};

extern void hwloc_alloc_root_sets(hwloc_obj_t root);
extern void hwloc_setup_pu_level(struct hwloc_topology *topology, unsigned nb_pus);
extern int hwloc_get_sysctlbyname(const char *name, int64_t *n);
extern int hwloc_get_sysctl(int name[], unsigned namelen, int64_t *n);

/* returns the number of CPU from the OS (only valid if thissystem) */
#define HWLOC_FALLBACK_NBPROCESSORS_INCLUDE_OFFLINE 1 /* by default we try to get only the online CPUs */
extern int hwloc_fallback_nbprocessors(unsigned flags);
/* returns the memory size from the OS (only valid if thissystem) */
extern int64_t hwloc_fallback_memsize(void);

extern int hwloc__object_cpusets_compare_first(hwloc_obj_t obj1, hwloc_obj_t obj2);
extern void hwloc__reorder_children(hwloc_obj_t parent);

extern void hwloc_topology_setup_defaults(struct hwloc_topology *topology);
extern void hwloc_topology_clear(struct hwloc_topology *topology);

/* insert memory object as memory child of normal parent */
extern struct hwloc_obj * hwloc__attach_memory_object(struct hwloc_topology *topology, hwloc_obj_t parent,
                                                      hwloc_obj_t obj, const char *reason);

extern hwloc_obj_t hwloc_get_obj_by_type_and_gp_index(hwloc_topology_t topology, hwloc_obj_type_t type, uint64_t gp_index);

extern void hwloc_pci_discovery_init(struct hwloc_topology *topology);
extern void hwloc_pci_discovery_prepare(struct hwloc_topology *topology);
extern void hwloc_pci_discovery_exit(struct hwloc_topology *topology);

/* Look for an object matching complete cpuset exactly, or insert one.
 * Return NULL on failure.
 * Return a good fallback (object above) on failure to insert.
 */
extern hwloc_obj_t hwloc_find_insert_io_parent_by_complete_cpuset(struct hwloc_topology *topology, hwloc_cpuset_t cpuset);

extern int hwloc__add_info(struct hwloc_info_s **infosp, unsigned *countp, const char *name, const char *value);
extern int hwloc__add_info_nodup(struct hwloc_info_s **infosp, unsigned *countp, const char *name, const char *value, int replace);
extern int hwloc__move_infos(struct hwloc_info_s **dst_infosp, unsigned *dst_countp, struct hwloc_info_s **src_infosp, unsigned *src_countp);
extern int hwloc__tma_dup_infos(struct hwloc_tma *tma, struct hwloc_info_s **dst_infosp, unsigned *dst_countp, struct hwloc_info_s *src_infos, unsigned src_count);
extern void hwloc__free_infos(struct hwloc_info_s *infos, unsigned count);

/* set native OS binding hooks */
extern void hwloc_set_native_binding_hooks(struct hwloc_binding_hooks *hooks, struct hwloc_topology_support *support);
/* set either native OS binding hooks (if thissystem), or dummy ones */
extern void hwloc_set_binding_hooks(struct hwloc_topology *topology);

#if defined(HWLOC_LINUX_SYS)
extern void hwloc_set_linuxfs_hooks(struct hwloc_binding_hooks *binding_hooks, struct hwloc_topology_support *support);
#endif /* HWLOC_LINUX_SYS */

#if defined(HWLOC_BGQ_SYS)
extern void hwloc_set_bgq_hooks(struct hwloc_binding_hooks *binding_hooks, struct hwloc_topology_support *support);
#endif /* HWLOC_BGQ_SYS */

#ifdef HWLOC_SOLARIS_SYS
extern void hwloc_set_solaris_hooks(struct hwloc_binding_hooks *binding_hooks, struct hwloc_topology_support *support);
#endif /* HWLOC_SOLARIS_SYS */

#ifdef HWLOC_AIX_SYS
extern void hwloc_set_aix_hooks(struct hwloc_binding_hooks *binding_hooks, struct hwloc_topology_support *support);
#endif /* HWLOC_AIX_SYS */

#ifdef HWLOC_WIN_SYS
extern void hwloc_set_windows_hooks(struct hwloc_binding_hooks *binding_hooks, struct hwloc_topology_support *support);
#endif /* HWLOC_WIN_SYS */

#ifdef HWLOC_DARWIN_SYS
extern void hwloc_set_darwin_hooks(struct hwloc_binding_hooks *binding_hooks, struct hwloc_topology_support *support);
#endif /* HWLOC_DARWIN_SYS */

#ifdef HWLOC_FREEBSD_SYS
extern void hwloc_set_freebsd_hooks(struct hwloc_binding_hooks *binding_hooks, struct hwloc_topology_support *support);
#endif /* HWLOC_FREEBSD_SYS */

#ifdef HWLOC_NETBSD_SYS
extern void hwloc_set_netbsd_hooks(struct hwloc_binding_hooks *binding_hooks, struct hwloc_topology_support *support);
#endif /* HWLOC_NETBSD_SYS */

#ifdef HWLOC_HPUX_SYS
extern void hwloc_set_hpux_hooks(struct hwloc_binding_hooks *binding_hooks, struct hwloc_topology_support *support);
#endif /* HWLOC_HPUX_SYS */

extern int hwloc_look_hardwired_fujitsu_k(struct hwloc_topology *topology);
extern int hwloc_look_hardwired_fujitsu_fx10(struct hwloc_topology *topology);
extern int hwloc_look_hardwired_fujitsu_fx100(struct hwloc_topology *topology);

/* Insert uname-specific names/values in the object infos array.
 * If cached_uname isn't NULL, it is used as a struct utsname instead of recalling uname.
 * Any field that starts with \0 is ignored.
 */
extern void hwloc_add_uname_info(struct hwloc_topology *topology, void *cached_uname);

/* Free obj and its attributes assuming it's not linked to a parent and doesn't have any child */
extern void hwloc_free_unlinked_object(hwloc_obj_t obj);

/* Free obj and its children, assuming it's not linked to a parent */
extern void hwloc_free_object_and_children(hwloc_obj_t obj);

/* Free obj, its next siblings, and their children, assuming they're not linked to a parent */
extern void hwloc_free_object_siblings_and_children(hwloc_obj_t obj);

/* This can be used for the alloc field to get allocated data that can be freed by free() */
void *hwloc_alloc_heap(hwloc_topology_t topology, size_t len);

/* This can be used for the alloc field to get allocated data that can be freed by munmap() */
void *hwloc_alloc_mmap(hwloc_topology_t topology, size_t len);

/* This can be used for the free_membind field to free data using free() */
int hwloc_free_heap(hwloc_topology_t topology, void *addr, size_t len);

/* This can be used for the free_membind field to free data using munmap() */
int hwloc_free_mmap(hwloc_topology_t topology, void *addr, size_t len);

/* Allocates unbound memory or fail, depending on whether STRICT is requested
 * or not */
static __hwloc_inline void *
hwloc_alloc_or_fail(hwloc_topology_t topology, size_t len, int flags)
{
  if (flags & HWLOC_MEMBIND_STRICT)
    return NULL;
  return hwloc_alloc(topology, len);
}

extern void hwloc_internal_distances_init(hwloc_topology_t topology);
extern void hwloc_internal_distances_prepare(hwloc_topology_t topology);
extern void hwloc_internal_distances_destroy(hwloc_topology_t topology);
extern int hwloc_internal_distances_dup(hwloc_topology_t new, hwloc_topology_t old);
extern void hwloc_internal_distances_refresh(hwloc_topology_t topology);
extern void hwloc_internal_distances_invalidate_cached_objs(hwloc_topology_t topology);

/* these distances_add() functions are higher-level than those in hwloc/plugins.h
 * but they may change in the future, hence they are not exported to plugins.
 */
extern int hwloc_internal_distances_add_by_index(hwloc_topology_t topology, const char *name, hwloc_obj_type_t unique_type, hwloc_obj_type_t *different_types, unsigned nbobjs, uint64_t *indexes, uint64_t *values, unsigned long kind, unsigned long flags);
extern int hwloc_internal_distances_add(hwloc_topology_t topology, const char *name, unsigned nbobjs, hwloc_obj_t *objs, uint64_t *values, unsigned long kind, unsigned long flags);

extern void hwloc_internal_memattrs_init(hwloc_topology_t topology);
extern void hwloc_internal_memattrs_prepare(hwloc_topology_t topology);
extern void hwloc_internal_memattrs_destroy(hwloc_topology_t topology);
extern void hwloc_internal_memattrs_need_refresh(hwloc_topology_t topology);
extern void hwloc_internal_memattrs_refresh(hwloc_topology_t topology);
extern int hwloc_internal_memattrs_dup(hwloc_topology_t new, hwloc_topology_t old);
extern int hwloc_internal_memattr_set_value(hwloc_topology_t topology, hwloc_memattr_id_t id, hwloc_obj_type_t target_type, hwloc_uint64_t target_gp_index, unsigned target_os_index, struct hwloc_internal_location_s *initiator, hwloc_uint64_t value);
extern int hwloc_internal_memattrs_guess_memory_tiers(hwloc_topology_t topology, int force_subtype);

extern void hwloc_internal_cpukinds_init(hwloc_topology_t topology);
extern int hwloc_internal_cpukinds_rank(hwloc_topology_t topology);
extern void hwloc_internal_cpukinds_destroy(hwloc_topology_t topology);
extern int hwloc_internal_cpukinds_dup(hwloc_topology_t new, hwloc_topology_t old);
#define HWLOC_CPUKINDS_REGISTER_FLAG_OVERWRITE_FORCED_EFFICIENCY (1<<0)
extern int hwloc_internal_cpukinds_register(hwloc_topology_t topology, hwloc_cpuset_t cpuset, int forced_efficiency, const struct hwloc_info_s *infos, unsigned nr_infos, unsigned long flags);
extern void hwloc_internal_cpukinds_restrict(hwloc_topology_t topology);

/* encode src buffer into target buffer.
 * targsize must be at least 4*((srclength+2)/3)+1.
 * target will be 0-terminated.
 */
extern int hwloc_encode_to_base64(const char *src, size_t srclength, char *target, size_t targsize);
/* decode src buffer into target buffer.
 * src is 0-terminated.
 * targsize must be at least srclength*3/4+1 (srclength not including \0)
 * but only srclength*3/4 characters will be meaningful
 * (the next one may be partially written during decoding, but it should be ignored).
 */
extern int hwloc_decode_from_base64(char const *src, char *target, size_t targsize);

/* On some systems, snprintf returns the size of written data, not the actually
 * required size. Sometimes it returns -1 on truncation too.
 * And sometimes it doesn't like NULL output buffers.
 * http://www.gnu.org/software/gnulib/manual/html_node/snprintf.html
 *
 * hwloc_snprintf behaves properly, but it's a bit overkill on the vast majority
 * of platforms, so don't enable it unless really needed.
 */
#ifdef HWLOC_HAVE_CORRECT_SNPRINTF
#define hwloc_snprintf snprintf
#else
extern int hwloc_snprintf(char *str, size_t size, const char *format, ...) __hwloc_attribute_format(printf, 3, 4);
#endif

/* Return the name of the currently running program, if supported.
 * If not NULL, must be freed by the caller.
 */
extern char * hwloc_progname(struct hwloc_topology *topology);

/* obj->attr->group.kind internal values.
 * the core will keep the smallest ones when merging two groups,
 * that's why user-given kinds are first.
 */
/* first, user-given groups, should remain as long as possible */
#define HWLOC_GROUP_KIND_USER				0	/* user-given, user may use subkind too */
#define HWLOC_GROUP_KIND_SYNTHETIC			10	/* subkind is group depth within synthetic description */
/* then, hardware-specific groups */
#define HWLOC_GROUP_KIND_INTEL_KNL_SUBNUMA_CLUSTER	100	/* no subkind */
#define HWLOC_GROUP_KIND_INTEL_EXTTOPOENUM_UNKNOWN	101	/* subkind is unknown level */
#define HWLOC_GROUP_KIND_INTEL_MODULE			102	/* no subkind */
#define HWLOC_GROUP_KIND_INTEL_TILE			103	/* no subkind */
#define HWLOC_GROUP_KIND_INTEL_DIE			104	/* no subkind */
#define HWLOC_GROUP_KIND_S390_BOOK			110	/* subkind 0 is book, subkind 1 is drawer (group of books) */
#define HWLOC_GROUP_KIND_AMD_COMPUTE_UNIT		120	/* no subkind */
#define HWLOC_GROUP_KIND_AMD_COMPLEX                    121     /* no subkind */
/* then, OS-specific groups */
#define HWLOC_GROUP_KIND_SOLARIS_PG_HW_PERF		200	/* subkind is group width */
#define HWLOC_GROUP_KIND_AIX_SDL_UNKNOWN		210	/* subkind is SDL level */
#define HWLOC_GROUP_KIND_WINDOWS_PROCESSOR_GROUP	220	/* no subkind */
#define HWLOC_GROUP_KIND_WINDOWS_RELATIONSHIP_UNKNOWN	221	/* no subkind */
#define HWLOC_GROUP_KIND_LINUX_CLUSTER                  222     /* no subkind */
/* distance groups */
#define HWLOC_GROUP_KIND_DISTANCE			900	/* subkind is round of adding these groups during distance based grouping */
/* finally, hwloc-specific groups required to insert something else, should disappear as soon as possible */
#define HWLOC_GROUP_KIND_IO				1000	/* no subkind */
#define HWLOC_GROUP_KIND_MEMORY				1001	/* no subkind */

/* memory allocator for topology objects */
struct hwloc_tma {
  void * (*malloc)(struct hwloc_tma *, size_t);
  void *data;
  int dontfree; /* when set, free() or realloc() cannot be used, and tma->malloc() cannot fail */
};

static __hwloc_inline void *
hwloc_tma_malloc(struct hwloc_tma *tma,
		 size_t size)
{
  if (tma) {
    return tma->malloc(tma, size);
  } else {
    return malloc(size);
  }
}

static __hwloc_inline void *
hwloc_tma_calloc(struct hwloc_tma *tma,
		 size_t size)
{
  char *ptr = hwloc_tma_malloc(tma, size);
  if (ptr)
    memset(ptr, 0, size);
  return ptr;
}

static __hwloc_inline char *
hwloc_tma_strdup(struct hwloc_tma *tma,
		 const char *src)
{
  size_t len = strlen(src);
  char *ptr = hwloc_tma_malloc(tma, len+1);
  if (ptr)
    memcpy(ptr, src, len+1);
  return ptr;
}

/* bitmap allocator to be used inside hwloc */
extern hwloc_bitmap_t hwloc_bitmap_tma_dup(struct hwloc_tma *tma, hwloc_const_bitmap_t old);

extern int hwloc__topology_dup(hwloc_topology_t *newp, hwloc_topology_t old, struct hwloc_tma *tma);
extern void hwloc__topology_disadopt(hwloc_topology_t  topology);

#endif /* HWLOC_PRIVATE_H */
