/*
 * Copyright © 2009 CNRS
 * Copyright © 2009-2021 Inria.  All rights reserved.
 * Copyright © 2009-2012 Université Bordeaux
 * Copyright © 2009-2020 Cisco Systems, Inc.  All rights reserved.
 * See COPYING in top-level directory.
 */

/*=====================================================================
 *                 PLEASE GO READ THE DOCUMENTATION!
 *         ------------------------------------------------
 *               $tarball_directory/doc/doxygen-doc/
 *                                or
 *           https://www.open-mpi.org/projects/hwloc/doc/
 *=====================================================================
 *
 * FAIR WARNING: Do NOT expect to be able to figure out all the
 * subtleties of hwloc by simply reading function prototypes and
 * constant descrptions here in this file.
 *
 * Hwloc has wonderful documentation in both PDF and HTML formats for
 * your reading pleasure.  The formal documentation explains a LOT of
 * hwloc-specific concepts, provides definitions, and discusses the
 * "big picture" for many of the things that you'll find here in this
 * header file.
 *
 * The PDF/HTML documentation was generated via Doxygen; much of what
 * you'll see in there is also here in this file.  BUT THERE IS A LOT
 * THAT IS IN THE PDF/HTML THAT IS ***NOT*** IN hwloc.h!
 *
 * There are entire paragraph-length descriptions, discussions, and
 * pretty prictures to explain subtle corner cases, provide concrete
 * examples, etc.
 *
 * Please, go read the documentation.  :-)
 *
 * Moreover there are several examples of hwloc use under doc/examples
 * in the source tree.
 *
 *=====================================================================*/

/** \file
 * \brief The hwloc API.
 *
 * See hwloc/bitmap.h for bitmap specific macros.
 * See hwloc/helper.h for high-level topology traversal helpers.
 * See hwloc/inlines.h for the actual inline code of some functions below.
 * See hwloc/export.h for exporting topologies to XML or to synthetic descriptions.
 * See hwloc/distances.h for querying and modifying distances between objects.
 * See hwloc/diff.h for manipulating differences between similar topologies.
 */

#ifndef HWLOC_H
#define HWLOC_H

#include "hwloc/autogen/config.h"

#include <sys/types.h>
#include <stdio.h>
#include <string.h>
#include <limits.h>

/*
 * Symbol transforms
 */
#include "hwloc/rename.h"

/*
 * Bitmap definitions
 */

#include "hwloc/bitmap.h"


#ifdef __cplusplus
extern "C" {
#endif


/** \defgroup hwlocality_api_version API version
 * @{
 */

/** \brief Indicate at build time which hwloc API version is being used.
 *
 * This number is updated to (X<<16)+(Y<<8)+Z when a new release X.Y.Z
 * actually modifies the API.
 *
 * Users may check for available features at build time using this number
 * (see \ref faq_version_api).
 *
 * \note This should not be confused with HWLOC_VERSION, the library version.
 * Two stable releases of the same series usually have the same ::HWLOC_API_VERSION
 * even if their HWLOC_VERSION are different.
 */
#define HWLOC_API_VERSION 0x00020400

/** \brief Indicate at runtime which hwloc API version was used at build time.
 *
 * Should be ::HWLOC_API_VERSION if running on the same version.
 */
HWLOC_DECLSPEC unsigned hwloc_get_api_version(void);

/** \brief Current component and plugin ABI version (see hwloc/plugins.h) */
#define HWLOC_COMPONENT_ABI 7

/** @} */



/** \defgroup hwlocality_object_sets Object Sets (hwloc_cpuset_t and hwloc_nodeset_t)
 *
 * Hwloc uses bitmaps to represent two distinct kinds of object sets:
 * CPU sets (::hwloc_cpuset_t) and NUMA node sets (::hwloc_nodeset_t).
 * These types are both typedefs to a common back end type
 * (::hwloc_bitmap_t), and therefore all the hwloc bitmap functions
 * are applicable to both ::hwloc_cpuset_t and ::hwloc_nodeset_t (see
 * \ref hwlocality_bitmap).
 *
 * The rationale for having two different types is that even though
 * the actions one wants to perform on these types are the same (e.g.,
 * enable and disable individual items in the set/mask), they're used
 * in very different contexts: one for specifying which processors to
 * use and one for specifying which NUMA nodes to use.  Hence, the
 * name difference is really just to reflect the intent of where the
 * type is used.
 *
 * @{
 */

/** \brief A CPU set is a bitmap whose bits are set according to CPU
 * physical OS indexes.
 *
 * It may be consulted and modified with the bitmap API as any
 * ::hwloc_bitmap_t (see hwloc/bitmap.h).
 *
 * Each bit may be converted into a PU object using
 * hwloc_get_pu_obj_by_os_index().
 */
typedef hwloc_bitmap_t hwloc_cpuset_t;
/** \brief A non-modifiable ::hwloc_cpuset_t. */
typedef hwloc_const_bitmap_t hwloc_const_cpuset_t;

/** \brief A node set is a bitmap whose bits are set according to NUMA
 * memory node physical OS indexes.
 *
 * It may be consulted and modified with the bitmap API as any
 * ::hwloc_bitmap_t (see hwloc/bitmap.h).
 * Each bit may be converted into a NUMA node object using
 * hwloc_get_numanode_obj_by_os_index().
 *
 * When binding memory on a system without any NUMA node,
 * the single main memory bank is considered as NUMA node #0.
 *
 * See also \ref hwlocality_helper_nodeset_convert.
 */
typedef hwloc_bitmap_t hwloc_nodeset_t;
/** \brief A non-modifiable ::hwloc_nodeset_t.
 */
typedef hwloc_const_bitmap_t hwloc_const_nodeset_t;

/** @} */



/** \defgroup hwlocality_object_types Object Types
 * @{
 */

/** \brief Type of topology object.
 *
 * \note Do not rely on the ordering or completeness of the values as new ones
 * may be defined in the future!  If you need to compare types, use
 * hwloc_compare_types() instead.
 */
typedef enum {

/** \cond */
#define HWLOC_OBJ_TYPE_MIN HWLOC_OBJ_MACHINE /* Sentinel value */
/** \endcond */

  HWLOC_OBJ_MACHINE,	/**< \brief Machine.
			  * A set of processors and memory with cache
			  * coherency.
			  *
			  * This type is always used for the root object of a topology,
			  * and never used anywhere else.
			  * Hence its parent is always \c NULL.
			  */

  HWLOC_OBJ_PACKAGE,	/**< \brief Physical package.
			  * The physical package that usually gets inserted
			  * into a socket on the motherboard.
			  * A processor package usually contains multiple cores,
			  * and possibly some dies.
			  */
  HWLOC_OBJ_CORE,	/**< \brief Core.
			  * A computation unit (may be shared by several
			  * PUs, aka logical processors).
			  */
  HWLOC_OBJ_PU,		/**< \brief Processing Unit, or (Logical) Processor.
			  * An execution unit (may share a core with some
			  * other logical processors, e.g. in the case of
			  * an SMT core).
			  *
			  * This is the smallest object representing CPU resources,
			  * it cannot have any child except Misc objects.
			  *
			  * Objects of this kind are always reported and can
			  * thus be used as fallback when others are not.
			  */

  HWLOC_OBJ_L1CACHE,	/**< \brief Level 1 Data (or Unified) Cache. */
  HWLOC_OBJ_L2CACHE,	/**< \brief Level 2 Data (or Unified) Cache. */
  HWLOC_OBJ_L3CACHE,	/**< \brief Level 3 Data (or Unified) Cache. */
  HWLOC_OBJ_L4CACHE,	/**< \brief Level 4 Data (or Unified) Cache. */
  HWLOC_OBJ_L5CACHE,	/**< \brief Level 5 Data (or Unified) Cache. */

  HWLOC_OBJ_L1ICACHE,	/**< \brief Level 1 instruction Cache (filtered out by default). */
  HWLOC_OBJ_L2ICACHE,	/**< \brief Level 2 instruction Cache (filtered out by default). */
  HWLOC_OBJ_L3ICACHE,	/**< \brief Level 3 instruction Cache (filtered out by default). */

  HWLOC_OBJ_GROUP,	/**< \brief Group objects.
			  * Objects which do not fit in the above but are
			  * detected by hwloc and are useful to take into
			  * account for affinity. For instance, some operating systems
			  * expose their arbitrary processors aggregation this
			  * way.  And hwloc may insert such objects to group
			  * NUMA nodes according to their distances.
			  * See also \ref faq_groups.
			  *
			  * These objects are removed when they do not bring
			  * any structure (see ::HWLOC_TYPE_FILTER_KEEP_STRUCTURE).
			  */

  HWLOC_OBJ_NUMANODE,	/**< \brief NUMA node.
			  * An object that contains memory that is directly
			  * and byte-accessible to the host processors.
			  * It is usually close to some cores (the corresponding objects
			  * are descendants of the NUMA node object in the hwloc tree).
			  *
			  * This is the smallest object representing Memory resources,
			  * it cannot have any child except Misc objects.
			  * However it may have Memory-side cache parents.
			  *
			  * There is always at least one such object in the topology
			  * even if the machine is not NUMA.
			  *
			  * Memory objects are not listed in the main children list,
			  * but rather in the dedicated Memory children list.
			  *
			  * NUMA nodes have a special depth ::HWLOC_TYPE_DEPTH_NUMANODE
			  * instead of a normal depth just like other objects in the
			  * main tree.
			  */

  HWLOC_OBJ_BRIDGE,	/**< \brief Bridge (filtered out by default).
			  * Any bridge (or PCI switch) that connects the host or an I/O bus,
			  * to another I/O bus.
			  *
			  * Bridges are not added to the topology unless their
			  * filtering is changed (see hwloc_topology_set_type_filter()
			  * and hwloc_topology_set_io_types_filter()).
			  *
			  * I/O objects are not listed in the main children list,
			  * but rather in the dedicated io children list.
			  * I/O objects have NULL CPU and node sets.
			  */
  HWLOC_OBJ_PCI_DEVICE,	/**< \brief PCI device (filtered out by default).
			  *
			  * PCI devices are not added to the topology unless their
			  * filtering is changed (see hwloc_topology_set_type_filter()
			  * and hwloc_topology_set_io_types_filter()).
			  *
			  * I/O objects are not listed in the main children list,
			  * but rather in the dedicated io children list.
			  * I/O objects have NULL CPU and node sets.
			  */
  HWLOC_OBJ_OS_DEVICE,	/**< \brief Operating system device (filtered out by default).
			  *
			  * OS devices are not added to the topology unless their
			  * filtering is changed (see hwloc_topology_set_type_filter()
			  * and hwloc_topology_set_io_types_filter()).
			  *
			  * I/O objects are not listed in the main children list,
			  * but rather in the dedicated io children list.
			  * I/O objects have NULL CPU and node sets.
			  */

  HWLOC_OBJ_MISC,	/**< \brief Miscellaneous objects (filtered out by default).
			  * Objects without particular meaning, that can e.g. be
			  * added by the application for its own use, or by hwloc
			  * for miscellaneous objects such as MemoryModule (DIMMs).
			  *
			  * They are not added to the topology unless their filtering
			  * is changed (see hwloc_topology_set_type_filter()).
			  *
			  * These objects are not listed in the main children list,
			  * but rather in the dedicated misc children list.
			  * Misc objects may only have Misc objects as children,
			  * and those are in the dedicated misc children list as well.
			  * Misc objects have NULL CPU and node sets.
			  */

  HWLOC_OBJ_MEMCACHE,	/**< \brief Memory-side cache (filtered out by default).
			  * A cache in front of a specific NUMA node.
			  *
			  * This object always has at least one NUMA node as a memory child.
			  *
			  * Memory objects are not listed in the main children list,
			  * but rather in the dedicated Memory children list.
			  *
			  * Memory-side cache have a special depth ::HWLOC_TYPE_DEPTH_MEMCACHE
			  * instead of a normal depth just like other objects in the
			  * main tree.
			  */

  HWLOC_OBJ_DIE,	/**< \brief Die within a physical package.
			 * A subpart of the physical package, that contains multiple cores.
			 */

  HWLOC_OBJ_TYPE_MAX    /**< \private Sentinel value */
} hwloc_obj_type_t;

/** \brief Cache type. */
typedef enum hwloc_obj_cache_type_e {
  HWLOC_OBJ_CACHE_UNIFIED,      /**< \brief Unified cache. */
  HWLOC_OBJ_CACHE_DATA,         /**< \brief Data cache. */
  HWLOC_OBJ_CACHE_INSTRUCTION   /**< \brief Instruction cache (filtered out by default). */
} hwloc_obj_cache_type_t;

/** \brief Type of one side (upstream or downstream) of an I/O bridge. */
typedef enum hwloc_obj_bridge_type_e {
  HWLOC_OBJ_BRIDGE_HOST,	/**< \brief Host-side of a bridge, only possible upstream. */
  HWLOC_OBJ_BRIDGE_PCI		/**< \brief PCI-side of a bridge. */
} hwloc_obj_bridge_type_t;

/** \brief Type of a OS device. */
typedef enum hwloc_obj_osdev_type_e {
  HWLOC_OBJ_OSDEV_BLOCK,	/**< \brief Operating system block device, or non-volatile memory device.
				  * For instance "sda" or "dax2.0" on Linux. */
  HWLOC_OBJ_OSDEV_GPU,		/**< \brief Operating system GPU device.
				  * For instance ":0.0" for a GL display,
				  * "card0" for a Linux DRM device. */
  HWLOC_OBJ_OSDEV_NETWORK,	/**< \brief Operating system network device.
				  * For instance the "eth0" interface on Linux. */
  HWLOC_OBJ_OSDEV_OPENFABRICS,	/**< \brief Operating system openfabrics device.
				  * For instance the "mlx4_0" InfiniBand HCA,
				  * or "hfi1_0" Omni-Path interface on Linux. */
  HWLOC_OBJ_OSDEV_DMA,		/**< \brief Operating system dma engine device.
				  * For instance the "dma0chan0" DMA channel on Linux. */
  HWLOC_OBJ_OSDEV_COPROC	/**< \brief Operating system co-processor device.
				  * For instance "opencl0d0" for a OpenCL device,
				  * "cuda0" for a CUDA device. */
} hwloc_obj_osdev_type_t;

/** \brief Compare the depth of two object types
 *
 * Types shouldn't be compared as they are, since newer ones may be added in
 * the future.  This function returns less than, equal to, or greater than zero
 * respectively if \p type1 objects usually include \p type2 objects, are the
 * same as \p type2 objects, or are included in \p type2 objects. If the types
 * can not be compared (because neither is usually contained in the other),
 * ::HWLOC_TYPE_UNORDERED is returned.  Object types containing CPUs can always
 * be compared (usually, a system contains machines which contain nodes which
 * contain packages which contain caches, which contain cores, which contain
 * processors).
 *
 * \note ::HWLOC_OBJ_PU will always be the deepest,
 * while ::HWLOC_OBJ_MACHINE is always the highest.
 *
 * \note This does not mean that the actual topology will respect that order:
 * e.g. as of today cores may also contain caches, and packages may also contain
 * nodes. This is thus just to be seen as a fallback comparison method.
 */
HWLOC_DECLSPEC int hwloc_compare_types (hwloc_obj_type_t type1, hwloc_obj_type_t type2) __hwloc_attribute_const;

/** \brief Value returned by hwloc_compare_types() when types can not be compared. \hideinitializer */
#define HWLOC_TYPE_UNORDERED INT_MAX

/** @} */



/** \defgroup hwlocality_objects Object Structure and Attributes
 * @{
 */

union hwloc_obj_attr_u;

/** \brief Structure of a topology object
 *
 * Applications must not modify any field except \p hwloc_obj.userdata.
 */
struct hwloc_obj {
  /* physical information */
  hwloc_obj_type_t type;		/**< \brief Type of object */
  char *subtype;			/**< \brief Subtype string to better describe the type field. */

  unsigned os_index;			/**< \brief OS-provided physical index number.
					 * It is not guaranteed unique across the entire machine,
					 * except for PUs and NUMA nodes.
					 * Set to HWLOC_UNKNOWN_INDEX if unknown or irrelevant for this object.
					 */
#define HWLOC_UNKNOWN_INDEX (unsigned)-1

  char *name;				/**< \brief Object-specific name if any.
					 * Mostly used for identifying OS devices and Misc objects where
					 * a name string is more useful than numerical indexes.
					 */

  hwloc_uint64_t total_memory; /**< \brief Total memory (in bytes) in NUMA nodes below this object. */

  union hwloc_obj_attr_u *attr;		/**< \brief Object type-specific Attributes,
					 * may be \c NULL if no attribute value was found */

  /* global position */
  int depth;				/**< \brief Vertical index in the hierarchy.
					 *
					 * For normal objects, this is the depth of the horizontal level
					 * that contains this object and its cousins of the same type.
					 * If the topology is symmetric, this is equal to the parent depth
					 * plus one, and also equal to the number of parent/child links
					 * from the root object to here.
					 *
					 * For special objects (NUMA nodes, I/O and Misc) that are not
					 * in the main tree, this is a special negative value that
					 * corresponds to their dedicated level,
					 * see hwloc_get_type_depth() and ::hwloc_get_type_depth_e.
					 * Those special values can be passed to hwloc functions such
					 * hwloc_get_nbobjs_by_depth() as usual.
					 */
  unsigned logical_index;		/**< \brief Horizontal index in the whole list of similar objects,
					 * hence guaranteed unique across the entire machine.
					 * Could be a "cousin_rank" since it's the rank within the "cousin" list below
					 * Note that this index may change when restricting the topology
					 * or when inserting a group.
					 */

  /* cousins are all objects of the same type (and depth) across the entire topology */
  struct hwloc_obj *next_cousin;	/**< \brief Next object of same type and depth */
  struct hwloc_obj *prev_cousin;	/**< \brief Previous object of same type and depth */

  /* children of the same parent are siblings, even if they may have different type and depth */
  struct hwloc_obj *parent;		/**< \brief Parent, \c NULL if root (Machine object) */
  unsigned sibling_rank;		/**< \brief Index in parent's \c children[] array. Or the index in parent's Memory, I/O or Misc children list. */
  struct hwloc_obj *next_sibling;	/**< \brief Next object below the same parent (inside the same list of children). */
  struct hwloc_obj *prev_sibling;	/**< \brief Previous object below the same parent (inside the same list of children). */
  /** @name List and array of normal children below this object (except Memory, I/O and Misc children). */
  /**@{*/
  unsigned arity;			/**< \brief Number of normal children.
					 * Memory, Misc and I/O children are not listed here
					 * but rather in their dedicated children list.
					 */
  struct hwloc_obj **children;		/**< \brief Normal children, \c children[0 .. arity -1] */
  struct hwloc_obj *first_child;	/**< \brief First normal child */
  struct hwloc_obj *last_child;		/**< \brief Last normal child */
  /**@}*/

  int symmetric_subtree;		/**< \brief Set if the subtree of normal objects below this object is symmetric,
					  * which means all normal children and their children have identical subtrees.
					  *
					  * Memory, I/O and Misc children are ignored.
					  *
					  * If set in the topology root object, lstopo may export the topology
					  * as a synthetic string.
					  */

  /** @name List of Memory children below this object. */
  /**@{*/
  unsigned memory_arity;		/**< \brief Number of Memory children.
					 * These children are listed in \p memory_first_child.
					 */
  struct hwloc_obj *memory_first_child;	/**< \brief First Memory child.
					 * NUMA nodes and Memory-side caches are listed here
					 * (\p memory_arity and \p memory_first_child)
					 * instead of in the normal children list.
					 * See also hwloc_obj_type_is_memory().
					 *
					 * A memory hierarchy starts from a normal CPU-side object
					 * (e.g. Package) and ends with NUMA nodes as leaves.
					 * There might exist some memory-side caches between them
					 * in the middle of the memory subtree.
					 */
  /**@}*/

  /** @name List of I/O children below this object. */
  /**@{*/
  unsigned io_arity;			/**< \brief Number of I/O children.
					 * These children are listed in \p io_first_child.
					 */
  struct hwloc_obj *io_first_child;	/**< \brief First I/O child.
					 * Bridges, PCI and OS devices are listed here (\p io_arity and \p io_first_child)
					 * instead of in the normal children list.
					 * See also hwloc_obj_type_is_io().
					 */
  /**@}*/

  /** @name List of Misc children below this object. */
  /**@{*/
  unsigned misc_arity;			/**< \brief Number of Misc children.
					 * These children are listed in \p misc_first_child.
					 */
  struct hwloc_obj *misc_first_child;	/**< \brief First Misc child.
					 * Misc objects are listed here (\p misc_arity and \p misc_first_child)
					 * instead of in the normal children list.
					 */
  /**@}*/

  /* cpusets and nodesets */
  hwloc_cpuset_t cpuset;		/**< \brief CPUs covered by this object
                                          *
                                          * This is the set of CPUs for which there are PU objects in the topology
                                          * under this object, i.e. which are known to be physically contained in this
                                          * object and known how (the children path between this object and the PU
                                          * objects).
                                          *
                                          * If the ::HWLOC_TOPOLOGY_FLAG_INCLUDE_DISALLOWED configuration flag is set,
                                          * some of these CPUs may not be allowed for binding,
                                          * see hwloc_topology_get_allowed_cpuset().
                                          *
					  * \note All objects have non-NULL CPU and node sets except Misc and I/O objects.
					  *
                                          * \note Its value must not be changed, hwloc_bitmap_dup() must be used instead.
                                          */
  hwloc_cpuset_t complete_cpuset;       /**< \brief The complete CPU set of processors of this object,
                                          *
                                          * This may include not only the same as the cpuset field, but also some CPUs for
                                          * which topology information is unknown or incomplete, some offlines CPUs, and
                                          * the CPUs that are ignored when the ::HWLOC_TOPOLOGY_FLAG_INCLUDE_DISALLOWED flag
                                          * is not set.
                                          * Thus no corresponding PU object may be found in the topology, because the
                                          * precise position is undefined. It is however known that it would be somewhere
                                          * under this object.
                                          *
                                          * \note Its value must not be changed, hwloc_bitmap_dup() must be used instead.
                                          */

  hwloc_nodeset_t nodeset;              /**< \brief NUMA nodes covered by this object or containing this object
                                          *
                                          * This is the set of NUMA nodes for which there are NUMA node objects in the
                                          * topology under or above this object, i.e. which are known to be physically
                                          * contained in this object or containing it and known how (the children path
                                          * between this object and the NUMA node objects).
                                          *
                                          * In the end, these nodes are those that are close to the current object.
                                          * Function hwloc_get_local_numanode_objs() may be used to list those NUMA
                                          * nodes more precisely.
                                          *
                                          * If the ::HWLOC_TOPOLOGY_FLAG_INCLUDE_DISALLOWED configuration flag is set,
                                          * some of these nodes may not be allowed for allocation,
                                          * see hwloc_topology_get_allowed_nodeset().
                                          *
                                          * If there are no NUMA nodes in the machine, all the memory is close to this
                                          * object, so only the first bit may be set in \p nodeset.
                                          *
					  * \note All objects have non-NULL CPU and node sets except Misc and I/O objects.
					  *
                                          * \note Its value must not be changed, hwloc_bitmap_dup() must be used instead.
                                          */
  hwloc_nodeset_t complete_nodeset;     /**< \brief The complete NUMA node set of this object,
                                          *
                                          * This may include not only the same as the nodeset field, but also some NUMA
                                          * nodes for which topology information is unknown or incomplete, some offlines
                                          * nodes, and the nodes that are ignored when the ::HWLOC_TOPOLOGY_FLAG_INCLUDE_DISALLOWED
                                          * flag is not set.
                                          * Thus no corresponding NUMA node object may be found in the topology, because the
                                          * precise position is undefined. It is however known that it would be
                                          * somewhere under this object.
                                          *
                                          * If there are no NUMA nodes in the machine, all the memory is close to this
                                          * object, so only the first bit is set in \p complete_nodeset.
                                          *
                                          * \note Its value must not be changed, hwloc_bitmap_dup() must be used instead.
                                          */

  struct hwloc_info_s *infos;		/**< \brief Array of stringified info type=name. */
  unsigned infos_count;			/**< \brief Size of infos array. */

  /* misc */
  void *userdata;			/**< \brief Application-given private data pointer,
					 * initialized to \c NULL, use it as you wish.
					 * See hwloc_topology_set_userdata_export_callback() in hwloc/export.h
					 * if you wish to export this field to XML. */

  hwloc_uint64_t gp_index;			/**< \brief Global persistent index.
					 * Generated by hwloc, unique across the topology (contrary to os_index)
					 * and persistent across topology changes (contrary to logical_index).
					 * Mostly used internally, but could also be used by application to identify objects.
					 */
};
/**
 * \brief Convenience typedef; a pointer to a struct hwloc_obj.
 */
typedef struct hwloc_obj * hwloc_obj_t;

/** \brief Object type-specific Attributes */
union hwloc_obj_attr_u {
  /** \brief NUMA node-specific Object Attributes */
  struct hwloc_numanode_attr_s {
    hwloc_uint64_t local_memory; /**< \brief Local memory (in bytes) */
    unsigned page_types_len; /**< \brief Size of array \p page_types */
    /** \brief Array of local memory page types, \c NULL if no local memory and \p page_types is 0.
     *
     * The array is sorted by increasing \p size fields.
     * It contains \p page_types_len slots.
     */
    struct hwloc_memory_page_type_s {
      hwloc_uint64_t size;	/**< \brief Size of pages */
      hwloc_uint64_t count;	/**< \brief Number of pages of this size */
    } * page_types;
  } numanode;

  /** \brief Cache-specific Object Attributes */
  struct hwloc_cache_attr_s {
    hwloc_uint64_t size;		  /**< \brief Size of cache in bytes */
    unsigned depth;			  /**< \brief Depth of cache (e.g., L1, L2, ...etc.) */
    unsigned linesize;			  /**< \brief Cache-line size in bytes. 0 if unknown */
    int associativity;			  /**< \brief Ways of associativity,
    					    *  -1 if fully associative, 0 if unknown */
    hwloc_obj_cache_type_t type;          /**< \brief Cache type */
  } cache;
  /** \brief Group-specific Object Attributes */
  struct hwloc_group_attr_s {
    unsigned depth;			  /**< \brief Depth of group object.
					   *   It may change if intermediate Group objects are added. */
    unsigned kind;			  /**< \brief Internally-used kind of group. */
    unsigned subkind;			  /**< \brief Internally-used subkind to distinguish different levels of groups with same kind */
    unsigned char dont_merge;		  /**< \brief Flag preventing groups from being automatically merged with identical parent or children. */
  } group;
  /** \brief PCI Device specific Object Attributes */
  struct hwloc_pcidev_attr_s {
#ifndef HWLOC_HAVE_32BITS_PCI_DOMAIN
    unsigned short domain; /* Only 16bits PCI domains are supported by default */
#else
    unsigned int domain; /* 32bits PCI domain support break the library ABI, hence it's disabled by default */
#endif
    unsigned char bus, dev, func;
    unsigned short class_id;
    unsigned short vendor_id, device_id, subvendor_id, subdevice_id;
    unsigned char revision;
    float linkspeed; /* in GB/s */
  } pcidev;
  /** \brief Bridge specific Object Attribues */
  struct hwloc_bridge_attr_s {
    union {
      struct hwloc_pcidev_attr_s pci;
    } upstream;
    hwloc_obj_bridge_type_t upstream_type;
    union {
      struct {
#ifndef HWLOC_HAVE_32BITS_PCI_DOMAIN
	unsigned short domain; /* Only 16bits PCI domains are supported by default */
#else
	unsigned int domain; /* 32bits PCI domain support break the library ABI, hence it's disabled by default */
#endif
	unsigned char secondary_bus, subordinate_bus;
      } pci;
    } downstream;
    hwloc_obj_bridge_type_t downstream_type;
    unsigned depth;
  } bridge;
  /** \brief OS Device specific Object Attributes */
  struct hwloc_osdev_attr_s {
    hwloc_obj_osdev_type_t type;
  } osdev;
};

/** \brief Object info
 *
 * \sa hwlocality_info_attr
 */
struct hwloc_info_s {
  char *name;	/**< \brief Info name */
  char *value;	/**< \brief Info value */
};

/** @} */



/** \defgroup hwlocality_creation Topology Creation and Destruction
 * @{
 */

struct hwloc_topology;
/** \brief Topology context
 *
 * To be initialized with hwloc_topology_init() and built with hwloc_topology_load().
 */
typedef struct hwloc_topology * hwloc_topology_t;

/** \brief Allocate a topology context.
 *
 * \param[out] topologyp is assigned a pointer to the new allocated context.
 *
 * \return 0 on success, -1 on error.
 */
HWLOC_DECLSPEC int hwloc_topology_init (hwloc_topology_t *topologyp);

/** \brief Build the actual topology
 *
 * Build the actual topology once initialized with hwloc_topology_init() and
 * tuned with \ref hwlocality_configuration and \ref hwlocality_setsource routines.
 * No other routine may be called earlier using this topology context.
 *
 * \param topology is the topology to be loaded with objects.
 *
 * \return 0 on success, -1 on error.
 *
 * \note On failure, the topology is reinitialized. It should be either
 * destroyed with hwloc_topology_destroy() or configured and loaded again.
 *
 * \note This function may be called only once per topology.
 *
 * \note The binding of the current thread or process may temporarily change
 * during this call but it will be restored before it returns.
 *
 * \sa hwlocality_configuration and hwlocality_setsource
 */
HWLOC_DECLSPEC int hwloc_topology_load(hwloc_topology_t topology);

/** \brief Terminate and free a topology context
 *
 * \param topology is the topology to be freed
 */
HWLOC_DECLSPEC void hwloc_topology_destroy (hwloc_topology_t topology);

/** \brief Duplicate a topology.
 *
 * The entire topology structure as well as its objects
 * are duplicated into a new one.
 *
 * This is useful for keeping a backup while modifying a topology.
 *
 * \note Object userdata is not duplicated since hwloc does not know what it point to.
 * The objects of both old and new topologies will point to the same userdata.
 */
HWLOC_DECLSPEC int hwloc_topology_dup(hwloc_topology_t *newtopology, hwloc_topology_t oldtopology);

/** \brief Verify that the topology is compatible with the current hwloc library.
 *
 * This is useful when using the same topology structure (in memory)
 * in different libraries that may use different hwloc installations
 * (for instance if one library embeds a specific version of hwloc,
 * while another library uses a default system-wide hwloc installation).
 *
 * If all libraries/programs use the same hwloc installation, this function
 * always returns success.
 *
 * \return \c 0 on success.
 *
 * \return \c -1 with \p errno set to \c EINVAL if incompatible.
 *
 * \note If sharing between processes with hwloc_shmem_topology_write(),
 * the relevant check is already performed inside hwloc_shmem_topology_adopt().
 */
HWLOC_DECLSPEC int hwloc_topology_abi_check(hwloc_topology_t topology);

/** \brief Run internal checks on a topology structure
 *
 * The program aborts if an inconsistency is detected in the given topology.
 *
 * \param topology is the topology to be checked
 *
 * \note This routine is only useful to developers.
 *
 * \note The input topology should have been previously loaded with
 * hwloc_topology_load().
 */
HWLOC_DECLSPEC void hwloc_topology_check(hwloc_topology_t topology);

/** @} */



/** \defgroup hwlocality_levels Object levels, depths and types
 * @{
 *
 * Be sure to see the figure in \ref termsanddefs that shows a
 * complete topology tree, including depths, child/sibling/cousin
 * relationships, and an example of an asymmetric topology where one
 * package has fewer caches than its peers.
 */

/** \brief Get the depth of the hierarchical tree of objects.
 *
 * This is the depth of ::HWLOC_OBJ_PU objects plus one.
 *
 * \note NUMA nodes, I/O and Misc objects are ignored when computing
 * the depth of the tree (they are placed on special levels).
 */
HWLOC_DECLSPEC int hwloc_topology_get_depth(hwloc_topology_t __hwloc_restrict topology) __hwloc_attribute_pure;

/** \brief Returns the depth of objects of type \p type.
 *
 * If no object of this type is present on the underlying architecture, or if
 * the OS doesn't provide this kind of information, the function returns
 * ::HWLOC_TYPE_DEPTH_UNKNOWN.
 *
 * If type is absent but a similar type is acceptable, see also
 * hwloc_get_type_or_below_depth() and hwloc_get_type_or_above_depth().
 *
 * If ::HWLOC_OBJ_GROUP is given, the function may return ::HWLOC_TYPE_DEPTH_MULTIPLE
 * if multiple levels of Groups exist.
 *
 * If a NUMA node, I/O or Misc object type is given, the function returns a virtual
 * value because these objects are stored in special levels that are not CPU-related.
 * This virtual depth may be passed to other hwloc functions such as
 * hwloc_get_obj_by_depth() but it should not be considered as an actual
 * depth by the application. In particular, it should not be compared with
 * any other object depth or with the entire topology depth.
 * \sa hwloc_get_memory_parents_depth().
 *
 * \sa hwloc_type_sscanf_as_depth() for returning the depth of objects
 * whose type is given as a string.
 */
HWLOC_DECLSPEC int hwloc_get_type_depth (hwloc_topology_t topology, hwloc_obj_type_t type);

enum hwloc_get_type_depth_e {
    HWLOC_TYPE_DEPTH_UNKNOWN = -1,    /**< \brief No object of given type exists in the topology. \hideinitializer */
    HWLOC_TYPE_DEPTH_MULTIPLE = -2,   /**< \brief Objects of given type exist at different depth in the topology (only for Groups). \hideinitializer */
    HWLOC_TYPE_DEPTH_NUMANODE = -3,   /**< \brief Virtual depth for NUMA nodes. \hideinitializer */
    HWLOC_TYPE_DEPTH_BRIDGE = -4,     /**< \brief Virtual depth for bridge object level. \hideinitializer */
    HWLOC_TYPE_DEPTH_PCI_DEVICE = -5, /**< \brief Virtual depth for PCI device object level. \hideinitializer */
    HWLOC_TYPE_DEPTH_OS_DEVICE = -6,  /**< \brief Virtual depth for software device object level. \hideinitializer */
    HWLOC_TYPE_DEPTH_MISC = -7,       /**< \brief Virtual depth for Misc object. \hideinitializer */
    HWLOC_TYPE_DEPTH_MEMCACHE = -8    /**< \brief Virtual depth for MemCache object. \hideinitializer */
};

/** \brief Return the depth of parents where memory objects are attached.
 *
 * Memory objects have virtual negative depths because they are not part of
 * the main CPU-side hierarchy of objects. This depth should not be compared
 * with other level depths.
 *
 * If all Memory objects are attached to Normal parents at the same depth,
 * this parent depth may be compared to other as usual, for instance
 * for knowing whether NUMA nodes is attached above or below Packages.
 *
 * \return The depth of Normal parents of all memory children
 * if all these parents have the same depth. For instance the depth of
 * the Package level if all NUMA nodes are attached to Package objects.
 *
 * \return ::HWLOC_TYPE_DEPTH_MULTIPLE if Normal parents of all
 * memory children do not have the same depth. For instance if some
 * NUMA nodes are attached to Packages while others are attached to
 * Groups.
 */
HWLOC_DECLSPEC int hwloc_get_memory_parents_depth (hwloc_topology_t topology);

/** \brief Returns the depth of objects of type \p type or below
 *
 * If no object of this type is present on the underlying architecture, the
 * function returns the depth of the first "present" object typically found
 * inside \p type.
 *
 * This function is only meaningful for normal object types.
 * If a memory, I/O or Misc object type is given, the corresponding virtual
 * depth is always returned (see hwloc_get_type_depth()).
 *
 * May return ::HWLOC_TYPE_DEPTH_MULTIPLE for ::HWLOC_OBJ_GROUP just like
 * hwloc_get_type_depth().
 */
static __hwloc_inline int
hwloc_get_type_or_below_depth (hwloc_topology_t topology, hwloc_obj_type_t type) __hwloc_attribute_pure;

/** \brief Returns the depth of objects of type \p type or above
 *
 * If no object of this type is present on the underlying architecture, the
 * function returns the depth of the first "present" object typically
 * containing \p type.
 *
 * This function is only meaningful for normal object types.
 * If a memory, I/O or Misc object type is given, the corresponding virtual
 * depth is always returned (see hwloc_get_type_depth()).
 *
 * May return ::HWLOC_TYPE_DEPTH_MULTIPLE for ::HWLOC_OBJ_GROUP just like
 * hwloc_get_type_depth().
 */
static __hwloc_inline int
hwloc_get_type_or_above_depth (hwloc_topology_t topology, hwloc_obj_type_t type) __hwloc_attribute_pure;

/** \brief Returns the type of objects at depth \p depth.
 *
 * \p depth should between 0 and hwloc_topology_get_depth()-1,
 * or a virtual depth such as ::HWLOC_TYPE_DEPTH_NUMANODE.
 *
 * \return (hwloc_obj_type_t)-1 if depth \p depth does not exist.
 */
HWLOC_DECLSPEC hwloc_obj_type_t hwloc_get_depth_type (hwloc_topology_t topology, int depth) __hwloc_attribute_pure;

/** \brief Returns the width of level at depth \p depth.
 */
HWLOC_DECLSPEC unsigned hwloc_get_nbobjs_by_depth (hwloc_topology_t topology, int depth) __hwloc_attribute_pure;

/** \brief Returns the width of level type \p type
 *
 * If no object for that type exists, 0 is returned.
 * If there are several levels with objects of that type, -1 is returned.
 */
static __hwloc_inline int
hwloc_get_nbobjs_by_type (hwloc_topology_t topology, hwloc_obj_type_t type) __hwloc_attribute_pure;

/** \brief Returns the top-object of the topology-tree.
 *
 * Its type is ::HWLOC_OBJ_MACHINE.
 */
static __hwloc_inline hwloc_obj_t
hwloc_get_root_obj (hwloc_topology_t topology) __hwloc_attribute_pure;

/** \brief Returns the topology object at logical index \p idx from depth \p depth */
HWLOC_DECLSPEC hwloc_obj_t hwloc_get_obj_by_depth (hwloc_topology_t topology, int depth, unsigned idx) __hwloc_attribute_pure;

/** \brief Returns the topology object at logical index \p idx with type \p type
 *
 * If no object for that type exists, \c NULL is returned.
 * If there are several levels with objects of that type (::HWLOC_OBJ_GROUP),
 * \c NULL is returned and the caller may fallback to hwloc_get_obj_by_depth().
 */
static __hwloc_inline hwloc_obj_t
hwloc_get_obj_by_type (hwloc_topology_t topology, hwloc_obj_type_t type, unsigned idx) __hwloc_attribute_pure;

/** \brief Returns the next object at depth \p depth.
 *
 * If \p prev is \c NULL, return the first object at depth \p depth.
 */
static __hwloc_inline hwloc_obj_t
hwloc_get_next_obj_by_depth (hwloc_topology_t topology, int depth, hwloc_obj_t prev);

/** \brief Returns the next object of type \p type.
 *
 * If \p prev is \c NULL, return the first object at type \p type.  If
 * there are multiple or no depth for given type, return \c NULL and
 * let the caller fallback to hwloc_get_next_obj_by_depth().
 */
static __hwloc_inline hwloc_obj_t
hwloc_get_next_obj_by_type (hwloc_topology_t topology, hwloc_obj_type_t type,
			    hwloc_obj_t prev);

/** @} */



/** \defgroup hwlocality_object_strings Converting between Object Types and Attributes, and Strings
 * @{
 */

/** \brief Return a constant stringified object type.
 *
 * This function is the basic way to convert a generic type into a string.
 * The output string may be parsed back by hwloc_type_sscanf().
 *
 * hwloc_obj_type_snprintf() may return a more precise output for a specific
 * object, but it requires the caller to provide the output buffer.
 */
HWLOC_DECLSPEC const char * hwloc_obj_type_string (hwloc_obj_type_t type) __hwloc_attribute_const;

/** \brief Stringify the type of a given topology object into a human-readable form.
 *
 * Contrary to hwloc_obj_type_string(), this function includes object-specific
 * attributes (such as the Group depth, the Bridge type, or OS device type)
 * in the output, and it requires the caller to provide the output buffer.
 *
 * The output is guaranteed to be the same for all objects of a same topology level.
 *
 * If \p verbose is 1, longer type names are used, e.g. L1Cache instead of L1.
 *
 * The output string may be parsed back by hwloc_type_sscanf().
 *
 * If \p size is 0, \p string may safely be \c NULL.
 *
 * \return the number of character that were actually written if not truncating,
 * or that would have been written (not including the ending \\0).
 */
HWLOC_DECLSPEC int hwloc_obj_type_snprintf(char * __hwloc_restrict string, size_t size,
					   hwloc_obj_t obj,
					   int verbose);

/** \brief Stringify the attributes of a given topology object into a human-readable form.
 *
 * Attribute values are separated by \p separator.
 *
 * Only the major attributes are printed in non-verbose mode.
 *
 * If \p size is 0, \p string may safely be \c NULL.
 *
 * \return the number of character that were actually written if not truncating,
 * or that would have been written (not including the ending \\0).
 */
HWLOC_DECLSPEC int hwloc_obj_attr_snprintf(char * __hwloc_restrict string, size_t size,
					   hwloc_obj_t obj, const char * __hwloc_restrict separator,
					   int verbose);

/** \brief Return an object type and attributes from a type string.
 *
 * Convert strings such as "Package" or "L1iCache" into the corresponding types.
 * Matching is case-insensitive, and only the first letters are actually
 * required to match.
 *
 * The matched object type is set in \p typep (which cannot be \c NULL).
 *
 * Type-specific attributes, for instance Cache type, Cache depth, Group depth,
 * Bridge type or OS Device type may be returned in \p attrp.
 * Attributes that are not specified in the string (for instance "Group"
 * without a depth, or "L2Cache" without a cache type) are set to -1.
 *
 * \p attrp is only filled if not \c NULL and if its size specified in \p attrsize
 * is large enough. It should be at least as large as union hwloc_obj_attr_u.
 *
 * \return 0 if a type was correctly identified, otherwise -1.
 *
 * \note This function is guaranteed to match any string returned by
 * hwloc_obj_type_string() or hwloc_obj_type_snprintf().
 *
 * \note This is an extended version of the now deprecated hwloc_obj_type_sscanf().
 */
HWLOC_DECLSPEC int hwloc_type_sscanf(const char *string,
				     hwloc_obj_type_t *typep,
				     union hwloc_obj_attr_u *attrp, size_t attrsize);

/** \brief Return an object type and its level depth from a type string.
 *
 * Convert strings such as "Package" or "L1iCache" into the corresponding types
 * and return in \p depthp the depth of the corresponding level in the
 * topology \p topology.
 *
 * If no object of this type is present on the underlying architecture,
 * ::HWLOC_TYPE_DEPTH_UNKNOWN is returned.
 *
 * If multiple such levels exist (for instance if giving Group without any depth),
 * the function may return ::HWLOC_TYPE_DEPTH_MULTIPLE instead.
 *
 * The matched object type is set in \p typep if \p typep is non \c NULL.
 *
 * \note This function is similar to hwloc_type_sscanf() followed
 * by hwloc_get_type_depth() but it also automatically disambiguates
 * multiple group levels etc.
 *
 * \note This function is guaranteed to match any string returned by
 * hwloc_obj_type_string() or hwloc_obj_type_snprintf().
 */
HWLOC_DECLSPEC int hwloc_type_sscanf_as_depth(const char *string,
					      hwloc_obj_type_t *typep,
					      hwloc_topology_t topology, int *depthp);

/** @} */



/** \defgroup hwlocality_info_attr Consulting and Adding Key-Value Info Attributes
 *
 * @{
 */

/** \brief Search the given key name in object infos and return the corresponding value.
 *
 * If multiple keys match the given name, only the first one is returned.
 *
 * \return \c NULL if no such key exists.
 */
static __hwloc_inline const char *
hwloc_obj_get_info_by_name(hwloc_obj_t obj, const char *name) __hwloc_attribute_pure;

/** \brief Add the given info name and value pair to the given object.
 *
 * The info is appended to the existing info array even if another key
 * with the same name already exists.
 *
 * The input strings are copied before being added in the object infos.
 *
 * \return \c 0 on success, \c -1 on error.
 *
 * \note This function may be used to enforce object colors in the lstopo
 * graphical output by using "lstopoStyle" as a name and "Background=#rrggbb"
 * as a value. See CUSTOM COLORS in the lstopo(1) manpage for details.
 *
 * \note If \p value contains some non-printable characters, they will
 * be dropped when exporting to XML, see hwloc_topology_export_xml() in hwloc/export.h.
 */
HWLOC_DECLSPEC int hwloc_obj_add_info(hwloc_obj_t obj, const char *name, const char *value);

/** @} */



/** \defgroup hwlocality_cpubinding CPU binding
 *
 * Some operating systems only support binding threads or processes to a single PU.
 * Others allow binding to larger sets such as entire Cores or Packages or
 * even random sets of invididual PUs. In such operating system, the scheduler
 * is free to run the task on one of these PU, then migrate it to another PU, etc.
 * It is often useful to call hwloc_bitmap_singlify() on the target CPU set before
 * passing it to the binding function to avoid these expensive migrations.
 * See the documentation of hwloc_bitmap_singlify() for details.
 *
 * Some operating systems do not provide all hwloc-supported
 * mechanisms to bind processes, threads, etc.
 * hwloc_topology_get_support() may be used to query about the actual CPU
 * binding support in the currently used operating system.
 *
 * When the requested binding operation is not available and the
 * ::HWLOC_CPUBIND_STRICT flag was passed, the function returns -1.
 * \p errno is set to \c ENOSYS when it is not possible to bind the requested kind of object
 * processes/threads. errno is set to \c EXDEV when the requested cpuset
 * can not be enforced (e.g. some systems only allow one CPU, and some
 * other systems only allow one NUMA node).
 *
 * If ::HWLOC_CPUBIND_STRICT was not passed, the function may fail as well,
 * or the operating system may use a slightly different operation
 * (with side-effects, smaller binding set, etc.)
 * when the requested operation is not exactly supported.
 *
 * The most portable version that should be preferred over the others,
 * whenever possible, is the following one which just binds the current program,
 * assuming it is single-threaded:
 *
 * \code
 * hwloc_set_cpubind(topology, set, 0),
 * \endcode
 *
 * If the program may be multithreaded, the following one should be preferred
 * to only bind the current thread:
 *
 * \code
 * hwloc_set_cpubind(topology, set, HWLOC_CPUBIND_THREAD),
 * \endcode
 *
 * \sa Some example codes are available under doc/examples/ in the source tree.
 *
 * \note To unbind, just call the binding function with either a full cpuset or
 * a cpuset equal to the system cpuset.
 *
 * \note On some operating systems, CPU binding may have effects on memory binding, see
 * ::HWLOC_CPUBIND_NOMEMBIND
 *
 * \note Running lstopo \--top or hwloc-ps can be a very convenient tool to check
 * how binding actually happened.
 * @{
 */

/** \brief Process/Thread binding flags.
 *
 * These bit flags can be used to refine the binding policy.
 *
 * The default (0) is to bind the current process, assumed to be
 * single-threaded, in a non-strict way.  This is the most portable
 * way to bind as all operating systems usually provide it.
 *
 * \note Not all systems support all kinds of binding.  See the
 * "Detailed Description" section of \ref hwlocality_cpubinding for a
 * description of errors that can occur.
 */
typedef enum {
  /** \brief Bind all threads of the current (possibly) multithreaded process.
   * \hideinitializer */
  HWLOC_CPUBIND_PROCESS = (1<<0),

  /** \brief Bind current thread of current process.
   * \hideinitializer */
  HWLOC_CPUBIND_THREAD = (1<<1),

  /** \brief Request for strict binding from the OS.
   *
   * By default, when the designated CPUs are all busy while other
   * CPUs are idle, operating systems may execute the thread/process
   * on those other CPUs instead of the designated CPUs, to let them
   * progress anyway.  Strict binding means that the thread/process
   * will _never_ execute on other cpus than the designated CPUs, even
   * when those are busy with other tasks and other CPUs are idle.
   *
   * \note Depending on the operating system, strict binding may not
   * be possible (e.g., the OS does not implement it) or not allowed
   * (e.g., for an administrative reasons), and the function will fail
   * in that case.
   *
   * When retrieving the binding of a process, this flag checks
   * whether all its threads  actually have the same binding. If the
   * flag is not given, the binding of each thread will be
   * accumulated.
   *
   * \note This flag is meaningless when retrieving the binding of a
   * thread.
   * \hideinitializer
   */
  HWLOC_CPUBIND_STRICT = (1<<2),

  /** \brief Avoid any effect on memory binding
   *
   * On some operating systems, some CPU binding function would also
   * bind the memory on the corresponding NUMA node.  It is often not
   * a problem for the application, but if it is, setting this flag
   * will make hwloc avoid using OS functions that would also bind
   * memory.  This will however reduce the support of CPU bindings,
   * i.e. potentially return -1 with errno set to ENOSYS in some
   * cases.
   *
   * This flag is only meaningful when used with functions that set
   * the CPU binding.  It is ignored when used with functions that get
   * CPU binding information.
   * \hideinitializer
   */
  HWLOC_CPUBIND_NOMEMBIND = (1<<3)
} hwloc_cpubind_flags_t;

/** \brief Bind current process or thread on cpus given in physical bitmap \p set.
 *
 * \return -1 with errno set to ENOSYS if the action is not supported
 * \return -1 with errno set to EXDEV if the binding cannot be enforced
 */
HWLOC_DECLSPEC int hwloc_set_cpubind(hwloc_topology_t topology, hwloc_const_cpuset_t set, int flags);

/** \brief Get current process or thread binding.
 *
 * Writes into \p set the physical cpuset which the process or thread (according to \e
 * flags) was last bound to.
 */
HWLOC_DECLSPEC int hwloc_get_cpubind(hwloc_topology_t topology, hwloc_cpuset_t set, int flags);

/** \brief Bind a process \p pid on cpus given in physical bitmap \p set.
 *
 * \note \p hwloc_pid_t is \p pid_t on Unix platforms,
 * and \p HANDLE on native Windows platforms.
 *
 * \note As a special case on Linux, if a tid (thread ID) is supplied
 * instead of a pid (process ID) and ::HWLOC_CPUBIND_THREAD is passed in flags,
 * the binding is applied to that specific thread.
 *
 * \note On non-Linux systems, ::HWLOC_CPUBIND_THREAD can not be used in \p flags.
 */
HWLOC_DECLSPEC int hwloc_set_proc_cpubind(hwloc_topology_t topology, hwloc_pid_t pid, hwloc_const_cpuset_t set, int flags);

/** \brief Get the current physical binding of process \p pid.
 *
 * \note \p hwloc_pid_t is \p pid_t on Unix platforms,
 * and \p HANDLE on native Windows platforms.
 *
 * \note As a special case on Linux, if a tid (thread ID) is supplied
 * instead of a pid (process ID) and HWLOC_CPUBIND_THREAD is passed in flags,
 * the binding for that specific thread is returned.
 *
 * \note On non-Linux systems, HWLOC_CPUBIND_THREAD can not be used in \p flags.
 */
HWLOC_DECLSPEC int hwloc_get_proc_cpubind(hwloc_topology_t topology, hwloc_pid_t pid, hwloc_cpuset_t set, int flags);

#ifdef hwloc_thread_t
/** \brief Bind a thread \p thread on cpus given in physical bitmap \p set.
 *
 * \note \p hwloc_thread_t is \p pthread_t on Unix platforms,
 * and \p HANDLE on native Windows platforms.
 *
 * \note ::HWLOC_CPUBIND_PROCESS can not be used in \p flags.
 */
HWLOC_DECLSPEC int hwloc_set_thread_cpubind(hwloc_topology_t topology, hwloc_thread_t thread, hwloc_const_cpuset_t set, int flags);
#endif

#ifdef hwloc_thread_t
/** \brief Get the current physical binding of thread \p tid.
 *
 * \note \p hwloc_thread_t is \p pthread_t on Unix platforms,
 * and \p HANDLE on native Windows platforms.
 *
 * \note ::HWLOC_CPUBIND_PROCESS can not be used in \p flags.
 */
HWLOC_DECLSPEC int hwloc_get_thread_cpubind(hwloc_topology_t topology, hwloc_thread_t thread, hwloc_cpuset_t set, int flags);
#endif

/** \brief Get the last physical CPU where the current process or thread ran.
 *
 * The operating system may move some tasks from one processor
 * to another at any time according to their binding,
 * so this function may return something that is already
 * outdated.
 *
 * \p flags can include either ::HWLOC_CPUBIND_PROCESS or ::HWLOC_CPUBIND_THREAD to
 * specify whether the query should be for the whole process (union of all CPUs
 * on which all threads are running), or only the current thread. If the
 * process is single-threaded, flags can be set to zero to let hwloc use
 * whichever method is available on the underlying OS.
 */
HWLOC_DECLSPEC int hwloc_get_last_cpu_location(hwloc_topology_t topology, hwloc_cpuset_t set, int flags);

/** \brief Get the last physical CPU where a process ran.
 *
 * The operating system may move some tasks from one processor
 * to another at any time according to their binding,
 * so this function may return something that is already
 * outdated.
 *
 * \note \p hwloc_pid_t is \p pid_t on Unix platforms,
 * and \p HANDLE on native Windows platforms.
 *
 * \note As a special case on Linux, if a tid (thread ID) is supplied
 * instead of a pid (process ID) and ::HWLOC_CPUBIND_THREAD is passed in flags,
 * the last CPU location of that specific thread is returned.
 *
 * \note On non-Linux systems, ::HWLOC_CPUBIND_THREAD can not be used in \p flags.
 */
HWLOC_DECLSPEC int hwloc_get_proc_last_cpu_location(hwloc_topology_t topology, hwloc_pid_t pid, hwloc_cpuset_t set, int flags);

/** @} */



/** \defgroup hwlocality_membinding Memory binding
 *
 * Memory binding can be done three ways:
 *
 * - explicit memory allocation thanks to hwloc_alloc_membind() and friends:
 *   the binding will have effect on the memory allocated by these functions.
 * - implicit memory binding through binding policy: hwloc_set_membind() and
 *   friends only define the current policy of the process, which will be
 *   applied to the subsequent calls to malloc() and friends.
 * - migration of existing memory ranges, thanks to hwloc_set_area_membind()
 *   and friends, which move already-allocated data.
 *
 * Not all operating systems support all three ways.
 * hwloc_topology_get_support() may be used to query about the actual memory
 * binding support in the currently used operating system.
 *
 * When the requested binding operation is not available and the
 * ::HWLOC_MEMBIND_STRICT flag was passed, the function returns -1.
 * \p errno will be set to \c ENOSYS when the system does support
 * the specified action or policy
 * (e.g., some systems only allow binding memory on a per-thread
 * basis, whereas other systems only allow binding memory for all
 * threads in a process).
 * \p errno will be set to EXDEV when the requested set can not be enforced
 * (e.g., some systems only allow binding memory to a single NUMA node).
 *
 * If ::HWLOC_MEMBIND_STRICT was not passed, the function may fail as well,
 * or the operating system may use a slightly different operation
 * (with side-effects, smaller binding set, etc.)
 * when the requested operation is not exactly supported.
 *
 * The most portable form that should be preferred over the others
 * whenever possible is as follows.
 * It allocates some memory hopefully bound to the specified set.
 * To do so, hwloc will possibly have to change the current memory
 * binding policy in order to actually get the memory bound, if the OS
 * does not provide any other way to simply allocate bound memory
 * without changing the policy for all allocations. That is the
 * difference with hwloc_alloc_membind(), which will never change the
 * current memory binding policy.
 *
 * \code
 * hwloc_alloc_membind_policy(topology, size, set,
 *                            HWLOC_MEMBIND_BIND, 0);
 * \endcode
 *
 * Each hwloc memory binding function takes a bitmap argument that
 * is a CPU set by default, or a NUMA memory node set if the flag
 * ::HWLOC_MEMBIND_BYNODESET is specified.
 * See \ref hwlocality_object_sets and \ref hwlocality_bitmap for a
 * discussion of CPU sets and NUMA memory node sets.
 * It is also possible to convert between CPU set and node set using
 * hwloc_cpuset_to_nodeset() or hwloc_cpuset_from_nodeset().
 *
 * Memory binding by CPU set cannot work for CPU-less NUMA memory nodes.
 * Binding by nodeset should therefore be preferred whenever possible.
 *
 * \sa Some example codes are available under doc/examples/ in the source tree.
 *
 * \note On some operating systems, memory binding affects the CPU
 * binding; see ::HWLOC_MEMBIND_NOCPUBIND
 * @{
 */

/** \brief Memory binding policy.
 *
 * These constants can be used to choose the binding policy.  Only one policy can
 * be used at a time (i.e., the values cannot be OR'ed together).
 *
 * Not all systems support all kinds of binding.
 * hwloc_topology_get_support() may be used to query about the actual memory
 * binding policy support in the currently used operating system.
 * See the "Detailed Description" section of \ref hwlocality_membinding
 * for a description of errors that can occur.
 */
typedef enum {
  /** \brief Reset the memory allocation policy to the system default.
   * Depending on the operating system, this may correspond to
   * ::HWLOC_MEMBIND_FIRSTTOUCH (Linux, FreeBSD),
   * or ::HWLOC_MEMBIND_BIND (AIX, HP-UX, Solaris, Windows).
   * This policy is never returned by get membind functions.
   * The nodeset argument is ignored.
   * \hideinitializer */
  HWLOC_MEMBIND_DEFAULT =	0,

  /** \brief Allocate each memory page individually on the local NUMA
   * node of the thread that touches it.
   *
   * The given nodeset should usually be hwloc_topology_get_topology_nodeset()
   * so that the touching thread may run and allocate on any node in the system.
   *
   * On AIX, if the nodeset is smaller, pages are allocated locally (if the local
   * node is in the nodeset) or from a random non-local node (otherwise).
   * \hideinitializer */
  HWLOC_MEMBIND_FIRSTTOUCH =	1,

  /** \brief Allocate memory on the specified nodes.
   * \hideinitializer */
  HWLOC_MEMBIND_BIND =		2,

  /** \brief Allocate memory on the given nodes in an interleaved
   * / round-robin manner.  The precise layout of the memory across
   * multiple NUMA nodes is OS/system specific. Interleaving can be
   * useful when threads distributed across the specified NUMA nodes
   * will all be accessing the whole memory range concurrently, since
   * the interleave will then balance the memory references.
   * \hideinitializer */
  HWLOC_MEMBIND_INTERLEAVE =	3,

  /** \brief For each page bound with this policy, by next time
   * it is touched (and next time only), it is moved from its current
   * location to the local NUMA node of the thread where the memory
   * reference occurred (if it needs to be moved at all).
   * \hideinitializer */
  HWLOC_MEMBIND_NEXTTOUCH =	4,

  /** \brief Returned by get_membind() functions when multiple
   * threads or parts of a memory area have differing memory binding
   * policies.
   * Also returned when binding is unknown because binding hooks are empty
   * when the topology is loaded from XML without HWLOC_THISSYSTEM=1, etc.
   * \hideinitializer */
  HWLOC_MEMBIND_MIXED = -1
} hwloc_membind_policy_t;

/** \brief Memory binding flags.
 *
 * These flags can be used to refine the binding policy.
 * All flags can be logically OR'ed together with the exception of
 * ::HWLOC_MEMBIND_PROCESS and ::HWLOC_MEMBIND_THREAD;
 * these two flags are mutually exclusive.
 *
 * Not all systems support all kinds of binding.
 * hwloc_topology_get_support() may be used to query about the actual memory
 * binding support in the currently used operating system.
 * See the "Detailed Description" section of \ref hwlocality_membinding
 * for a description of errors that can occur.
 */
typedef enum {
  /** \brief Set policy for all threads of the specified (possibly
   * multithreaded) process.  This flag is mutually exclusive with
   * ::HWLOC_MEMBIND_THREAD.
   * \hideinitializer */
  HWLOC_MEMBIND_PROCESS =       (1<<0),

 /** \brief Set policy for a specific thread of the current process.
  * This flag is mutually exclusive with ::HWLOC_MEMBIND_PROCESS.
  * \hideinitializer */
  HWLOC_MEMBIND_THREAD =        (1<<1),

 /** Request strict binding from the OS.  The function will fail if
  * the binding can not be guaranteed / completely enforced.
  *
  * This flag has slightly different meanings depending on which
  * function it is used with.
  * \hideinitializer  */
  HWLOC_MEMBIND_STRICT =        (1<<2),

 /** \brief Migrate existing allocated memory.  If the memory cannot
  * be migrated and the ::HWLOC_MEMBIND_STRICT flag is passed, an error
  * will be returned.
  * \hideinitializer  */
  HWLOC_MEMBIND_MIGRATE =       (1<<3),

  /** \brief Avoid any effect on CPU binding.
   *
   * On some operating systems, some underlying memory binding
   * functions also bind the application to the corresponding CPU(s).
   * Using this flag will cause hwloc to avoid using OS functions that
   * could potentially affect CPU bindings.  Note, however, that using
   * NOCPUBIND may reduce hwloc's overall memory binding
   * support. Specifically: some of hwloc's memory binding functions
   * may fail with errno set to ENOSYS when used with NOCPUBIND.
   * \hideinitializer
   */
  HWLOC_MEMBIND_NOCPUBIND =     (1<<4),

  /** \brief Consider the bitmap argument as a nodeset.
   *
   * The bitmap argument is considered a nodeset if this flag is given,
   * or a cpuset otherwise by default.
   *
   * Memory binding by CPU set cannot work for CPU-less NUMA memory nodes.
   * Binding by nodeset should therefore be preferred whenever possible.
   * \hideinitializer
   */
  HWLOC_MEMBIND_BYNODESET =     (1<<5)
} hwloc_membind_flags_t;

/** \brief Set the default memory binding policy of the current
 * process or thread to prefer the NUMA node(s) specified by \p set
 *
 * If neither ::HWLOC_MEMBIND_PROCESS nor ::HWLOC_MEMBIND_THREAD is
 * specified, the current process is assumed to be single-threaded.
 * This is the most portable form as it permits hwloc to use either
 * process-based OS functions or thread-based OS functions, depending
 * on which are available.
 *
 * If ::HWLOC_MEMBIND_BYNODESET is specified, set is considered a nodeset.
 * Otherwise it's a cpuset.
 *
 * \return -1 with errno set to ENOSYS if the action is not supported
 * \return -1 with errno set to EXDEV if the binding cannot be enforced
 */
HWLOC_DECLSPEC int hwloc_set_membind(hwloc_topology_t topology, hwloc_const_bitmap_t set, hwloc_membind_policy_t policy, int flags);

/** \brief Query the default memory binding policy and physical locality of the
 * current process or thread.
 *
 * This function has two output parameters: \p set and \p policy.
 * The values returned in these parameters depend on both the \p flags
 * passed in and the current memory binding policies and nodesets in
 * the queried target.
 *
 * Passing the ::HWLOC_MEMBIND_PROCESS flag specifies that the query
 * target is the current policies and nodesets for all the threads in
 * the current process.  Passing ::HWLOC_MEMBIND_THREAD specifies that
 * the query target is the current policy and nodeset for only the
 * thread invoking this function.
 *
 * If neither of these flags are passed (which is the most portable
 * method), the process is assumed to be single threaded.  This allows
 * hwloc to use either process-based OS functions or thread-based OS
 * functions, depending on which are available.
 *
 * ::HWLOC_MEMBIND_STRICT is only meaningful when ::HWLOC_MEMBIND_PROCESS
 * is also specified.  In this case, hwloc will check the default
 * memory policies and nodesets for all threads in the process.  If
 * they are not identical, -1 is returned and errno is set to EXDEV.
 * If they are identical, the values are returned in \p set and \p
 * policy.
 *
 * Otherwise, if ::HWLOC_MEMBIND_PROCESS is specified (and
 * ::HWLOC_MEMBIND_STRICT is \em not specified), the default set
 * from each thread is logically OR'ed together.
 * If all threads' default policies are the same, \p policy is set to
 * that policy.  If they are different, \p policy is set to
 * ::HWLOC_MEMBIND_MIXED.
 *
 * In the ::HWLOC_MEMBIND_THREAD case (or when neither
 * ::HWLOC_MEMBIND_PROCESS or ::HWLOC_MEMBIND_THREAD is specified), there
 * is only one set and policy; they are returned in \p set and
 * \p policy, respectively.
 *
 * If ::HWLOC_MEMBIND_BYNODESET is specified, set is considered a nodeset.
 * Otherwise it's a cpuset.
 *
 * If any other flags are specified, -1 is returned and errno is set
 * to EINVAL.
 */
HWLOC_DECLSPEC int hwloc_get_membind(hwloc_topology_t topology, hwloc_bitmap_t set, hwloc_membind_policy_t * policy, int flags);

/** \brief Set the default memory binding policy of the specified
 * process to prefer the NUMA node(s) specified by \p set
 *
 * If ::HWLOC_MEMBIND_BYNODESET is specified, set is considered a nodeset.
 * Otherwise it's a cpuset.
 *
 * \return -1 with errno set to ENOSYS if the action is not supported
 * \return -1 with errno set to EXDEV if the binding cannot be enforced
 *
 * \note \p hwloc_pid_t is \p pid_t on Unix platforms,
 * and \p HANDLE on native Windows platforms.
 */
HWLOC_DECLSPEC int hwloc_set_proc_membind(hwloc_topology_t topology, hwloc_pid_t pid, hwloc_const_bitmap_t set, hwloc_membind_policy_t policy, int flags);

/** \brief Query the default memory binding policy and physical locality of the
 * specified process.
 *
 * This function has two output parameters: \p set and \p policy.
 * The values returned in these parameters depend on both the \p flags
 * passed in and the current memory binding policies and nodesets in
 * the queried target.
 *
 * Passing the ::HWLOC_MEMBIND_PROCESS flag specifies that the query
 * target is the current policies and nodesets for all the threads in
 * the specified process.  If ::HWLOC_MEMBIND_PROCESS is not specified
 * (which is the most portable method), the process is assumed to be
 * single threaded.  This allows hwloc to use either process-based OS
 * functions or thread-based OS functions, depending on which are
 * available.
 *
 * Note that it does not make sense to pass ::HWLOC_MEMBIND_THREAD to
 * this function.
 *
 * If ::HWLOC_MEMBIND_STRICT is specified, hwloc will check the default
 * memory policies and nodesets for all threads in the specified
 * process.  If they are not identical, -1 is returned and errno is
 * set to EXDEV.  If they are identical, the values are returned in \p
 * set and \p policy.
 *
 * Otherwise, \p set is set to the logical OR of all threads'
 * default set.  If all threads' default policies
 * are the same, \p policy is set to that policy.  If they are
 * different, \p policy is set to ::HWLOC_MEMBIND_MIXED.
 *
 * If ::HWLOC_MEMBIND_BYNODESET is specified, set is considered a nodeset.
 * Otherwise it's a cpuset.
 *
 * If any other flags are specified, -1 is returned and errno is set
 * to EINVAL.
 *
 * \note \p hwloc_pid_t is \p pid_t on Unix platforms,
 * and \p HANDLE on native Windows platforms.
 */
HWLOC_DECLSPEC int hwloc_get_proc_membind(hwloc_topology_t topology, hwloc_pid_t pid, hwloc_bitmap_t set, hwloc_membind_policy_t * policy, int flags);

/** \brief Bind the already-allocated memory identified by (addr, len)
 * to the NUMA node(s) specified by \p set.
 *
 * If ::HWLOC_MEMBIND_BYNODESET is specified, set is considered a nodeset.
 * Otherwise it's a cpuset.
 *
 * \return 0 if \p len is 0.
 * \return -1 with errno set to ENOSYS if the action is not supported
 * \return -1 with errno set to EXDEV if the binding cannot be enforced
 */
HWLOC_DECLSPEC int hwloc_set_area_membind(hwloc_topology_t topology, const void *addr, size_t len, hwloc_const_bitmap_t set, hwloc_membind_policy_t policy, int flags);

/** \brief Query the CPUs near the physical NUMA node(s) and binding policy of
 * the memory identified by (\p addr, \p len ).
 *
 * This function has two output parameters: \p set and \p policy.
 * The values returned in these parameters depend on both the \p flags
 * passed in and the memory binding policies and nodesets of the pages
 * in the address range.
 *
 * If ::HWLOC_MEMBIND_STRICT is specified, the target pages are first
 * checked to see if they all have the same memory binding policy and
 * nodeset.  If they do not, -1 is returned and errno is set to EXDEV.
 * If they are identical across all pages, the set and policy are
 * returned in \p set and \p policy, respectively.
 *
 * If ::HWLOC_MEMBIND_STRICT is not specified, the union of all NUMA
 * node(s) containing pages in the address range is calculated.
 * If all pages in the target have the same policy, it is returned in
 * \p policy.  Otherwise, \p policy is set to ::HWLOC_MEMBIND_MIXED.
 *
 * If ::HWLOC_MEMBIND_BYNODESET is specified, set is considered a nodeset.
 * Otherwise it's a cpuset.
 *
 * If any other flags are specified, -1 is returned and errno is set
 * to EINVAL.
 *
 * If \p len is 0, -1 is returned and errno is set to EINVAL.
 */
HWLOC_DECLSPEC int hwloc_get_area_membind(hwloc_topology_t topology, const void *addr, size_t len, hwloc_bitmap_t set, hwloc_membind_policy_t * policy, int flags);

/** \brief Get the NUMA nodes where memory identified by (\p addr, \p len ) is physically allocated.
 *
 * Fills \p set according to the NUMA nodes where the memory area pages
 * are physically allocated. If no page is actually allocated yet,
 * \p set may be empty.
 *
 * If pages spread to multiple nodes, it is not specified whether they spread
 * equitably, or whether most of them are on a single node, etc.
 *
 * The operating system may move memory pages from one processor
 * to another at any time according to their binding,
 * so this function may return something that is already
 * outdated.
 *
 * If ::HWLOC_MEMBIND_BYNODESET is specified in \p flags, set is
 * considered a nodeset. Otherwise it's a cpuset.
 *
 * If \p len is 0, \p set is emptied.
 */
HWLOC_DECLSPEC int hwloc_get_area_memlocation(hwloc_topology_t topology, const void *addr, size_t len, hwloc_bitmap_t set, int flags);

/** \brief Allocate some memory
 *
 * This is equivalent to malloc(), except that it tries to allocate
 * page-aligned memory from the OS.
 *
 * \note The allocated memory should be freed with hwloc_free().
 */
HWLOC_DECLSPEC void *hwloc_alloc(hwloc_topology_t topology, size_t len);

/** \brief Allocate some memory on NUMA memory nodes specified by \p set
 *
 * \return NULL with errno set to ENOSYS if the action is not supported
 * and ::HWLOC_MEMBIND_STRICT is given
 * \return NULL with errno set to EXDEV if the binding cannot be enforced
 * and ::HWLOC_MEMBIND_STRICT is given
 * \return NULL with errno set to ENOMEM if the memory allocation failed
 * even before trying to bind.
 *
 * If ::HWLOC_MEMBIND_BYNODESET is specified, set is considered a nodeset.
 * Otherwise it's a cpuset.
 *
 * \note The allocated memory should be freed with hwloc_free().
 */
HWLOC_DECLSPEC void *hwloc_alloc_membind(hwloc_topology_t topology, size_t len, hwloc_const_bitmap_t set, hwloc_membind_policy_t policy, int flags) __hwloc_attribute_malloc;

/** \brief Allocate some memory on NUMA memory nodes specified by \p set
 *
 * This is similar to hwloc_alloc_membind_nodeset() except that it is allowed to change
 * the current memory binding policy, thus providing more binding support, at
 * the expense of changing the current state.
 *
 * If ::HWLOC_MEMBIND_BYNODESET is specified, set is considered a nodeset.
 * Otherwise it's a cpuset.
 */
static __hwloc_inline void *
hwloc_alloc_membind_policy(hwloc_topology_t topology, size_t len, hwloc_const_bitmap_t set, hwloc_membind_policy_t policy, int flags) __hwloc_attribute_malloc;

/** \brief Free memory that was previously allocated by hwloc_alloc()
 * or hwloc_alloc_membind().
 */
HWLOC_DECLSPEC int hwloc_free(hwloc_topology_t topology, void *addr, size_t len);

/** @} */



/** \defgroup hwlocality_setsource Changing the Source of Topology Discovery
 *
 * If none of the functions below is called, the default is to detect all the objects
 * of the machine that the caller is allowed to access.
 *
 * This default behavior may also be modified through environment variables
 * if the application did not modify it already.
 * Setting HWLOC_XMLFILE in the environment enforces the discovery from a XML
 * file as if hwloc_topology_set_xml() had been called.
 * Setting HWLOC_SYNTHETIC enforces a synthetic topology as if
 * hwloc_topology_set_synthetic() had been called.
 *
 * Finally, HWLOC_THISSYSTEM enforces the return value of
 * hwloc_topology_is_thissystem().
 *
 * @{
 */

/** \brief Change which process the topology is viewed from.
 *
 * On some systems, processes may have different views of the machine, for
 * instance the set of allowed CPUs. By default, hwloc exposes the view from
 * the current process. Calling hwloc_topology_set_pid() permits to make it
 * expose the topology of the machine from the point of view of another
 * process.
 *
 * \note \p hwloc_pid_t is \p pid_t on Unix platforms,
 * and \p HANDLE on native Windows platforms.
 *
 * \note -1 is returned and errno is set to ENOSYS on platforms that do not
 * support this feature.
 */
HWLOC_DECLSPEC int hwloc_topology_set_pid(hwloc_topology_t __hwloc_restrict topology, hwloc_pid_t pid);

/** \brief Enable synthetic topology.
 *
 * Gather topology information from the given \p description,
 * a space-separated string of <type:number> describing
 * the object type and arity at each level.
 * All types may be omitted (space-separated string of numbers) so that
 * hwloc chooses all types according to usual topologies.
 * See also the \ref synthetic.
 *
 * Setting the environment variable HWLOC_SYNTHETIC
 * may also result in this behavior.
 *
 * If \p description was properly parsed and describes a valid topology
 * configuration, this function returns 0.
 * Otherwise -1 is returned and errno is set to EINVAL.
 *
 * Note that this function does not actually load topology
 * information; it just tells hwloc where to load it from.  You'll
 * still need to invoke hwloc_topology_load() to actually load the
 * topology information.
 *
 * \note For convenience, this backend provides empty binding hooks which just
 * return success.
 *
 * \note On success, the synthetic component replaces the previously enabled
 * component (if any), but the topology is not actually modified until
 * hwloc_topology_load().
 */
HWLOC_DECLSPEC int hwloc_topology_set_synthetic(hwloc_topology_t __hwloc_restrict topology, const char * __hwloc_restrict description);

/** \brief Enable XML-file based topology.
 *
 * Gather topology information from the XML file given at \p xmlpath.
 * Setting the environment variable HWLOC_XMLFILE may also result in this behavior.
 * This file may have been generated earlier with hwloc_topology_export_xml() in hwloc/export.h,
 * or lstopo file.xml.
 *
 * Note that this function does not actually load topology
 * information; it just tells hwloc where to load it from.  You'll
 * still need to invoke hwloc_topology_load() to actually load the
 * topology information.
 *
 * \return -1 with errno set to EINVAL on failure to read the XML file.
 *
 * \note See also hwloc_topology_set_userdata_import_callback()
 * for importing application-specific object userdata.
 *
 * \note For convenience, this backend provides empty binding hooks which just
 * return success.  To have hwloc still actually call OS-specific hooks, the
 * ::HWLOC_TOPOLOGY_FLAG_IS_THISSYSTEM has to be set to assert that the loaded
 * file is really the underlying system.
 *
 * \note On success, the XML component replaces the previously enabled
 * component (if any), but the topology is not actually modified until
 * hwloc_topology_load().
 */
HWLOC_DECLSPEC int hwloc_topology_set_xml(hwloc_topology_t __hwloc_restrict topology, const char * __hwloc_restrict xmlpath);

/** \brief Enable XML based topology using a memory buffer (instead of
 * a file, as with hwloc_topology_set_xml()).
 *
 * Gather topology information from the XML memory buffer given at \p
 * buffer and of length \p size.  This buffer may have been filled
 * earlier with hwloc_topology_export_xmlbuffer() in hwloc/export.h.
 *
 * Note that this function does not actually load topology
 * information; it just tells hwloc where to load it from.  You'll
 * still need to invoke hwloc_topology_load() to actually load the
 * topology information.
 *
 * \return -1 with errno set to EINVAL on failure to read the XML buffer.
 *
 * \note See also hwloc_topology_set_userdata_import_callback()
 * for importing application-specific object userdata.
 *
 * \note For convenience, this backend provides empty binding hooks which just
 * return success.  To have hwloc still actually call OS-specific hooks, the
 * ::HWLOC_TOPOLOGY_FLAG_IS_THISSYSTEM has to be set to assert that the loaded
 * file is really the underlying system.
 *
 * \note On success, the XML component replaces the previously enabled
 * component (if any), but the topology is not actually modified until
 * hwloc_topology_load().
 */
HWLOC_DECLSPEC int hwloc_topology_set_xmlbuffer(hwloc_topology_t __hwloc_restrict topology, const char * __hwloc_restrict buffer, int size);

/** \brief Flags to be passed to hwloc_topology_set_components()
 */
enum hwloc_topology_components_flag_e {
  /** \brief Blacklist the target component from being used.
   * \hideinitializer
   */
  HWLOC_TOPOLOGY_COMPONENTS_FLAG_BLACKLIST = (1UL<<0)
};

/** \brief Prevent a discovery component from being used for a topology.
 *
 * \p name is the name of the discovery component that should not be used
 * when loading topology \p topology. The name is a string such as "cuda".
 *
 * For components with multiple phases, it may also be suffixed with the name
 * of a phase, for instance "linux:io".
 *
 * \p flags should be ::HWLOC_TOPOLOGY_COMPONENTS_FLAG_BLACKLIST.
 *
 * This may be used to avoid expensive parts of the discovery process.
 * For instance, CUDA-specific discovery may be expensive and unneeded
 * while generic I/O discovery could still be useful.
 */
HWLOC_DECLSPEC int hwloc_topology_set_components(hwloc_topology_t __hwloc_restrict topology, unsigned long flags, const char * __hwloc_restrict name);

/** @} */



/** \defgroup hwlocality_configuration Topology Detection Configuration and Query
 *
 * Several functions can optionally be called between hwloc_topology_init() and
 * hwloc_topology_load() to configure how the detection should be performed,
 * e.g. to ignore some objects types, define a synthetic topology, etc.
 *
 * @{
 */

/** \brief Flags to be set onto a topology context before load.
 *
 * Flags should be given to hwloc_topology_set_flags().
 * They may also be returned by hwloc_topology_get_flags().
 */
enum hwloc_topology_flags_e {
 /** \brief Detect the whole system, ignore reservations, include disallowed objects.
   *
   * Gather all resources, even if some were disabled by the administrator.
   * For instance, ignore Linux Cgroup/Cpusets and gather all processors and memory nodes.
   *
   * When this flag is not set, PUs and NUMA nodes that are disallowed are not added to the topology.
   * Parent objects (package, core, cache, etc.) are added only if some of their children are allowed.
   * All existing PUs and NUMA nodes in the topology are allowed.
   * hwloc_topology_get_allowed_cpuset() and hwloc_topology_get_allowed_nodeset()
   * are equal to the root object cpuset and nodeset.
   *
   * When this flag is set, the actual sets of allowed PUs and NUMA nodes are given
   * by hwloc_topology_get_allowed_cpuset() and hwloc_topology_get_allowed_nodeset().
   * They may be smaller than the root object cpuset and nodeset.
   *
   * If the current topology is exported to XML and reimported later, this flag
   * should be set again in the reimported topology so that disallowed resources
   * are reimported as well.
   * \hideinitializer
   */
  HWLOC_TOPOLOGY_FLAG_INCLUDE_DISALLOWED = (1UL<<0),

 /** \brief Assume that the selected backend provides the topology for the
   * system on which we are running.
   *
   * This forces hwloc_topology_is_thissystem() to return 1, i.e. makes hwloc assume that
   * the selected backend provides the topology for the system on which we are running,
   * even if it is not the OS-specific backend but the XML backend for instance.
   * This means making the binding functions actually call the OS-specific
   * system calls and really do binding, while the XML backend would otherwise
   * provide empty hooks just returning success.
   *
   * Setting the environment variable HWLOC_THISSYSTEM may also result in the
   * same behavior.
   *
   * This can be used for efficiency reasons to first detect the topology once,
   * save it to an XML file, and quickly reload it later through the XML
   * backend, but still having binding functions actually do bind.
   * \hideinitializer
   */
  HWLOC_TOPOLOGY_FLAG_IS_THISSYSTEM = (1UL<<1),

 /** \brief Get the set of allowed resources from the local operating system even if the topology was loaded from XML or synthetic description.
   *
   * If the topology was loaded from XML or from a synthetic string,
   * restrict it by applying the current process restrictions such as
   * Linux Cgroup/Cpuset.
   *
   * This is useful when the topology is not loaded directly from
   * the local machine (e.g. for performance reason) and it comes
   * with all resources, while the running process is restricted
   * to only parts of the machine.
   *
   * This flag is ignored unless ::HWLOC_TOPOLOGY_FLAG_IS_THISSYSTEM is
   * also set since the loaded topology must match the underlying machine
   * where restrictions will be gathered from.
   *
   * Setting the environment variable HWLOC_THISSYSTEM_ALLOWED_RESOURCES
   * would result in the same behavior.
   * \hideinitializer
   */
  HWLOC_TOPOLOGY_FLAG_THISSYSTEM_ALLOWED_RESOURCES = (1UL<<2),

  /** \brief Import support from the imported topology.
   *
   * When importing a XML topology from a remote machine, binding is
   * disabled by default (see ::HWLOC_TOPOLOGY_FLAG_IS_THISSYSTEM).
   * This disabling is also marked by putting zeroes in the corresponding
   * supported feature bits reported by hwloc_topology_get_support().
   *
   * The flag ::HWLOC_TOPOLOGY_FLAG_IMPORT_SUPPORT actually imports
   * support bits from the remote machine. It also sets the flag
   * \p imported_support in the struct hwloc_topology_misc_support array.
   * If the imported XML did not contain any support information
   * (exporter hwloc is too old), this flag is not set.
   *
   * Note that these supported features are only relevant for the hwloc
   * installation that actually exported the XML topology
   * (it may vary with the operating system, or with how hwloc was compiled).
   *
   * Note that setting this flag however does not enable binding for the
   * locally imported hwloc topology, it only reports what the remote
   * hwloc and machine support.
   *
   */
  HWLOC_TOPOLOGY_FLAG_IMPORT_SUPPORT = (1UL<<3)
};

/** \brief Set OR'ed flags to non-yet-loaded topology.
 *
 * Set a OR'ed set of ::hwloc_topology_flags_e onto a topology that was not yet loaded.
 *
 * If this function is called multiple times, the last invokation will erase
 * and replace the set of flags that was previously set.
 *
 * The flags set in a topology may be retrieved with hwloc_topology_get_flags()
 */
HWLOC_DECLSPEC int hwloc_topology_set_flags (hwloc_topology_t topology, unsigned long flags);

/** \brief Get OR'ed flags of a topology.
 *
 * Get the OR'ed set of ::hwloc_topology_flags_e of a topology.
 *
 * \return the flags previously set with hwloc_topology_set_flags().
 */
HWLOC_DECLSPEC unsigned long hwloc_topology_get_flags (hwloc_topology_t topology);

/** \brief Does the topology context come from this system?
 *
 * \return 1 if this topology context was built using the system
 * running this program.
 * \return 0 instead (for instance if using another file-system root,
 * a XML topology file, or a synthetic topology).
 */
HWLOC_DECLSPEC int hwloc_topology_is_thissystem(hwloc_topology_t  __hwloc_restrict topology) __hwloc_attribute_pure;

/** \brief Flags describing actual discovery support for this topology. */
struct hwloc_topology_discovery_support {
  /** \brief Detecting the number of PU objects is supported. */
  unsigned char pu;
  /** \brief Detecting the number of NUMA nodes is supported. */
  unsigned char numa;
  /** \brief Detecting the amount of memory in NUMA nodes is supported. */
  unsigned char numa_memory;
  /** \brief Detecting and identifying PU objects that are not available to the current process is supported. */
  unsigned char disallowed_pu;
  /** \brief Detecting and identifying NUMA nodes that are not available to the current process is supported. */
  unsigned char disallowed_numa;
  /** \brief Detecting the efficiency of CPU kinds is supported, see \ref hwlocality_cpukinds. */
  unsigned char cpukind_efficiency;
};

/** \brief Flags describing actual PU binding support for this topology.
 *
 * A flag may be set even if the feature isn't supported in all cases
 * (e.g. binding to random sets of non-contiguous objects).
 */
struct hwloc_topology_cpubind_support {
  /** Binding the whole current process is supported.  */
  unsigned char set_thisproc_cpubind;
  /** Getting the binding of the whole current process is supported.  */
  unsigned char get_thisproc_cpubind;
  /** Binding a whole given process is supported.  */
  unsigned char set_proc_cpubind;
  /** Getting the binding of a whole given process is supported.  */
  unsigned char get_proc_cpubind;
  /** Binding the current thread only is supported.  */
  unsigned char set_thisthread_cpubind;
  /** Getting the binding of the current thread only is supported.  */
  unsigned char get_thisthread_cpubind;
  /** Binding a given thread only is supported.  */
  unsigned char set_thread_cpubind;
  /** Getting the binding of a given thread only is supported.  */
  unsigned char get_thread_cpubind;
  /** Getting the last processors where the whole current process ran is supported */
  unsigned char get_thisproc_last_cpu_location;
  /** Getting the last processors where a whole process ran is supported */
  unsigned char get_proc_last_cpu_location;
  /** Getting the last processors where the current thread ran is supported */
  unsigned char get_thisthread_last_cpu_location;
};

/** \brief Flags describing actual memory binding support for this topology.
 *
 * A flag may be set even if the feature isn't supported in all cases
 * (e.g. binding to random sets of non-contiguous objects).
 */
struct hwloc_topology_membind_support {
  /** Binding the whole current process is supported.  */
  unsigned char set_thisproc_membind;
  /** Getting the binding of the whole current process is supported.  */
  unsigned char get_thisproc_membind;
  /** Binding a whole given process is supported.  */
  unsigned char set_proc_membind;
  /** Getting the binding of a whole given process is supported.  */
  unsigned char get_proc_membind;
  /** Binding the current thread only is supported.  */
  unsigned char set_thisthread_membind;
  /** Getting the binding of the current thread only is supported.  */
  unsigned char get_thisthread_membind;
  /** Binding a given memory area is supported. */
  unsigned char set_area_membind;
  /** Getting the binding of a given memory area is supported.  */
  unsigned char get_area_membind;
  /** Allocating a bound memory area is supported. */
  unsigned char alloc_membind;
  /** First-touch policy is supported. */
  unsigned char firsttouch_membind;
  /** Bind policy is supported. */
  unsigned char bind_membind;
  /** Interleave policy is supported. */
  unsigned char interleave_membind;
  /** Next-touch migration policy is supported. */
  unsigned char nexttouch_membind;
  /** Migration flags is supported. */
  unsigned char migrate_membind;
  /** Getting the last NUMA nodes where a memory area was allocated is supported */
  unsigned char get_area_memlocation;
};

/** \brief Flags describing miscellaneous features.
 */
struct hwloc_topology_misc_support {
  /** Support was imported when importing another topology, see ::HWLOC_TOPOLOGY_FLAG_IMPORT_SUPPORT. */
  unsigned char imported_support;
};

/** \brief Set of flags describing actual support for this topology.
 *
 * This is retrieved with hwloc_topology_get_support() and will be valid until
 * the topology object is destroyed.  Note: the values are correct only after
 * discovery.
 */
struct hwloc_topology_support {
  struct hwloc_topology_discovery_support *discovery;
  struct hwloc_topology_cpubind_support *cpubind;
  struct hwloc_topology_membind_support *membind;
  struct hwloc_topology_misc_support *misc;
};

/** \brief Retrieve the topology support.
 *
 * Each flag indicates whether a feature is supported.
 * If set to 0, the feature is not supported.
 * If set to 1, the feature is supported, but the corresponding
 * call may still fail in some corner cases.
 *
 * These features are also listed by hwloc-info \--support
 *
 * The reported features are what the current topology supports
 * on the current machine. If the topology was exported to XML
 * from another machine and later imported here, support still
 * describes what is supported for this imported topology after
 * import. By default, binding will be reported as unsupported
 * in this case (see ::HWLOC_TOPOLOGY_FLAG_IS_THISSYSTEM).
 *
 * Topology flag ::HWLOC_TOPOLOGY_FLAG_IMPORT_SUPPORT may be used
 * to report the supported features of the original remote machine
 * instead. If it was successfully imported, \p imported_support
 * will be set in the struct hwloc_topology_misc_support array.
 */
HWLOC_DECLSPEC const struct hwloc_topology_support *hwloc_topology_get_support(hwloc_topology_t __hwloc_restrict topology);

/** \brief Type filtering flags.
 *
 * By default, most objects are kept (::HWLOC_TYPE_FILTER_KEEP_ALL).
 * Instruction caches, I/O and Misc objects are ignored by default (::HWLOC_TYPE_FILTER_KEEP_NONE).
 * Die and Group levels are ignored unless they bring structure (::HWLOC_TYPE_FILTER_KEEP_STRUCTURE).
 *
 * Note that group objects are also ignored individually (without the entire level)
 * when they do not bring structure.
 */
enum hwloc_type_filter_e {
  /** \brief Keep all objects of this type.
   *
   * Cannot be set for ::HWLOC_OBJ_GROUP (groups are designed only to add more structure to the topology).
   * \hideinitializer
   */
  HWLOC_TYPE_FILTER_KEEP_ALL = 0,

  /** \brief Ignore all objects of this type.
   *
   * The bottom-level type ::HWLOC_OBJ_PU, the ::HWLOC_OBJ_NUMANODE type, and
   * the top-level type ::HWLOC_OBJ_MACHINE may not be ignored.
   * \hideinitializer
   */
  HWLOC_TYPE_FILTER_KEEP_NONE = 1,

  /** \brief Only ignore objects if their entire level does not bring any structure.
   *
   * Keep the entire level of objects if at least one of these objects adds
   * structure to the topology. An object brings structure when it has multiple
   * children and it is not the only child of its parent.
   *
   * If all objects in the level are the only child of their parent, and if none
   * of them has multiple children, the entire level is removed.
   *
   * Cannot be set for I/O and Misc objects since the topology structure does not matter there.
   * \hideinitializer
   */
  HWLOC_TYPE_FILTER_KEEP_STRUCTURE = 2,

  /** \brief Only keep likely-important objects of the given type.
   *
   * It is only useful for I/O object types.
   * For ::HWLOC_OBJ_PCI_DEVICE and ::HWLOC_OBJ_OS_DEVICE, it means that only objects
   * of major/common kinds are kept (storage, network, OpenFabrics, CUDA,
   * OpenCL, RSMI, NVML, and displays).
   * Also, only OS devices directly attached on PCI (e.g. no USB) are reported.
   * For ::HWLOC_OBJ_BRIDGE, it means that bridges are kept only if they have children.
   *
   * This flag equivalent to ::HWLOC_TYPE_FILTER_KEEP_ALL for Normal, Memory and Misc types
   * since they are likely important.
   * \hideinitializer
   */
  HWLOC_TYPE_FILTER_KEEP_IMPORTANT = 3
};

/** \brief Set the filtering for the given object type.
 */
HWLOC_DECLSPEC int hwloc_topology_set_type_filter(hwloc_topology_t topology, hwloc_obj_type_t type, enum hwloc_type_filter_e filter);

/** \brief Get the current filtering for the given object type.
 */
HWLOC_DECLSPEC int hwloc_topology_get_type_filter(hwloc_topology_t topology, hwloc_obj_type_t type, enum hwloc_type_filter_e *filter);

/** \brief Set the filtering for all object types.
 *
 * If some types do not support this filtering, they are silently ignored.
 */
HWLOC_DECLSPEC int hwloc_topology_set_all_types_filter(hwloc_topology_t topology, enum hwloc_type_filter_e filter);

/** \brief Set the filtering for all CPU cache object types.
 *
 * Memory-side caches are not involved since they are not CPU caches.
 */
HWLOC_DECLSPEC int hwloc_topology_set_cache_types_filter(hwloc_topology_t topology, enum hwloc_type_filter_e filter);

/** \brief Set the filtering for all CPU instruction cache object types.
 *
 * Memory-side caches are not involved since they are not CPU caches.
 */
HWLOC_DECLSPEC int hwloc_topology_set_icache_types_filter(hwloc_topology_t topology, enum hwloc_type_filter_e filter);

/** \brief Set the filtering for all I/O object types.
 */
HWLOC_DECLSPEC int hwloc_topology_set_io_types_filter(hwloc_topology_t topology, enum hwloc_type_filter_e filter);

/** \brief Set the topology-specific userdata pointer.
 *
 * Each topology may store one application-given private data pointer.
 * It is initialized to \c NULL.
 * hwloc will never modify it.
 *
 * Use it as you wish, after hwloc_topology_init() and until hwloc_topolog_destroy().
 *
 * This pointer is not exported to XML.
 */
HWLOC_DECLSPEC void hwloc_topology_set_userdata(hwloc_topology_t topology, const void *userdata);

/** \brief Retrieve the topology-specific userdata pointer.
 *
 * Retrieve the application-given private data pointer that was
 * previously set with hwloc_topology_set_userdata().
 */
HWLOC_DECLSPEC void * hwloc_topology_get_userdata(hwloc_topology_t topology);

/** @} */



/** \defgroup hwlocality_tinker Modifying a loaded Topology
 * @{
 */

/** \brief Flags to be given to hwloc_topology_restrict(). */
enum hwloc_restrict_flags_e {
  /** \brief Remove all objects that became CPU-less.
   * By default, only objects that contain no PU and no memory are removed.
   * This flag may not be used with ::HWLOC_RESTRICT_FLAG_BYNODESET.
   * \hideinitializer
   */
  HWLOC_RESTRICT_FLAG_REMOVE_CPULESS = (1UL<<0),

  /** \brief Restrict by nodeset instead of CPU set.
   * Only keep objects whose nodeset is included or partially included in the given set.
   * This flag may not be used with ::HWLOC_RESTRICT_FLAG_REMOVE_CPULESS.
   */
  HWLOC_RESTRICT_FLAG_BYNODESET =  (1UL<<3),

  /** \brief Remove all objects that became Memory-less.
   * By default, only objects that contain no PU and no memory are removed.
   * This flag may only be used with ::HWLOC_RESTRICT_FLAG_BYNODESET.
   * \hideinitializer
   */
  HWLOC_RESTRICT_FLAG_REMOVE_MEMLESS = (1UL<<4),

  /** \brief Move Misc objects to ancestors if their parents are removed during restriction.
   * If this flag is not set, Misc objects are removed when their parents are removed.
   * \hideinitializer
   */
  HWLOC_RESTRICT_FLAG_ADAPT_MISC = (1UL<<1),

  /** \brief Move I/O objects to ancestors if their parents are removed during restriction.
   * If this flag is not set, I/O devices and bridges are removed when their parents are removed.
   * \hideinitializer
   */
  HWLOC_RESTRICT_FLAG_ADAPT_IO = (1UL<<2)
};

/** \brief Restrict the topology to the given CPU set or nodeset.
 *
 * Topology \p topology is modified so as to remove all objects that
 * are not included (or partially included) in the CPU set \p set.
 * All objects CPU and node sets are restricted accordingly.
 *
 * If ::HWLOC_RESTRICT_FLAG_BYNODESET is passed in \p flags,
 * \p set is considered a nodeset instead of a CPU set.
 *
 * \p flags is a OR'ed set of ::hwloc_restrict_flags_e.
 *
 * \note This call may not be reverted by restricting back to a larger
 * set. Once dropped during restriction, objects may not be brought
 * back, except by loading another topology with hwloc_topology_load().
 *
 * \return 0 on success.
 *
 * \return -1 with errno set to EINVAL if the input set is invalid.
 * The topology is not modified in this case.
 *
 * \return -1 with errno set to ENOMEM on failure to allocate internal data.
 * The topology is reinitialized in this case. It should be either
 * destroyed with hwloc_topology_destroy() or configured and loaded again.
 */
HWLOC_DECLSPEC int hwloc_topology_restrict(hwloc_topology_t __hwloc_restrict topology, hwloc_const_bitmap_t set, unsigned long flags);

/** \brief Flags to be given to hwloc_topology_allow(). */
enum hwloc_allow_flags_e {
  /** \brief Mark all objects as allowed in the topology.
   *
   * \p cpuset and \p nođeset given to hwloc_topology_allow() must be \c NULL.
   * \hideinitializer */
  HWLOC_ALLOW_FLAG_ALL = (1UL<<0),

  /** \brief Only allow objects that are available to the current process.
   *
   * The topology must have ::HWLOC_TOPOLOGY_FLAG_IS_THISSYSTEM so that the set
   * of available resources can actually be retrieved from the operating system.
   *
   * \p cpuset and \p nođeset given to hwloc_topology_allow() must be \c NULL.
   * \hideinitializer */
  HWLOC_ALLOW_FLAG_LOCAL_RESTRICTIONS = (1UL<<1),

  /** \brief Allow a custom set of objects, given to hwloc_topology_allow() as \p cpuset and/or \p nodeset parameters.
   * \hideinitializer */
  HWLOC_ALLOW_FLAG_CUSTOM = (1UL<<2)
};

/** \brief Change the sets of allowed PUs and NUMA nodes in the topology.
 *
 * This function only works if the ::HWLOC_TOPOLOGY_FLAG_INCLUDE_DISALLOWED
 * was set on the topology. It does not modify any object, it only changes
 * the sets returned by hwloc_topology_get_allowed_cpuset() and
 * hwloc_topology_get_allowed_nodeset().
 *
 * It is notably useful when importing a topology from another process
 * running in a different Linux Cgroup.
 *
 * \p flags must be set to one flag among ::hwloc_allow_flags_e.
 *
 * \note Removing objects from a topology should rather be performed with
 * hwloc_topology_restrict().
 */
HWLOC_DECLSPEC int hwloc_topology_allow(hwloc_topology_t __hwloc_restrict topology, hwloc_const_cpuset_t cpuset, hwloc_const_nodeset_t nodeset, unsigned long flags);

/** \brief Add a MISC object as a leaf of the topology
 *
 * A new MISC object will be created and inserted into the topology at the
 * position given by parent. It is appended to the list of existing Misc children,
 * without ever adding any intermediate hierarchy level. This is useful for
 * annotating the topology without actually changing the hierarchy.
 *
 * \p name is supposed to be unique across all Misc objects in the topology.
 * It will be duplicated to setup the new object attributes.
 *
 * The new leaf object will not have any \p cpuset.
 *
 * \return the newly-created object
 *
 * \return \c NULL on error.
 *
 * \return \c NULL if Misc objects are filtered-out of the topology (::HWLOC_TYPE_FILTER_KEEP_NONE).
 *
 * \note If \p name contains some non-printable characters, they will
 * be dropped when exporting to XML, see hwloc_topology_export_xml() in hwloc/export.h.
 */
HWLOC_DECLSPEC hwloc_obj_t hwloc_topology_insert_misc_object(hwloc_topology_t topology, hwloc_obj_t parent, const char *name);

/** \brief Allocate a Group object to insert later with hwloc_topology_insert_group_object().
 *
 * This function returns a new Group object.
 *
 * The caller should (at least) initialize its sets before inserting
 * the object in the topology. See hwloc_topology_insert_group_object().
 */
HWLOC_DECLSPEC hwloc_obj_t hwloc_topology_alloc_group_object(hwloc_topology_t topology);

/** \brief Add more structure to the topology by adding an intermediate Group
 *
 * The caller should first allocate a new Group object with hwloc_topology_alloc_group_object().
 * Then it must setup at least one of its CPU or node sets to specify
 * the final location of the Group in the topology.
 * Then the object can be passed to this function for actual insertion in the topology.
 *
 * Either the cpuset or nodeset field (or both, if compatible) must be set
 * to a non-empty bitmap. The complete_cpuset or complete_nodeset may be set
 * instead if inserting with respect to the complete topology
 * (including disallowed, offline or unknown objects).
 * If grouping several objects, hwloc_obj_add_other_obj_sets() is an easy way
 * to build the Group sets iteratively.
 * These sets cannot be larger than the current topology, or they would get
 * restricted silently.
 * The core will setup the other sets after actual insertion.
 *
 * The \p subtype object attribute may be defined (to a dynamically
 * allocated string) to display something else than "Group" as the
 * type name for this object in lstopo.
 * Custom name/value info pairs may be added with hwloc_obj_add_info() after
 * insertion.
 *
 * The group \p dont_merge attribute may be set to \c 1 to prevent
 * the hwloc core from ever merging this object with another
 * hierarchically-identical object.
 * This is useful when the Group itself describes an important feature
 * that cannot be exposed anywhere else in the hierarchy.
 *
 * The group \p kind attribute may be set to a high value such
 * as \c 0xffffffff to tell hwloc that this new Group should always
 * be discarded in favor of any existing Group with the same locality.
 *
 * \return The inserted object if it was properly inserted.
 *
 * \return An existing object if the Group was merged or discarded
 * because the topology already contained an object at the same
 * location (the Group did not add any hierarchy information).
 *
 * \return \c NULL if the insertion failed because of conflicting sets in topology tree.
 *
 * \return \c NULL if Group objects are filtered-out of the topology (::HWLOC_TYPE_FILTER_KEEP_NONE).
 *
 * \return \c NULL if the object was discarded because no set was
 * initialized in the Group before insert, or all of them were empty.
 */
HWLOC_DECLSPEC hwloc_obj_t hwloc_topology_insert_group_object(hwloc_topology_t topology, hwloc_obj_t group);

/** \brief Setup object cpusets/nodesets by OR'ing another object's sets.
 *
 * For each defined cpuset or nodeset in \p src, allocate the corresponding set
 * in \p dst and add \p src to it by OR'ing sets.
 *
 * This function is convenient between hwloc_topology_alloc_group_object()
 * and hwloc_topology_insert_group_object(). It builds the sets of the new Group
 * that will be inserted as a new intermediate parent of several objects.
 */
HWLOC_DECLSPEC int hwloc_obj_add_other_obj_sets(hwloc_obj_t dst, hwloc_obj_t src);

/** \brief Refresh internal structures after topology modification.
 *
 * Modifying the topology (by restricting, adding objects, modifying structures
 * such as distances or memory attributes, etc.) may cause some internal caches
 * to become invalid. These caches are automatically refreshed when accessed
 * but this refreshing is not thread-safe.
 *
 * This function is not thread-safe either, but it is a good way to end a
 * non-thread-safe phase of topology modification. Once this refresh is done,
 * multiple threads may concurrently consult the topology, objects, distances,
 * attributes, etc.
 *
 * See also \ref threadsafety
 */
HWLOC_DECLSPEC int hwloc_topology_refresh(hwloc_topology_t topology);

/** @} */



#ifdef __cplusplus
} /* extern "C" */
#endif


/* high-level helpers */
#include "hwloc/helper.h"

/* inline code of some functions above */
#include "hwloc/inlines.h"

/* memory attributes */
#include "hwloc/memattrs.h"

/* kinds of CPU cores */
#include "hwloc/cpukinds.h"

/* exporting to XML or synthetic */
#include "hwloc/export.h"

/* distances */
#include "hwloc/distances.h"

/* topology diffs */
#include "hwloc/diff.h"

/* deprecated headers */
#include "hwloc/deprecated.h"

#endif /* HWLOC_H */
