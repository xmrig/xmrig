/*
 * Copyright © 2019-2024 Inria.  All rights reserved.
 * See COPYING in top-level directory.
 */

/** \file
 * \brief Memory node attributes.
 */

#ifndef HWLOC_MEMATTR_H
#define HWLOC_MEMATTR_H

#include "hwloc.h"

#ifdef __cplusplus
extern "C" {
#elif 0
}
#endif

/** \defgroup hwlocality_memattrs Comparing memory node attributes for finding where to allocate on
 *
 * Platforms with heterogeneous memory require ways to decide whether
 * a buffer should be allocated on "fast" memory (such as HBM),
 * "normal" memory (DDR) or even "slow" but large-capacity memory
 * (non-volatile memory).
 * These memory nodes are called "Targets" while the CPU accessing them
 * is called the "Initiator". Access performance depends on their
 * locality (NUMA platforms) as well as the intrinsic performance
 * of the targets (heterogeneous platforms).
 *
 * The following attributes describe the performance of memory accesses
 * from an Initiator to a memory Target, for instance their latency
 * or bandwidth.
 * Initiators performing these memory accesses are usually some PUs or Cores
 * (described as a CPU set).
 * Hence a Core may choose where to allocate a memory buffer by comparing
 * the attributes of different target memory nodes nearby.
 *
 * There are also some attributes that are system-wide.
 * Their value does not depend on a specific initiator performing
 * an access.
 * The memory node Capacity is an example of such attribute without
 * initiator.
 *
 * One way to use this API is to start with a cpuset describing the Cores where
 * a program is bound. The best target NUMA node for allocating memory in this
 * program on these Cores may be obtained by passing this cpuset as an initiator
 * to hwloc_memattr_get_best_target() with the relevant memory attribute.
 * For instance, if the code is latency limited, use the Latency attribute.
 *
 * A more flexible approach consists in getting the list of local NUMA nodes
 * by passing this cpuset to hwloc_get_local_numanode_objs().
 * Attribute values for these nodes, if any, may then be obtained with
 * hwloc_memattr_get_value() and manually compared with the desired criteria.
 *
 * Memory attributes are also used internally to build Memory Tiers which provide
 * an easy way to distinguish NUMA nodes of different kinds, as explained
 * in \ref heteromem.
 *
 * \sa An example is available in doc/examples/memory-attributes.c in the source tree.
 *
 * \note The API also supports specific objects as initiator,
 * but it is currently not used internally by hwloc.
 * Users may for instance use it to provide custom performance
 * values for host memory accesses performed by GPUs.
 *
 * \note The interface actually also accepts targets that are not NUMA nodes.
 * @{
 */

/** \brief Predefined memory attribute IDs.
 * See ::hwloc_memattr_id_t for the generic definition of IDs
 * for predefined or custom attributes.
 */
enum hwloc_memattr_id_e {
  /** \brief
   * The \"Capacity\" is returned in bytes (local_memory attribute in objects).
   *
   * Best capacity nodes are nodes with <b>higher capacity</b>.
   *
   * No initiator is involved when looking at this attribute.
   * The corresponding attribute flags are ::HWLOC_MEMATTR_FLAG_HIGHER_FIRST.
   *
   * Capacity values may not be modified using hwloc_memattr_set_value().
   * \hideinitializer
   */
  HWLOC_MEMATTR_ID_CAPACITY = 0,

  /** \brief
   * The \"Locality\" is returned as the number of PUs in that locality
   * (e.g. the weight of its cpuset).
   *
   * Best locality nodes are nodes with <b>smaller locality</b>
   * (nodes that are local to very few PUs).
   * Poor locality nodes are nodes with larger locality
   * (nodes that are local to the entire machine).
   *
   * No initiator is involved when looking at this attribute.
   * The corresponding attribute flags are ::HWLOC_MEMATTR_FLAG_HIGHER_FIRST.

   * Locality values may not be modified using hwloc_memattr_set_value().
   * \hideinitializer
   */
  HWLOC_MEMATTR_ID_LOCALITY = 1,

  /** \brief
   * The \"Bandwidth\" is returned in MiB/s, as seen from the given initiator location.
   *
   * Best bandwidth nodes are nodes with <b>higher bandwidth</b>.
   *
   * The corresponding attribute flags are ::HWLOC_MEMATTR_FLAG_HIGHER_FIRST
   * and ::HWLOC_MEMATTR_FLAG_NEED_INITIATOR.
   *
   * This is the average bandwidth for read and write accesses. If the platform
   * provides individual read and write bandwidths but no explicit average value,
   * hwloc computes and returns the average.
   * \hideinitializer
   */
  HWLOC_MEMATTR_ID_BANDWIDTH = 2,

  /** \brief
   * The \"ReadBandwidth\" is returned in MiB/s, as seen from the given initiator location.
   *
   * Best bandwidth nodes are nodes with <b>higher bandwidth</b>.
   *
   * The corresponding attribute flags are ::HWLOC_MEMATTR_FLAG_HIGHER_FIRST
   * and ::HWLOC_MEMATTR_FLAG_NEED_INITIATOR.
   * \hideinitializer
   */
  HWLOC_MEMATTR_ID_READ_BANDWIDTH = 4,

  /** \brief
   * The \"WriteBandwidth\" is returned in MiB/s, as seen from the given initiator location.
   *
   * Best bandwidth nodes are nodes with <b>higher bandwidth</b>.
   *
   * The corresponding attribute flags are ::HWLOC_MEMATTR_FLAG_HIGHER_FIRST
   * and ::HWLOC_MEMATTR_FLAG_NEED_INITIATOR.
   * \hideinitializer
   */
  HWLOC_MEMATTR_ID_WRITE_BANDWIDTH = 5,

  /** \brief
   * The \"Latency\" is returned as nanoseconds, as seen from the given initiator location.
   *
   * Best latency nodes are nodes with <b>smaller latency</b>.
   *
   * The corresponding attribute flags are ::HWLOC_MEMATTR_FLAG_LOWER_FIRST
   * and ::HWLOC_MEMATTR_FLAG_NEED_INITIATOR.
   *
   * This is the average latency for read and write accesses. If the platform
   * provides individual read and write latencies but no explicit average value,
   * hwloc computes and returns the average.
   * \hideinitializer
   */
  HWLOC_MEMATTR_ID_LATENCY = 3,

  /** \brief
   * The \"ReadLatency\" is returned as nanoseconds, as seen from the given initiator location.
   *
   * Best latency nodes are nodes with <b>smaller latency</b>.
   *
   * The corresponding attribute flags are ::HWLOC_MEMATTR_FLAG_LOWER_FIRST
   * and ::HWLOC_MEMATTR_FLAG_NEED_INITIATOR.
   * \hideinitializer
   */
  HWLOC_MEMATTR_ID_READ_LATENCY = 6,

  /** \brief
   * The \"WriteLatency\" is returned as nanoseconds, as seen from the given initiator location.
   *
   * Best latency nodes are nodes with <b>smaller latency</b>.
   *
   * The corresponding attribute flags are ::HWLOC_MEMATTR_FLAG_LOWER_FIRST
   * and ::HWLOC_MEMATTR_FLAG_NEED_INITIATOR.
   * \hideinitializer
   */
  HWLOC_MEMATTR_ID_WRITE_LATENCY = 7,

  /* TODO persistence? */

  HWLOC_MEMATTR_ID_MAX /**< \private
                        * Sentinel value for predefined attributes.
                        * Dynamically registered custom attributes start here.
                        */
};

/** \brief A memory attribute identifier.
 *
 * hwloc predefines some commonly-used attributes in ::hwloc_memattr_id_e.
 * One may then dynamically register custom ones with hwloc_memattr_register(),
 * they will be assigned IDs immediately after the predefined ones.
 * See \ref hwlocality_memattrs_manage for more information about
 * existing attribute IDs.
 */
typedef unsigned hwloc_memattr_id_t;

/** \brief Return the identifier of the memory attribute with the given name.
 *
 * \return 0 on success.
 * \return -1 with errno set to \c EINVAL if no such attribute exists.
 */
HWLOC_DECLSPEC int
hwloc_memattr_get_by_name(hwloc_topology_t topology,
                          const char *name,
                          hwloc_memattr_id_t *id);


/** \brief Type of location. */
enum hwloc_location_type_e {
  /** \brief Location is given as a cpuset, in the location cpuset union field. \hideinitializer */
  HWLOC_LOCATION_TYPE_CPUSET = 1,
  /** \brief Location is given as an object, in the location object union field. \hideinitializer */
  HWLOC_LOCATION_TYPE_OBJECT = 0
};

/** \brief Where to measure attributes from. */
struct hwloc_location {
  /** \brief Type of location. */
  enum hwloc_location_type_e type;
  /** \brief Actual location. */
  union hwloc_location_u {
    /** \brief Location as a cpuset, when the location type is ::HWLOC_LOCATION_TYPE_CPUSET. */
    hwloc_cpuset_t cpuset;
    /** \brief Location as an object, when the location type is ::HWLOC_LOCATION_TYPE_OBJECT. */
    hwloc_obj_t object;
  } location;
};


/** \brief Flags for selecting target NUMA nodes. */
enum hwloc_local_numanode_flag_e {
  /** \brief Select NUMA nodes whose locality is larger than the given cpuset.
   * For instance, if a single PU (or its cpuset) is given in \p initiator,
   * select all nodes close to the package that contains this PU.
   * \hideinitializer
   */
  HWLOC_LOCAL_NUMANODE_FLAG_LARGER_LOCALITY = (1UL<<0),

  /** \brief Select NUMA nodes whose locality is smaller than the given cpuset.
   * For instance, if a package (or its cpuset) is given in \p initiator,
   * also select nodes that are attached to only a half of that package.
   * \hideinitializer
   */
  HWLOC_LOCAL_NUMANODE_FLAG_SMALLER_LOCALITY = (1UL<<1),

  /** \brief Select all NUMA nodes in the topology.
   * The initiator \p initiator is ignored.
   * \hideinitializer
   */
  HWLOC_LOCAL_NUMANODE_FLAG_ALL = (1UL<<2)
};

/** \brief Return an array of local NUMA nodes.
 *
 * By default only select the NUMA nodes whose locality is exactly
 * the given \p location. More nodes may be selected if additional flags
 * are given as a OR'ed set of ::hwloc_local_numanode_flag_e.
 *
 * If \p location is given as an explicit object, its CPU set is used
 * to find NUMA nodes with the corresponding locality.
 * If the object does not have a CPU set (e.g. I/O object), the CPU
 * parent (where the I/O object is attached) is used.
 *
 * On input, \p nr points to the number of nodes that may be stored
 * in the \p nodes array.
 * On output, \p nr will be changed to the number of stored nodes,
 * or the number of nodes that would have been stored if there were
 * enough room.
 *
 * \return 0 on success or -1 on error.
 *
 * \note Some of these NUMA nodes may not have any memory attribute
 * values and hence not be reported as actual targets in other functions.
 *
 * \note The number of NUMA nodes in the topology (obtained by
 * hwloc_bitmap_weight() on the root object nodeset) may be used
 * to allocate the \p nodes array.
 *
 * \note When an object CPU set is given as locality, for instance a Package,
 * and when flags contain both ::HWLOC_LOCAL_NUMANODE_FLAG_LARGER_LOCALITY
 * and ::HWLOC_LOCAL_NUMANODE_FLAG_SMALLER_LOCALITY,
 * the returned array corresponds to the nodeset of that object.
 */
HWLOC_DECLSPEC int
hwloc_get_local_numanode_objs(hwloc_topology_t topology,
                              struct hwloc_location *location,
                              unsigned *nr,
                              hwloc_obj_t *nodes,
                              unsigned long flags);



/** \brief Return an attribute value for a specific target NUMA node.
 *
 * If the attribute does not relate to a specific initiator
 * (it does not have the flag ::HWLOC_MEMATTR_FLAG_NEED_INITIATOR),
 * location \p initiator is ignored and may be \c NULL.
 *
 * \p target_node cannot be \c NULL. If \p attribute is ::HWLOC_MEMATTR_ID_CAPACITY,
 * \p target_node must be a NUMA node. If it is ::HWLOC_MEMATTR_ID_LOCALITY,
 * \p target_node must have a CPU set.
 *
 * \p flags must be \c 0 for now.
 *
 * \return 0 on success.
 * \return -1 on error, for instance with errno set to \c EINVAL if flags
 * are invalid or no such attribute exists.
 *
 * \note The initiator \p initiator should be of type ::HWLOC_LOCATION_TYPE_CPUSET
 * when refering to accesses performed by CPU cores.
 * ::HWLOC_LOCATION_TYPE_OBJECT is currently unused internally by hwloc,
 * but users may for instance use it to provide custom information about
 * host memory accesses performed by GPUs.
 */
HWLOC_DECLSPEC int
hwloc_memattr_get_value(hwloc_topology_t topology,
                        hwloc_memattr_id_t attribute,
                        hwloc_obj_t target_node,
                        struct hwloc_location *initiator,
                        unsigned long flags,
                        hwloc_uint64_t *value);

/** \brief Return the best target NUMA node for the given attribute and initiator.
 *
 * If the attribute does not relate to a specific initiator
 * (it does not have the flag ::HWLOC_MEMATTR_FLAG_NEED_INITIATOR),
 * location \p initiator is ignored and may be \c NULL.
 *
 * If \p value is non \c NULL, the corresponding value is returned there.
 *
 * If multiple targets have the same attribute values, only one is
 * returned (and there is no way to clarify how that one is chosen).
 * Applications that want to detect targets with identical/similar
 * values, or that want to look at values for multiple attributes,
 * should rather get all values using hwloc_memattr_get_value()
 * and manually select the target they consider the best.
 *
 * \p flags must be \c 0 for now.
 *
 * \return 0 on success.
 * \return -1 with errno set to \c ENOENT if there are no matching targets.
 * \return -1 with errno set to \c EINVAL if flags are invalid,
 * or no such attribute exists.
 *
 * \note The initiator \p initiator should be of type ::HWLOC_LOCATION_TYPE_CPUSET
 * when refering to accesses performed by CPU cores.
 * ::HWLOC_LOCATION_TYPE_OBJECT is currently unused internally by hwloc,
 * but users may for instance use it to provide custom information about
 * host memory accesses performed by GPUs.
 */
HWLOC_DECLSPEC int
hwloc_memattr_get_best_target(hwloc_topology_t topology,
                              hwloc_memattr_id_t attribute,
                              struct hwloc_location *initiator,
                              unsigned long flags,
                              hwloc_obj_t *best_target, hwloc_uint64_t *value);

/** \brief Return the best initiator for the given attribute and target NUMA node.
 *
 * If \p value is non \c NULL, the corresponding value is returned there.
 *
 * If multiple initiators have the same attribute values, only one is
 * returned (and there is no way to clarify how that one is chosen).
 * Applications that want to detect initiators with identical/similar
 * values, or that want to look at values for multiple attributes,
 * should rather get all values using hwloc_memattr_get_value()
 * and manually select the initiator they consider the best.
 *
 * The returned initiator should not be modified or freed,
 * it belongs to the topology.
 *
 * \p target_node cannot be \c NULL.
 *
 * \p flags must be \c 0 for now.
 *
 * \return 0 on success.
 * \return -1 with errno set to \c ENOENT if there are no matching initiators.
 * \return -1 with errno set to \c EINVAL if the attribute does not relate to a specific initiator
 * (it does not have the flag ::HWLOC_MEMATTR_FLAG_NEED_INITIATOR).
 */
HWLOC_DECLSPEC int
hwloc_memattr_get_best_initiator(hwloc_topology_t topology,
                                 hwloc_memattr_id_t attribute,
                                 hwloc_obj_t target_node,
                                 unsigned long flags,
                                 struct hwloc_location *best_initiator, hwloc_uint64_t *value);

/** \brief Return the target NUMA nodes that have some values for a given attribute.
 *
 * Return targets for the given attribute in the \p targets array
 * (for the given initiator if any).
 * If \p values is not \c NULL, the corresponding attribute values
 * are stored in the array it points to.
 *
 * On input, \p nr points to the number of targets that may be stored
 * in the array \p targets (and \p values).
 * On output, \p nr points to the number of targets (and values) that
 * were actually found, even if some of them couldn't be stored in the array.
 * Targets that couldn't be stored are ignored, but the function still
 * returns success (\c 0). The caller may find out by comparing the value pointed
 * by \p nr before and after the function call.
 *
 * The returned targets should not be modified or freed,
 * they belong to the topology.
 *
 * Argument \p initiator is ignored if the attribute does not relate to a specific
 * initiator (it does not have the flag ::HWLOC_MEMATTR_FLAG_NEED_INITIATOR).
 * Otherwise \p initiator may be non \c NULL to report only targets
 * that have a value for that initiator.
 *
 * \p flags must be \c 0 for now.
 *
 * \note This function is meant for tools and debugging (listing internal information)
 * rather than for application queries. Applications should rather select useful
 * NUMA nodes with hwloc_get_local_numanode_objs() and then look at their attribute
 * values.
 *
 * \return 0 on success or -1 on error.
 *
 * \note The initiator \p initiator should be of type ::HWLOC_LOCATION_TYPE_CPUSET
 * when referring to accesses performed by CPU cores.
 * ::HWLOC_LOCATION_TYPE_OBJECT is currently unused internally by hwloc,
 * but users may for instance use it to provide custom information about
 * host memory accesses performed by GPUs.
 */
HWLOC_DECLSPEC int
hwloc_memattr_get_targets(hwloc_topology_t topology,
                          hwloc_memattr_id_t attribute,
                          struct hwloc_location *initiator,
                          unsigned long flags,
                          unsigned *nr, hwloc_obj_t *targets, hwloc_uint64_t *values);

/** \brief Return the initiators that have values for a given attribute for a specific target NUMA node.
 *
 * Return initiators for the given attribute and target node in the
 * \p initiators array.
 * If \p values is not \c NULL, the corresponding attribute values
 * are stored in the array it points to.
 *
 * On input, \p nr points to the number of initiators that may be stored
 * in the array \p initiators (and \p values).
 * On output, \p nr points to the number of initiators (and values) that
 * were actually found, even if some of them couldn't be stored in the array.
 * Initiators that couldn't be stored are ignored, but the function still
 * returns success (\c 0). The caller may find out by comparing the value pointed
 * by \p nr before and after the function call.
 *
 * The returned initiators should not be modified or freed,
 * they belong to the topology.
 *
 * \p target_node cannot be \c NULL.
 *
 * \p flags must be \c 0 for now.
 *
 * If the attribute does not relate to a specific initiator
 * (it does not have the flag ::HWLOC_MEMATTR_FLAG_NEED_INITIATOR),
 * no initiator is returned.
 *
 * \return 0 on success or -1 on error.
 *
 * \note This function is meant for tools and debugging (listing internal information)
 * rather than for application queries. Applications should rather select useful
 * NUMA nodes with hwloc_get_local_numanode_objs() and then look at their attribute
 * values for some relevant initiators.
 */
HWLOC_DECLSPEC int
hwloc_memattr_get_initiators(hwloc_topology_t topology,
                             hwloc_memattr_id_t attribute,
                             hwloc_obj_t target_node,
                             unsigned long flags,
                             unsigned *nr, struct hwloc_location *initiators, hwloc_uint64_t *values);

/** @} */


/** \defgroup hwlocality_memattrs_manage Managing memory attributes
 *
 * Memory attribues are identified by an ID (::hwloc_memattr_id_t)
 * and a name. hwloc_memattr_get_name() and hwloc_memattr_get_by_name()
 * convert between them (or return error if the attribute does not exist).
 *
 * The set of valid ::hwloc_memattr_id_t is a contigous set starting at \c 0.
 * It first contains predefined attributes, as listed
 * in ::hwloc_memattr_id_e (from \c 0 to \c HWLOC_MEMATTR_ID_MAX-1).
 * Then custom attributes may be dynamically registered with
 * hwloc_memattr_register(). They will get the following IDs
 * (\c HWLOC_MEMATTR_ID_MAX for the first one, etc.).
 *
 * To iterate over all valid attributes
 * (either predefined or dynamically registered custom ones),
 * one may iterate over IDs starting from \c 0 until hwloc_memattr_get_name()
 * or hwloc_memattr_get_flags() returns an error.
 *
 * The values for an existing attribute or for custom dynamically registered ones
 * may be set or modified with hwloc_memattr_set_value().
 *
 * @{
 */

/** \brief Return the name of a memory attribute.
 *
 * The output pointer \p name cannot be \c NULL.
 *
 * \return 0 on success.
 * \return -1 with errno set to \c EINVAL if the attribute does not exist.
 */
HWLOC_DECLSPEC int
hwloc_memattr_get_name(hwloc_topology_t topology,
                       hwloc_memattr_id_t attribute,
                       const char **name);

/** \brief Return the flags of the given attribute.
 *
 * Flags are a OR'ed set of ::hwloc_memattr_flag_e.
 *
 * The output pointer \p flags cannot be \c NULL.
 *
 * \return 0 on success.
 * \return -1 with errno set to \c EINVAL if the attribute does not exist.
 */
HWLOC_DECLSPEC int
hwloc_memattr_get_flags(hwloc_topology_t topology,
                        hwloc_memattr_id_t attribute,
                        unsigned long *flags);

/** \brief Memory attribute flags.
 * Given to hwloc_memattr_register() and returned by hwloc_memattr_get_flags().
 */
enum hwloc_memattr_flag_e {
  /** \brief The best nodes for this memory attribute are those with the higher values.
   * For instance Bandwidth.
   */
  HWLOC_MEMATTR_FLAG_HIGHER_FIRST = (1UL<<0),
  /** \brief The best nodes for this memory attribute are those with the lower values.
   * For instance Latency.
   */
  HWLOC_MEMATTR_FLAG_LOWER_FIRST = (1UL<<1),
  /** \brief The value returned for this memory attribute depends on the given initiator.
   * For instance Bandwidth and Latency, but not Capacity.
   */
  HWLOC_MEMATTR_FLAG_NEED_INITIATOR = (1UL<<2)
};

/** \brief Register a new memory attribute.
 *
 * Add a new custom memory attribute.
 * Flags are a OR'ed set of ::hwloc_memattr_flag_e. It must contain one of
 * ::HWLOC_MEMATTR_FLAG_HIGHER_FIRST or ::HWLOC_MEMATTR_FLAG_LOWER_FIRST but not both.
 *
 * The new attribute \p id is immediately after the last existing attribute ID
 * (which is either the ID of the last registered attribute if any,
 * or the ID of the last predefined attribute in ::hwloc_memattr_id_e).
 *
 * \return 0 on success.
 * \return -1 with errno set to \c EINVAL if an invalid set of flags is given.
 * \return -1 with errno set to \c EBUSY if another attribute already uses this name.
 */
HWLOC_DECLSPEC int
hwloc_memattr_register(hwloc_topology_t topology,
                       const char *name,
                       unsigned long flags,
                       hwloc_memattr_id_t *id);

/** \brief Set an attribute value for a specific target NUMA node.
 *
 * If the attribute does not relate to a specific initiator
 * (it does not have the flag ::HWLOC_MEMATTR_FLAG_NEED_INITIATOR),
 * location \p initiator is ignored and may be \c NULL.
 *
 * The initiator will be copied into the topology,
 * the caller should free anything allocated to store the initiator,
 * for instance the cpuset.
 *
 * \p target_node cannot be \c NULL.
 *
 * \p attribute cannot be ::HWLOC_MEMATTR_FLAG_ID_CAPACITY or
 * ::HWLOC_MEMATTR_FLAG_ID_LOCALITY.
 *
 * \p flags must be \c 0 for now.
 *
 * \note The initiator \p initiator should be of type ::HWLOC_LOCATION_TYPE_CPUSET
 * when referring to accesses performed by CPU cores.
 * ::HWLOC_LOCATION_TYPE_OBJECT is currently unused internally by hwloc,
 * but users may for instance use it to provide custom information about
 * host memory accesses performed by GPUs.
 *
 * \return 0 on success or -1 on error.
 */
HWLOC_DECLSPEC int
hwloc_memattr_set_value(hwloc_topology_t topology,
                        hwloc_memattr_id_t attribute,
                        hwloc_obj_t target_node,
                        struct hwloc_location *initiator,
                        unsigned long flags,
                        hwloc_uint64_t value);

/** @} */

#ifdef __cplusplus
} /* extern "C" */
#endif


#endif /* HWLOC_MEMATTR_H */
