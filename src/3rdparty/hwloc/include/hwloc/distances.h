/*
 * Copyright © 2010-2020 Inria.  All rights reserved.
 * See COPYING in top-level directory.
 */

/** \file
 * \brief Object distances.
 */

#ifndef HWLOC_DISTANCES_H
#define HWLOC_DISTANCES_H

#ifndef HWLOC_H
#error Please include the main hwloc.h instead
#endif


#ifdef __cplusplus
extern "C" {
#elif 0
}
#endif


/** \defgroup hwlocality_distances_get Retrieve distances between objects
 * @{
 */

/** \brief Matrix of distances between a set of objects.
 *
 * This matrix often contains latencies between NUMA nodes
 * (as reported in the System Locality Distance Information Table (SLIT)
 * in the ACPI specification), which may or may not be physically accurate.
 * It corresponds to the latency for accessing the memory of one node
 * from a core in another node.
 * The corresponding kind is ::HWLOC_DISTANCES_KIND_FROM_OS | ::HWLOC_DISTANCES_KIND_FROM_USER.
 * The name of this distances structure is "NUMALatency".
 *
 * The matrix may also contain bandwidths between random sets of objects,
 * possibly provided by the user, as specified in the \p kind attribute.
 */
struct hwloc_distances_s {
  unsigned nbobjs;		/**< \brief Number of objects described by the distance matrix. */
  hwloc_obj_t *objs;		/**< \brief Array of objects described by the distance matrix.
				 * These objects are not in any particular order,
				 * see hwloc_distances_obj_index() and hwloc_distances_obj_pair_values()
				 * for easy ways to find objects in this array and their corresponding values.
				 */
  unsigned long kind;		/**< \brief OR'ed set of ::hwloc_distances_kind_e. */
  hwloc_uint64_t *values;	/**< \brief Matrix of distances between objects, stored as a one-dimension array.
				 *
				 * Distance from i-th to j-th object is stored in slot i*nbobjs+j.
				 * The meaning of the value depends on the \p kind attribute.
				 */
};

/** \brief Kinds of distance matrices.
 *
 * The \p kind attribute of struct hwloc_distances_s is a OR'ed set
 * of kinds.
 *
 * A kind of format HWLOC_DISTANCES_KIND_FROM_* specifies where the
 * distance information comes from, if known.
 *
 * A kind of format HWLOC_DISTANCES_KIND_MEANS_* specifies whether
 * values are latencies or bandwidths, if applicable.
 */
enum hwloc_distances_kind_e {
  /** \brief These distances were obtained from the operating system or hardware.
   * \hideinitializer
   */
  HWLOC_DISTANCES_KIND_FROM_OS = (1UL<<0),
  /** \brief These distances were provided by the user.
   * \hideinitializer
   */
  HWLOC_DISTANCES_KIND_FROM_USER = (1UL<<1),

  /** \brief Distance values are similar to latencies between objects.
   * Values are smaller for closer objects, hence minimal on the diagonal
   * of the matrix (distance between an object and itself).
   * It could also be the number of network hops between objects, etc.
   * \hideinitializer
   */
  HWLOC_DISTANCES_KIND_MEANS_LATENCY = (1UL<<2),
  /** \brief Distance values are similar to bandwidths between objects.
   * Values are higher for closer objects, hence maximal on the diagonal
   * of the matrix (distance between an object and itself).
   * Such values are currently ignored for distance-based grouping.
   * \hideinitializer
   */
  HWLOC_DISTANCES_KIND_MEANS_BANDWIDTH = (1UL<<3),

  /** \brief This distances structure covers objects of different types.
   * \hideinitializer
   */
  HWLOC_DISTANCES_KIND_HETEROGENEOUS_TYPES = (1UL<<4)
};

/** \brief Retrieve distance matrices.
 *
 * Retrieve distance matrices from the topology into the \p distances array.
 *
 * \p flags is currently unused, should be \c 0.
 *
 * \p kind serves as a filter. If \c 0, all distance matrices are returned.
 * If it contains some HWLOC_DISTANCES_KIND_FROM_*, only distance matrices
 * whose kind matches one of these are returned.
 * If it contains some HWLOC_DISTANCES_KIND_MEANS_*, only distance matrices
 * whose kind matches one of these are returned.
 *
 * On input, \p nr points to the number of distance matrices that may be stored
 * in \p distances.
 * On output, \p nr points to the number of distance matrices that were actually
 * found, even if some of them couldn't be stored in \p distances.
 * Distance matrices that couldn't be stored are ignored, but the function still
 * returns success (\c 0). The caller may find out by comparing the value pointed
 * by \p nr before and after the function call.
 *
 * Each distance matrix returned in the \p distances array should be released
 * by the caller using hwloc_distances_release().
 */
HWLOC_DECLSPEC int
hwloc_distances_get(hwloc_topology_t topology,
		    unsigned *nr, struct hwloc_distances_s **distances,
		    unsigned long kind, unsigned long flags);

/** \brief Retrieve distance matrices for object at a specific depth in the topology.
 *
 * Identical to hwloc_distances_get() with the additional \p depth filter.
 */
HWLOC_DECLSPEC int
hwloc_distances_get_by_depth(hwloc_topology_t topology, int depth,
			     unsigned *nr, struct hwloc_distances_s **distances,
			     unsigned long kind, unsigned long flags);

/** \brief Retrieve distance matrices for object of a specific type.
 *
 * Identical to hwloc_distances_get() with the additional \p type filter.
 */
HWLOC_DECLSPEC int
hwloc_distances_get_by_type(hwloc_topology_t topology, hwloc_obj_type_t type,
			    unsigned *nr, struct hwloc_distances_s **distances,
			    unsigned long kind, unsigned long flags);

/** \brief Retrieve a distance matrix with the given name.
 *
 * Usually only one distances structure may match a given name.
 *
 * The name of the most common structure is "NUMALatency".
 */
HWLOC_DECLSPEC int
hwloc_distances_get_by_name(hwloc_topology_t topology, const char *name,
			    unsigned *nr, struct hwloc_distances_s **distances,
			    unsigned long flags);

/** \brief Get a description of what a distances structure contains.
 *
 * For instance "NUMALatency" for hardware-provided NUMA distances (ACPI SLIT),
 * or NULL if unknown.
 */
HWLOC_DECLSPEC const char *
hwloc_distances_get_name(hwloc_topology_t topology, struct hwloc_distances_s *distances);

/** \brief Release a distance matrix structure previously returned by hwloc_distances_get().
 *
 * \note This function is not required if the structure is removed with hwloc_distances_release_remove().
 */
HWLOC_DECLSPEC void
hwloc_distances_release(hwloc_topology_t topology, struct hwloc_distances_s *distances);

/** @} */



/** \defgroup hwlocality_distances_consult Helpers for consulting distance matrices
 * @{
 */

/** \brief Find the index of an object in a distances structure.
 *
 * \return -1 if object \p obj is not involved in structure \p distances.
 */
static __hwloc_inline int
hwloc_distances_obj_index(struct hwloc_distances_s *distances, hwloc_obj_t obj)
{
  unsigned i;
  for(i=0; i<distances->nbobjs; i++)
    if (distances->objs[i] == obj)
      return (int)i;
  return -1;
}

/** \brief Find the values between two objects in a distance matrices.
 *
 * The distance from \p obj1 to \p obj2 is stored in the value pointed by
 * \p value1to2 and reciprocally.
 *
 * \return -1 if object \p obj1 or \p obj2 is not involved in structure \p distances.
 */
static __hwloc_inline int
hwloc_distances_obj_pair_values(struct hwloc_distances_s *distances,
				hwloc_obj_t obj1, hwloc_obj_t obj2,
				hwloc_uint64_t *value1to2, hwloc_uint64_t *value2to1)
{
  int i1 = hwloc_distances_obj_index(distances, obj1);
  int i2 = hwloc_distances_obj_index(distances, obj2);
  if (i1 < 0 || i2 < 0)
    return -1;
  *value1to2 = distances->values[i1 * distances->nbobjs + i2];
  *value2to1 = distances->values[i2 * distances->nbobjs + i1];
  return 0;
}

/** @} */



/** \defgroup hwlocality_distances_add Add or remove distances between objects
 * @{
 */

/** \brief Flags for adding a new distances to a topology. */
enum hwloc_distances_add_flag_e {
  /** \brief Try to group objects based on the newly provided distance information.
   * \hideinitializer
   */
  HWLOC_DISTANCES_ADD_FLAG_GROUP = (1UL<<0),
  /** \brief If grouping, consider the distance values as inaccurate and relax the
   * comparisons during the grouping algorithms. The actual accuracy may be modified
   * through the HWLOC_GROUPING_ACCURACY environment variable (see \ref envvar).
   * \hideinitializer
   */
  HWLOC_DISTANCES_ADD_FLAG_GROUP_INACCURATE = (1UL<<1)
};

/** \brief Provide a new distance matrix.
 *
 * Provide the matrix of distances between a set of objects given by \p nbobjs
 * and the \p objs array. \p nbobjs must be at least 2.
 * The distances are stored as a one-dimension array in \p values.
 * The distance from object i to object j is in slot i*nbobjs+j.
 *
 * \p kind specifies the kind of distance as a OR'ed set of ::hwloc_distances_kind_e.
 * Kind ::HWLOC_DISTANCES_KIND_HETEROGENEOUS_TYPES will be automatically added
 * if objects of different types are given.
 *
 * \p flags configures the behavior of the function using an optional OR'ed set of
 * ::hwloc_distances_add_flag_e.
 */
HWLOC_DECLSPEC int hwloc_distances_add(hwloc_topology_t topology,
				       unsigned nbobjs, hwloc_obj_t *objs, hwloc_uint64_t *values,
				       unsigned long kind, unsigned long flags);

/** \brief Remove all distance matrices from a topology.
 *
 * Remove all distance matrices, either provided by the user or
 * gathered through the OS.
 *
 * If these distances were used to group objects, these additional
 * Group objects are not removed from the topology.
 */
HWLOC_DECLSPEC int hwloc_distances_remove(hwloc_topology_t topology);

/** \brief Remove distance matrices for objects at a specific depth in the topology.
 *
 * Identical to hwloc_distances_remove() but only applies to one level of the topology.
 */
HWLOC_DECLSPEC int hwloc_distances_remove_by_depth(hwloc_topology_t topology, int depth);

/** \brief Remove distance matrices for objects of a specific type in the topology.
 *
 * Identical to hwloc_distances_remove() but only applies to one level of the topology.
 */
static __hwloc_inline int
hwloc_distances_remove_by_type(hwloc_topology_t topology, hwloc_obj_type_t type)
{
  int depth = hwloc_get_type_depth(topology, type);
  if (depth == HWLOC_TYPE_DEPTH_UNKNOWN || depth == HWLOC_TYPE_DEPTH_MULTIPLE)
    return 0;
  return hwloc_distances_remove_by_depth(topology, depth);
}

/** \brief Release and remove the given distance matrice from the topology.
 *
 * This function includes a call to hwloc_distances_release().
 */
HWLOC_DECLSPEC int hwloc_distances_release_remove(hwloc_topology_t topology, struct hwloc_distances_s *distances);

/** @} */


#ifdef __cplusplus
} /* extern "C" */
#endif


#endif /* HWLOC_DISTANCES_H */
