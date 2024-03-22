/*
 * Copyright © 2009 CNRS
 * Copyright © 2009-2023 Inria.  All rights reserved.
 * Copyright © 2009-2012 Université Bordeaux
 * Copyright © 2009-2011 Cisco Systems, Inc.  All rights reserved.
 * See COPYING in top-level directory.
 */

/** \file
 * \brief The bitmap API, for use in hwloc itself.
 */

#ifndef HWLOC_BITMAP_H
#define HWLOC_BITMAP_H

#include "hwloc/autogen/config.h"

#include <assert.h>


#ifdef __cplusplus
extern "C" {
#endif


/** \defgroup hwlocality_bitmap The bitmap API
 *
 * The ::hwloc_bitmap_t type represents a set of integers (positive or null).
 * A bitmap may be of infinite size (all bits are set after some point).
 * A bitmap may even be full if all bits are set.
 *
 * Bitmaps are used by hwloc for sets of OS processors
 * (which may actually be hardware threads) as by ::hwloc_cpuset_t
 * (a typedef for ::hwloc_bitmap_t), or sets of NUMA memory nodes
 * as ::hwloc_nodeset_t (also a typedef for ::hwloc_bitmap_t).
 * Those are used for cpuset and nodeset fields in the ::hwloc_obj structure,
 * see \ref hwlocality_object_sets.
 *
 * <em>Both CPU and node sets are always indexed by OS physical number.</em>
 * However users should usually not build CPU and node sets manually
 * (e.g. with hwloc_bitmap_set()).
 * One should rather use existing object sets and combine them with
 * hwloc_bitmap_or(), etc.
 * For instance, binding the current thread on a pair of cores may be performed with:
 * \code
 * hwloc_obj_t core1 = ... , core2 = ... ;
 * hwloc_bitmap_t set = hwloc_bitmap_alloc();
 * hwloc_bitmap_or(set, core1->cpuset, core2->cpuset);
 * hwloc_set_cpubind(topology, set, HWLOC_CPUBIND_THREAD);
 * hwloc_bitmap_free(set);
 * \endcode
 *
 * \note Most functions below return 0 on success and -1 on error.
 * The usual error case would be an internal failure to realloc/extend
 * the storage of the bitmap (\p errno would be set to \c ENOMEM).
 * See also \ref hwlocality_api_error_reporting.
 *
 * \note Several examples of using the bitmap API are available under the
 * doc/examples/ directory in the source tree.
 * Regression tests such as tests/hwloc/hwloc_bitmap*.c also make intensive use
 * of this API.
 * @{
 */


/** \brief
 * Set of bits represented as an opaque pointer to an internal bitmap.
 */
typedef struct hwloc_bitmap_s * hwloc_bitmap_t;
/** \brief a non-modifiable ::hwloc_bitmap_t */
typedef const struct hwloc_bitmap_s * hwloc_const_bitmap_t;


/*
 * Bitmap allocation, freeing and copying.
 */

/** \brief Allocate a new empty bitmap.
 *
 * \returns A valid bitmap or \c NULL.
 *
 * The bitmap should be freed by a corresponding call to
 * hwloc_bitmap_free().
 */
HWLOC_DECLSPEC hwloc_bitmap_t hwloc_bitmap_alloc(void) __hwloc_attribute_malloc;

/** \brief Allocate a new full bitmap.
 *
 * \returns A valid bitmap or \c NULL.
 *
 * The bitmap should be freed by a corresponding call to
 * hwloc_bitmap_free().
 */
HWLOC_DECLSPEC hwloc_bitmap_t hwloc_bitmap_alloc_full(void) __hwloc_attribute_malloc;

/** \brief Free bitmap \p bitmap.
 *
 * If \p bitmap is \c NULL, no operation is performed.
 */
HWLOC_DECLSPEC void hwloc_bitmap_free(hwloc_bitmap_t bitmap);

/** \brief Duplicate bitmap \p bitmap by allocating a new bitmap and copying \p bitmap contents.
 *
 * If \p bitmap is \c NULL, \c NULL is returned.
 */
HWLOC_DECLSPEC hwloc_bitmap_t hwloc_bitmap_dup(hwloc_const_bitmap_t bitmap) __hwloc_attribute_malloc;

/** \brief Copy the contents of bitmap \p src into the already allocated bitmap \p dst */
HWLOC_DECLSPEC int hwloc_bitmap_copy(hwloc_bitmap_t dst, hwloc_const_bitmap_t src);


/*
 * Bitmap/String Conversion
 */

/** \brief Stringify a bitmap.
 *
 * Up to \p buflen characters may be written in buffer \p buf.
 *
 * If \p buflen is 0, \p buf may safely be \c NULL.
 *
 * \return the number of characters that were actually written if not truncating,
 * or that would have been written (not including the ending \\0).
 */
HWLOC_DECLSPEC int hwloc_bitmap_snprintf(char * __hwloc_restrict buf, size_t buflen, hwloc_const_bitmap_t bitmap);

/** \brief Stringify a bitmap into a newly allocated string.
 *
 * \return 0 on success, -1 on error.
 */
HWLOC_DECLSPEC int hwloc_bitmap_asprintf(char ** strp, hwloc_const_bitmap_t bitmap);

/** \brief Parse a bitmap string and stores it in bitmap \p bitmap.
 *
 * \return 0 on success, -1 on error.
 */
HWLOC_DECLSPEC int hwloc_bitmap_sscanf(hwloc_bitmap_t bitmap, const char * __hwloc_restrict string);

/** \brief Stringify a bitmap in the list format.
 *
 * Lists are comma-separated indexes or ranges.
 * Ranges are dash separated indexes.
 * The last range may not have an ending indexes if the bitmap is infinitely set.
 *
 * Up to \p buflen characters may be written in buffer \p buf.
 *
 * If \p buflen is 0, \p buf may safely be \c NULL.
 *
 * \return the number of characters that were actually written if not truncating,
 * or that would have been written (not including the ending \\0).
 */
HWLOC_DECLSPEC int hwloc_bitmap_list_snprintf(char * __hwloc_restrict buf, size_t buflen, hwloc_const_bitmap_t bitmap);

/** \brief Stringify a bitmap into a newly allocated list string.
 *
 * \return 0 on success, -1 on error.
 */
HWLOC_DECLSPEC int hwloc_bitmap_list_asprintf(char ** strp, hwloc_const_bitmap_t bitmap);

/** \brief Parse a list string and stores it in bitmap \p bitmap.
 *
 * \return 0 on success, -1 on error.
 */
HWLOC_DECLSPEC int hwloc_bitmap_list_sscanf(hwloc_bitmap_t bitmap, const char * __hwloc_restrict string);

/** \brief Stringify a bitmap in the taskset-specific format.
 *
 * The taskset command manipulates bitmap strings that contain a single
 * (possible very long) hexadecimal number starting with 0x.
 *
 * Up to \p buflen characters may be written in buffer \p buf.
 *
 * If \p buflen is 0, \p buf may safely be \c NULL.
 *
 * \return the number of characters that were actually written if not truncating,
 * or that would have been written (not including the ending \\0).
 */
HWLOC_DECLSPEC int hwloc_bitmap_taskset_snprintf(char * __hwloc_restrict buf, size_t buflen, hwloc_const_bitmap_t bitmap);

/** \brief Stringify a bitmap into a newly allocated taskset-specific string.
 *
 * \return 0 on success, -1 on error.
 */
HWLOC_DECLSPEC int hwloc_bitmap_taskset_asprintf(char ** strp, hwloc_const_bitmap_t bitmap);

/** \brief Parse a taskset-specific bitmap string and stores it in bitmap \p bitmap.
 *
 * \return 0 on success, -1 on error.
 */
HWLOC_DECLSPEC int hwloc_bitmap_taskset_sscanf(hwloc_bitmap_t bitmap, const char * __hwloc_restrict string);


/*
 * Building bitmaps.
 */

/** \brief Empty the bitmap \p bitmap */
HWLOC_DECLSPEC void hwloc_bitmap_zero(hwloc_bitmap_t bitmap);

/** \brief Fill bitmap \p bitmap with all possible indexes (even if those objects don't exist or are otherwise unavailable) */
HWLOC_DECLSPEC void hwloc_bitmap_fill(hwloc_bitmap_t bitmap);

/** \brief Empty the bitmap \p bitmap and add bit \p id */
HWLOC_DECLSPEC int hwloc_bitmap_only(hwloc_bitmap_t bitmap, unsigned id);

/** \brief Fill the bitmap \p and clear the index \p id */
HWLOC_DECLSPEC int hwloc_bitmap_allbut(hwloc_bitmap_t bitmap, unsigned id);

/** \brief Setup bitmap \p bitmap from unsigned long \p mask */
HWLOC_DECLSPEC int hwloc_bitmap_from_ulong(hwloc_bitmap_t bitmap, unsigned long mask);

/** \brief Setup bitmap \p bitmap from unsigned long \p mask used as \p i -th subset */
HWLOC_DECLSPEC int hwloc_bitmap_from_ith_ulong(hwloc_bitmap_t bitmap, unsigned i, unsigned long mask);

/** \brief Setup bitmap \p bitmap from unsigned longs \p masks used as first \p nr subsets */
HWLOC_DECLSPEC int hwloc_bitmap_from_ulongs(hwloc_bitmap_t bitmap, unsigned nr, const unsigned long *masks);


/*
 * Modifying bitmaps.
 */

/** \brief Add index \p id in bitmap \p bitmap */
HWLOC_DECLSPEC int hwloc_bitmap_set(hwloc_bitmap_t bitmap, unsigned id);

/** \brief Add indexes from \p begin to \p end in bitmap \p bitmap.
 *
 * If \p end is \c -1, the range is infinite.
 */
HWLOC_DECLSPEC int hwloc_bitmap_set_range(hwloc_bitmap_t bitmap, unsigned begin, int end);

/** \brief Replace \p i -th subset of bitmap \p bitmap with unsigned long \p mask */
HWLOC_DECLSPEC int hwloc_bitmap_set_ith_ulong(hwloc_bitmap_t bitmap, unsigned i, unsigned long mask);

/** \brief Remove index \p id from bitmap \p bitmap */
HWLOC_DECLSPEC int hwloc_bitmap_clr(hwloc_bitmap_t bitmap, unsigned id);

/** \brief Remove indexes from \p begin to \p end in bitmap \p bitmap.
 *
 * If \p end is \c -1, the range is infinite.
 */
HWLOC_DECLSPEC int hwloc_bitmap_clr_range(hwloc_bitmap_t bitmap, unsigned begin, int end);

/** \brief Keep a single index among those set in bitmap \p bitmap
 *
 * May be useful before binding so that the process does not
 * have a chance of migrating between multiple processors
 * in the original mask.
 * Instead of running the task on any PU inside the given CPU set,
 * the operating system scheduler will be forced to run it on a single
 * of these PUs.
 * It avoids a migration overhead and cache-line ping-pongs between PUs.
 *
 * \note This function is NOT meant to distribute multiple processes
 * within a single CPU set. It always return the same single bit when
 * called multiple times on the same input set. hwloc_distrib() may
 * be used for generating CPU sets to distribute multiple tasks below
 * a single multi-PU object.
 *
 * \note This function cannot be applied to an object set directly. It
 * should be applied to a copy (which may be obtained with hwloc_bitmap_dup()).
 */
HWLOC_DECLSPEC int hwloc_bitmap_singlify(hwloc_bitmap_t bitmap);


/*
 * Consulting bitmaps.
 */

/** \brief Convert the beginning part of bitmap \p bitmap into unsigned long \p mask */
HWLOC_DECLSPEC unsigned long hwloc_bitmap_to_ulong(hwloc_const_bitmap_t bitmap) __hwloc_attribute_pure;

/** \brief Convert the \p i -th subset of bitmap \p bitmap into unsigned long mask */
HWLOC_DECLSPEC unsigned long hwloc_bitmap_to_ith_ulong(hwloc_const_bitmap_t bitmap, unsigned i) __hwloc_attribute_pure;

/** \brief Convert the first \p nr subsets of bitmap \p bitmap into the array of \p nr unsigned long \p masks
 *
 * \p nr may be determined earlier with hwloc_bitmap_nr_ulongs().
 *
 * \return 0
 */
HWLOC_DECLSPEC int hwloc_bitmap_to_ulongs(hwloc_const_bitmap_t bitmap, unsigned nr, unsigned long *masks);

/** \brief Return the number of unsigned longs required for storing bitmap \p bitmap entirely
 *
 * This is the number of contiguous unsigned longs from the very first bit of the bitmap
 * (even if unset) up to the last set bit.
 * This is useful for knowing the \p nr parameter to pass to hwloc_bitmap_to_ulongs()
 * (or which calls to hwloc_bitmap_to_ith_ulong() are needed)
 * to entirely convert a bitmap into multiple unsigned longs.
 *
 * When called on the output of hwloc_topology_get_topology_cpuset(),
 * the returned number is large enough for all cpusets of the topology.
 *
 * \return the number of unsigned longs required.
 * \return -1 if \p bitmap is infinite.
 */
HWLOC_DECLSPEC int hwloc_bitmap_nr_ulongs(hwloc_const_bitmap_t bitmap) __hwloc_attribute_pure;

/** \brief Test whether index \p id is part of bitmap \p bitmap.
 *
 * \return 1 if the bit at index \p id is set in bitmap \p bitmap, 0 otherwise.
 */
HWLOC_DECLSPEC int hwloc_bitmap_isset(hwloc_const_bitmap_t bitmap, unsigned id) __hwloc_attribute_pure;

/** \brief Test whether bitmap \p bitmap is empty
 *
 * \return 1 if bitmap is empty, 0 otherwise.
 */
HWLOC_DECLSPEC int hwloc_bitmap_iszero(hwloc_const_bitmap_t bitmap) __hwloc_attribute_pure;

/** \brief Test whether bitmap \p bitmap is completely full
 *
 * \return 1 if bitmap is full, 0 otherwise.
 *
 * \note A full bitmap is always infinitely set.
 */
HWLOC_DECLSPEC int hwloc_bitmap_isfull(hwloc_const_bitmap_t bitmap) __hwloc_attribute_pure;

/** \brief Compute the first index (least significant bit) in bitmap \p bitmap
 *
 * \return the first index set in \p bitmap.
 * \return -1 if \p bitmap is empty.
 */
HWLOC_DECLSPEC int hwloc_bitmap_first(hwloc_const_bitmap_t bitmap) __hwloc_attribute_pure;

/** \brief Compute the next index in bitmap \p bitmap which is after index \p prev
 *
 * \return the first index set in \p bitmap if \p prev is \c -1.
 * \return the next index set in \p bitmap if \p prev is not \c -1.
 * \return -1 if no index with higher index is set in \p bitmap.
 */
HWLOC_DECLSPEC int hwloc_bitmap_next(hwloc_const_bitmap_t bitmap, int prev) __hwloc_attribute_pure;

/** \brief Compute the last index (most significant bit) in bitmap \p bitmap
 *
 * \return the last index set in \p bitmap.
 * \return -1 if \p bitmap is empty, or if \p bitmap is infinitely set.
 */
HWLOC_DECLSPEC int hwloc_bitmap_last(hwloc_const_bitmap_t bitmap) __hwloc_attribute_pure;

/** \brief Compute the "weight" of bitmap \p bitmap (i.e., number of
 * indexes that are in the bitmap).
 *
 * \return the number of indexes that are in the bitmap.
 * \return -1 if \p bitmap is infinitely set.
 */
HWLOC_DECLSPEC int hwloc_bitmap_weight(hwloc_const_bitmap_t bitmap) __hwloc_attribute_pure;

/** \brief Compute the first unset index (least significant bit) in bitmap \p bitmap
 *
 * \return the first unset index in \p bitmap.
 * \return -1 if \p bitmap is full.
 */
HWLOC_DECLSPEC int hwloc_bitmap_first_unset(hwloc_const_bitmap_t bitmap) __hwloc_attribute_pure;

/** \brief Compute the next unset index in bitmap \p bitmap which is after index \p prev
 *
 * \return the first index unset in \p bitmap if \p prev is \c -1.
 * \return the next index unset in \p bitmap if \p prev is not \c -1.
 * \return -1 if no index with higher index is unset in \p bitmap.
 */
HWLOC_DECLSPEC int hwloc_bitmap_next_unset(hwloc_const_bitmap_t bitmap, int prev) __hwloc_attribute_pure;

/** \brief Compute the last unset index (most significant bit) in bitmap \p bitmap
 *
 * \return the last index unset in \p bitmap.
 * \return -1 if \p bitmap is full, or if \p bitmap is not infinitely set.
 */
HWLOC_DECLSPEC int hwloc_bitmap_last_unset(hwloc_const_bitmap_t bitmap) __hwloc_attribute_pure;

/** \brief Loop macro iterating on bitmap \p bitmap
 *
 * The loop must start with hwloc_bitmap_foreach_begin() and end
 * with hwloc_bitmap_foreach_end() followed by a terminating ';'.
 *
 * \p id is the loop variable; it should be an unsigned int.  The
 * first iteration will set \p id to the lowest index in the bitmap.
 * Successive iterations will iterate through, in order, all remaining
 * indexes set in the bitmap.  To be specific: each iteration will return a
 * value for \p id such that hwloc_bitmap_isset(bitmap, id) is true.
 *
 * The assert prevents the loop from being infinite if the bitmap is infinitely set.
 *
 * \hideinitializer
 */
#define hwloc_bitmap_foreach_begin(id, bitmap) \
do { \
        assert(hwloc_bitmap_weight(bitmap) != -1); \
        for (id = hwloc_bitmap_first(bitmap); \
             (unsigned) id != (unsigned) -1; \
             id = hwloc_bitmap_next(bitmap, id)) {

/** \brief End of loop macro iterating on a bitmap.
 *
 * Needs a terminating ';'.
 *
 * \sa hwloc_bitmap_foreach_begin()
 * \hideinitializer
 */
#define hwloc_bitmap_foreach_end()		\
        } \
} while (0)


/*
 * Combining bitmaps.
 */

/** \brief Or bitmaps \p bitmap1 and \p bitmap2 and store the result in bitmap \p res
 *
 * \p res can be the same as \p bitmap1 or \p bitmap2
 */
HWLOC_DECLSPEC int hwloc_bitmap_or (hwloc_bitmap_t res, hwloc_const_bitmap_t bitmap1, hwloc_const_bitmap_t bitmap2);

/** \brief And bitmaps \p bitmap1 and \p bitmap2 and store the result in bitmap \p res
 *
 * \p res can be the same as \p bitmap1 or \p bitmap2
 */
HWLOC_DECLSPEC int hwloc_bitmap_and (hwloc_bitmap_t res, hwloc_const_bitmap_t bitmap1, hwloc_const_bitmap_t bitmap2);

/** \brief And bitmap \p bitmap1 and the negation of \p bitmap2 and store the result in bitmap \p res
 *
 * \p res can be the same as \p bitmap1 or \p bitmap2
 */
HWLOC_DECLSPEC int hwloc_bitmap_andnot (hwloc_bitmap_t res, hwloc_const_bitmap_t bitmap1, hwloc_const_bitmap_t bitmap2);

/** \brief Xor bitmaps \p bitmap1 and \p bitmap2 and store the result in bitmap \p res
 *
 * \p res can be the same as \p bitmap1 or \p bitmap2
 */
HWLOC_DECLSPEC int hwloc_bitmap_xor (hwloc_bitmap_t res, hwloc_const_bitmap_t bitmap1, hwloc_const_bitmap_t bitmap2);

/** \brief Negate bitmap \p bitmap and store the result in bitmap \p res
 *
 * \p res can be the same as \p bitmap
 */
HWLOC_DECLSPEC int hwloc_bitmap_not (hwloc_bitmap_t res, hwloc_const_bitmap_t bitmap);


/*
 * Comparing bitmaps.
 */

/** \brief Test whether bitmaps \p bitmap1 and \p bitmap2 intersects.
 *
 * \return 1 if bitmaps intersect, 0 otherwise.
 *
 * \note The empty bitmap does not intersect any other bitmap.
 */
HWLOC_DECLSPEC int hwloc_bitmap_intersects (hwloc_const_bitmap_t bitmap1, hwloc_const_bitmap_t bitmap2) __hwloc_attribute_pure;

/** \brief Test whether bitmap \p sub_bitmap is part of bitmap \p super_bitmap.
 *
 * \return 1 if \p sub_bitmap is included in \p super_bitmap, 0 otherwise.
 *
 * \note The empty bitmap is considered included in any other bitmap.
 */
HWLOC_DECLSPEC int hwloc_bitmap_isincluded (hwloc_const_bitmap_t sub_bitmap, hwloc_const_bitmap_t super_bitmap) __hwloc_attribute_pure;

/** \brief Test whether bitmap \p bitmap1 is equal to bitmap \p bitmap2.
 *
 * \return 1 if bitmaps are equal, 0 otherwise.
 */
HWLOC_DECLSPEC int hwloc_bitmap_isequal (hwloc_const_bitmap_t bitmap1, hwloc_const_bitmap_t bitmap2) __hwloc_attribute_pure;

/** \brief Compare bitmaps \p bitmap1 and \p bitmap2 using their lowest index.
 *
 * A bitmap is considered smaller if its least significant bit is smaller.
 * The empty bitmap is considered higher than anything (because its least significant bit does not exist).
 *
 * \return -1 if \p bitmap1 is considered smaller than \p bitmap2.
 * \return 1 if \p bitmap1 is considered larger than \p bitmap2.
 *
 * For instance comparing binary bitmaps 0011 and 0110 returns -1
 * (hence 0011 is considered smaller than 0110)
 * because least significant bit of 0011 (0001) is smaller than least significant bit of 0110 (0010).
 * Comparing 01001 and 00110 would also return -1 for the same reason.
 *
 * \return 0 if bitmaps are considered equal, even if they are not strictly equal.
 * They just need to have the same least significant bit.
 * For instance, comparing binary bitmaps 0010 and 0110 returns 0 because they have the same least significant bit.
 */
HWLOC_DECLSPEC int hwloc_bitmap_compare_first(hwloc_const_bitmap_t bitmap1, hwloc_const_bitmap_t bitmap2) __hwloc_attribute_pure;

/** \brief Compare bitmaps \p bitmap1 and \p bitmap2 in lexicographic order.
 *
 * Lexicographic comparison of bitmaps, starting for their highest indexes.
 * Compare last indexes first, then second, etc.
 * The empty bitmap is considered lower than anything.
 *
 * \return -1 if \p bitmap1 is considered smaller than \p bitmap2.
 * \return 1 if \p bitmap1 is considered larger than \p bitmap2.
 * \return 0 if bitmaps are equal (contrary to hwloc_bitmap_compare_first()).
 *
 * For instance comparing binary bitmaps 0011 and 0110 returns -1
 * (hence 0011 is considered smaller than 0110).
 * Comparing 00101 and 01010 returns -1 too.
 *
 * \note This is different from the non-existing hwloc_bitmap_compare_last()
 * which would only compare the highest index of each bitmap.
 */
HWLOC_DECLSPEC int hwloc_bitmap_compare(hwloc_const_bitmap_t bitmap1, hwloc_const_bitmap_t bitmap2) __hwloc_attribute_pure;

/** @} */


#ifdef __cplusplus
} /* extern "C" */
#endif


#endif /* HWLOC_BITMAP_H */
