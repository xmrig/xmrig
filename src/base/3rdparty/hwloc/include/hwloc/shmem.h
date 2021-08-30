/*
 * Copyright © 2013-2018 Inria.  All rights reserved.
 * See COPYING in top-level directory.
 */

/** \file
 * \brief Sharing topologies between processes
 */

#ifndef HWLOC_SHMEM_H
#define HWLOC_SHMEM_H

#include "hwloc.h"

#ifdef __cplusplus
extern "C" {
#elif 0
}
#endif


/** \defgroup hwlocality_shmem Sharing topologies between processes
 *
 * These functions are used to share a topology between processes by
 * duplicating it into a file-backed shared-memory buffer.
 *
 * The master process must first get the required shared-memory size
 * for storing this topology with hwloc_shmem_topology_get_length().
 *
 * Then it must find a virtual memory area of that size that is available
 * in all processes (identical virtual addresses in all processes).
 * On Linux, this can be done by comparing holes found in /proc/\<pid\>/maps
 * for each process.
 *
 * Once found, it must open a destination file for storing the buffer,
 * and pass it to hwloc_shmem_topology_write() together with
 * virtual memory address and length obtained above.
 *
 * Other processes may then adopt this shared topology by opening the
 * same file and passing it to hwloc_shmem_topology_adopt() with the
 * exact same virtual memory address and length.
 *
 * @{
 */

/** \brief Get the required shared memory length for storing a topology.
 *
 * This length (in bytes) must be used in hwloc_shmem_topology_write()
 * and hwloc_shmem_topology_adopt() later.
 *
 * \note Flags \p flags are currently unused, must be 0.
 */
HWLOC_DECLSPEC int hwloc_shmem_topology_get_length(hwloc_topology_t topology,
						   size_t *lengthp,
						   unsigned long flags);

/** \brief Duplicate a topology to a shared memory file.
 *
 * Temporarily map a file in virtual memory and duplicate the
 * topology \p topology by allocating duplicates in there.
 *
 * The segment of the file pointed by descriptor \p fd,
 * starting at offset \p fileoffset, and of length \p length (in bytes),
 * will be temporarily mapped at virtual address \p mmap_address
 * during the duplication.
 *
 * The mapping length \p length must have been previously obtained with
 * hwloc_shmem_topology_get_length()
 * and the topology must not have been modified in the meantime.
 *
 * \note Flags \p flags are currently unused, must be 0.
 *
 * \note The object userdata pointer is duplicated but the pointed buffer
 * is not. However the caller may also allocate it manually in shared memory
 * to share it as well.
 *
 * \return -1 with errno set to EBUSY if the virtual memory mapping defined
 * by \p mmap_address and \p length isn't available in the process.
 * \return -1 with errno set to EINVAL if \p fileoffset, \p mmap_address
 * or \p length aren't page-aligned.
 */
HWLOC_DECLSPEC int hwloc_shmem_topology_write(hwloc_topology_t topology,
					      int fd, hwloc_uint64_t fileoffset,
					      void *mmap_address, size_t length,
					      unsigned long flags);

/** \brief Adopt a shared memory topology stored in a file.
 *
 * Map a file in virtual memory and adopt the topology that was previously
 * stored there with hwloc_shmem_topology_write().
 *
 * The returned adopted topology in \p topologyp can be used just like any
 * topology. And it must be destroyed with hwloc_topology_destroy() as usual.
 *
 * However the topology is read-only.
 * For instance, it cannot be modified with hwloc_topology_restrict()
 * and object userdata pointers cannot be changed.
 *
 * The segment of the file pointed by descriptor \p fd,
 * starting at offset \p fileoffset, and of length \p length (in bytes),
 * will be mapped at virtual address \p mmap_address.
 *
 * The file pointed by descriptor \p fd, the offset \p fileoffset,
 * the requested mapping virtual address \p mmap_address and the length \p length
 * must be identical to what was given to hwloc_shmem_topology_write() earlier.
 *
 * \note Flags \p flags are currently unused, must be 0.
 *
 * \note The object userdata pointer should not be used unless the process
 * that created the shared topology also placed userdata-pointed buffers
 * in shared memory.
 *
 * \note This function takes care of calling hwloc_topology_abi_check().
 *
 * \return -1 with errno set to EBUSY if the virtual memory mapping defined
 * by \p mmap_address and \p length isn't available in the process.
 *
 * \return -1 with errno set to EINVAL if \p fileoffset, \p mmap_address
 * or \p length aren't page-aligned, or do not match what was given to
 * hwloc_shmem_topology_write() earlier.
 *
 * \return -1 with errno set to EINVAL if the layout of the topology structure
 * is different between the writer process and the adopter process.
 */
HWLOC_DECLSPEC int hwloc_shmem_topology_adopt(hwloc_topology_t *topologyp,
					      int fd, hwloc_uint64_t fileoffset,
					      void *mmap_address, size_t length,
					      unsigned long flags);
/** @} */


#ifdef __cplusplus
} /* extern "C" */
#endif


#endif /* HWLOC_SHMEM_H */
