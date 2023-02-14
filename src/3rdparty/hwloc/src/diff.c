/*
 * Copyright © 2013-2022 Inria.  All rights reserved.
 * See COPYING in top-level directory.
 */

#include "private/autogen/config.h"
#include "private/private.h"
#include "private/misc.h"

int hwloc_topology_diff_destroy(hwloc_topology_diff_t diff)
{
	hwloc_topology_diff_t next;
	while (diff) {
		next = diff->generic.next;
		switch (diff->generic.type) {
		default:
			break;
		case HWLOC_TOPOLOGY_DIFF_OBJ_ATTR:
			switch (diff->obj_attr.diff.generic.type) {
			default:
				break;
			case HWLOC_TOPOLOGY_DIFF_OBJ_ATTR_NAME:
			case HWLOC_TOPOLOGY_DIFF_OBJ_ATTR_INFO:
				free(diff->obj_attr.diff.string.name);
				free(diff->obj_attr.diff.string.oldvalue);
				free(diff->obj_attr.diff.string.newvalue);
				break;
			}
			break;
		}
		free(diff);
		diff = next;
	}
	return 0;
}

/************************
 * Computing diffs
 */

static void hwloc_append_diff(hwloc_topology_diff_t newdiff,
			      hwloc_topology_diff_t *firstdiffp,
			      hwloc_topology_diff_t *lastdiffp)
{
	if (*firstdiffp)
		(*lastdiffp)->generic.next = newdiff;
	else
		*firstdiffp = newdiff;
	*lastdiffp = newdiff;
	newdiff->generic.next = NULL;
}

static int hwloc_append_diff_too_complex(hwloc_obj_t obj1,
					 hwloc_topology_diff_t *firstdiffp,
					 hwloc_topology_diff_t *lastdiffp)
{
	hwloc_topology_diff_t newdiff;
	newdiff = malloc(sizeof(*newdiff));
	if (!newdiff)
		return -1;

	newdiff->too_complex.type = HWLOC_TOPOLOGY_DIFF_TOO_COMPLEX;
	newdiff->too_complex.obj_depth = obj1->depth;
	newdiff->too_complex.obj_index = obj1->logical_index;
	hwloc_append_diff(newdiff, firstdiffp, lastdiffp);
	return 0;
}

static int hwloc_append_diff_obj_attr_string(hwloc_obj_t obj,
					     hwloc_topology_diff_obj_attr_type_t type,
					     const char *name,
					     const char *oldvalue,
					     const char *newvalue,
					     hwloc_topology_diff_t *firstdiffp,
					     hwloc_topology_diff_t *lastdiffp)
{
	hwloc_topology_diff_t newdiff;
	newdiff = malloc(sizeof(*newdiff));
	if (!newdiff)
		return -1;

	newdiff->obj_attr.type = HWLOC_TOPOLOGY_DIFF_OBJ_ATTR;
	newdiff->obj_attr.obj_depth = obj->depth;
	newdiff->obj_attr.obj_index = obj->logical_index;
	newdiff->obj_attr.diff.string.type = type;
	newdiff->obj_attr.diff.string.name = name ? strdup(name) : NULL;
	newdiff->obj_attr.diff.string.oldvalue = oldvalue ? strdup(oldvalue) : NULL;
	newdiff->obj_attr.diff.string.newvalue = newvalue ? strdup(newvalue) : NULL;
	hwloc_append_diff(newdiff, firstdiffp, lastdiffp);
	return 0;
}

static int hwloc_append_diff_obj_attr_uint64(hwloc_obj_t obj,
					     hwloc_topology_diff_obj_attr_type_t type,
					     hwloc_uint64_t idx,
					     hwloc_uint64_t oldvalue,
					     hwloc_uint64_t newvalue,
					     hwloc_topology_diff_t *firstdiffp,
					     hwloc_topology_diff_t *lastdiffp)
{
	hwloc_topology_diff_t newdiff;
	newdiff = malloc(sizeof(*newdiff));
	if (!newdiff)
		return -1;

	newdiff->obj_attr.type = HWLOC_TOPOLOGY_DIFF_OBJ_ATTR;
	newdiff->obj_attr.obj_depth = obj->depth;
	newdiff->obj_attr.obj_index = obj->logical_index;
	newdiff->obj_attr.diff.uint64.type = type;
	newdiff->obj_attr.diff.uint64.index = idx;
	newdiff->obj_attr.diff.uint64.oldvalue = oldvalue;
	newdiff->obj_attr.diff.uint64.newvalue = newvalue;
	hwloc_append_diff(newdiff, firstdiffp, lastdiffp);
	return 0;
}

static int
hwloc_diff_trees(hwloc_topology_t topo1, hwloc_obj_t obj1,
		 hwloc_topology_t topo2, hwloc_obj_t obj2,
		 unsigned flags,
		 hwloc_topology_diff_t *firstdiffp, hwloc_topology_diff_t *lastdiffp)
{
	unsigned i;
	int err;
	hwloc_obj_t child1, child2;

	if (obj1->depth != obj2->depth)
		goto out_too_complex;

	if (obj1->type != obj2->type)
		goto out_too_complex;
	if ((!obj1->subtype) != (!obj2->subtype)
	    || (obj1->subtype && strcmp(obj1->subtype, obj2->subtype)))
		goto out_too_complex;

	if (obj1->os_index != obj2->os_index)
		/* we could allow different os_index for non-PU non-NUMAnode objects
		 * but it's likely useless anyway */
		goto out_too_complex;

#define _SETS_DIFFERENT(_set1, _set2) \
 (   ( !(_set1) != !(_set2) ) \
  || ( (_set1) && !hwloc_bitmap_isequal(_set1, _set2) ) )
#define SETS_DIFFERENT(_set, _obj1, _obj2) _SETS_DIFFERENT((_obj1)->_set, (_obj2)->_set)
	if (SETS_DIFFERENT(cpuset, obj1, obj2)
	    || SETS_DIFFERENT(complete_cpuset, obj1, obj2)
	    || SETS_DIFFERENT(nodeset, obj1, obj2)
	    || SETS_DIFFERENT(complete_nodeset, obj1, obj2))
		goto out_too_complex;

	/* no need to check logical_index, sibling_rank, symmetric_subtree,
	 * the parents did it */

	/* gp_index don't have to be strictly identical */

	if ((!obj1->name) != (!obj2->name)
	    || (obj1->name && strcmp(obj1->name, obj2->name))) {
		err = hwloc_append_diff_obj_attr_string(obj1,
						       HWLOC_TOPOLOGY_DIFF_OBJ_ATTR_NAME,
						       NULL,
						       obj1->name,
						       obj2->name,
						       firstdiffp, lastdiffp);
		if (err < 0)
			return err;
	}

	/* type-specific attrs */
	switch (obj1->type) {
	default:
		break;
	case HWLOC_OBJ_NUMANODE:
		if (obj1->attr->numanode.local_memory != obj2->attr->numanode.local_memory) {
			err = hwloc_append_diff_obj_attr_uint64(obj1,
								HWLOC_TOPOLOGY_DIFF_OBJ_ATTR_SIZE,
								0,
								obj1->attr->numanode.local_memory,
								obj2->attr->numanode.local_memory,
								firstdiffp, lastdiffp);
			if (err < 0)
				return err;
		}
		/* ignore memory page_types */
		break;
	case HWLOC_OBJ_L1CACHE:
	case HWLOC_OBJ_L2CACHE:
	case HWLOC_OBJ_L3CACHE:
	case HWLOC_OBJ_L4CACHE:
	case HWLOC_OBJ_L5CACHE:
	case HWLOC_OBJ_L1ICACHE:
	case HWLOC_OBJ_L2ICACHE:
	case HWLOC_OBJ_L3ICACHE:
		if (memcmp(obj1->attr, obj2->attr, sizeof(obj1->attr->cache)))
			goto out_too_complex;
		break;
	case HWLOC_OBJ_GROUP:
		if (memcmp(obj1->attr, obj2->attr, sizeof(obj1->attr->group)))
			goto out_too_complex;
		break;
	case HWLOC_OBJ_PCI_DEVICE:
		if (memcmp(obj1->attr, obj2->attr, sizeof(obj1->attr->pcidev)))
			goto out_too_complex;
		break;
	case HWLOC_OBJ_BRIDGE:
		if (memcmp(obj1->attr, obj2->attr, sizeof(obj1->attr->bridge)))
			goto out_too_complex;
		break;
	case HWLOC_OBJ_OS_DEVICE:
		if (memcmp(obj1->attr, obj2->attr, sizeof(obj1->attr->osdev)))
			goto out_too_complex;
		break;
	}

	/* infos */
	if (obj1->infos_count != obj2->infos_count)
		goto out_too_complex;
	for(i=0; i<obj1->infos_count; i++) {
		struct hwloc_info_s *info1 = &obj1->infos[i], *info2 = &obj2->infos[i];
		if (strcmp(info1->name, info2->name))
			goto out_too_complex;
		if (strcmp(info1->value, info2->value)) {
			err = hwloc_append_diff_obj_attr_string(obj1,
								HWLOC_TOPOLOGY_DIFF_OBJ_ATTR_INFO,
								info1->name,
								info1->value,
								info2->value,
								firstdiffp, lastdiffp);
			if (err < 0)
				return err;
		}
	}

	/* ignore userdata */

	/* children */
	for(child1 = obj1->first_child, child2 = obj2->first_child;
	    child1 != NULL && child2 != NULL;
	    child1 = child1->next_sibling, child2 = child2->next_sibling) {
		err = hwloc_diff_trees(topo1, child1,
				       topo2, child2,
				       flags,
				       firstdiffp, lastdiffp);
		if (err < 0)
			return err;
	}
	if (child1 || child2)
		goto out_too_complex;

	/* memory children */
	for(child1 = obj1->memory_first_child, child2 = obj2->memory_first_child;
	    child1 != NULL && child2 != NULL;
	    child1 = child1->next_sibling, child2 = child2->next_sibling) {
		err = hwloc_diff_trees(topo1, child1,
				       topo2, child2,
				       flags,
				       firstdiffp, lastdiffp);
		if (err < 0)
			return err;
	}
	if (child1 || child2)
		goto out_too_complex;

	/* I/O children */
	for(child1 = obj1->io_first_child, child2 = obj2->io_first_child;
	    child1 != NULL && child2 != NULL;
	    child1 = child1->next_sibling, child2 = child2->next_sibling) {
		err = hwloc_diff_trees(topo1, child1,
				       topo2, child2,
				       flags,
				       firstdiffp, lastdiffp);
		if (err < 0)
			return err;
	}
	if (child1 || child2)
		goto out_too_complex;

	/* misc children */
	for(child1 = obj1->misc_first_child, child2 = obj2->misc_first_child;
	    child1 != NULL && child2 != NULL;
	    child1 = child1->next_sibling, child2 = child2->next_sibling) {
		err = hwloc_diff_trees(topo1, child1,
				       topo2, child2,
				       flags,
				       firstdiffp, lastdiffp);
		if (err < 0)
			return err;
	}
	if (child1 || child2)
		goto out_too_complex;

	return 0;

out_too_complex:
	hwloc_append_diff_too_complex(obj1, firstdiffp, lastdiffp);
	return 0;
}

int hwloc_topology_diff_build(hwloc_topology_t topo1,
			      hwloc_topology_t topo2,
			      unsigned long flags,
			      hwloc_topology_diff_t *diffp)
{
	hwloc_topology_diff_t lastdiff, tmpdiff;
	struct hwloc_internal_distances_s *dist1, *dist2;
	unsigned i;
	int err;

	if (!topo1->is_loaded || !topo2->is_loaded) {
	  errno = EINVAL;
	  return -1;
	}

	if (flags != 0) {
		errno = EINVAL;
		return -1;
	}

	*diffp = NULL;
	err = hwloc_diff_trees(topo1, hwloc_get_root_obj(topo1),
			       topo2, hwloc_get_root_obj(topo2),
			       flags,
			       diffp, &lastdiff);
	if (!err) {
		tmpdiff = *diffp;
		while (tmpdiff) {
			if (tmpdiff->generic.type == HWLOC_TOPOLOGY_DIFF_TOO_COMPLEX) {
				err = 1;
				break;
			}
			tmpdiff = tmpdiff->generic.next;
		}
	}

	if (!err) {
		if (SETS_DIFFERENT(allowed_cpuset, topo1, topo2)
		    || SETS_DIFFERENT(allowed_nodeset, topo1, topo2))
                  goto roottoocomplex;
	}

	if (!err) {
		/* distances */
		hwloc_internal_distances_refresh(topo1);
		hwloc_internal_distances_refresh(topo2);
		dist1 = topo1->first_dist;
		dist2 = topo2->first_dist;
		while (dist1 || dist2) {
			if (!!dist1 != !!dist2)
                          goto roottoocomplex;
			if (dist1->unique_type != dist2->unique_type
			    || dist1->different_types || dist2->different_types /* too lazy to support this case */
			    || dist1->nbobjs != dist2->nbobjs
			    || dist1->kind != dist2->kind
			    || memcmp(dist1->values, dist2->values, dist1->nbobjs * dist1->nbobjs * sizeof(*dist1->values)))
                          goto roottoocomplex;
			for(i=0; i<dist1->nbobjs; i++)
				/* gp_index isn't enforced above. so compare logical_index instead, which is enforced. requires distances refresh() above */
				if (dist1->objs[i]->logical_index != dist2->objs[i]->logical_index)
                                  goto roottoocomplex;
			dist1 = dist1->next;
			dist2 = dist2->next;
		}
	}

        if (!err) {
          /* memattrs */
          hwloc_internal_memattrs_refresh(topo1);
          hwloc_internal_memattrs_refresh(topo2);
          if (topo1->nr_memattrs != topo2->nr_memattrs)
            goto roottoocomplex;
          for(i=0; i<topo1->nr_memattrs; i++) {
            struct hwloc_internal_memattr_s *imattr1 = &topo1->memattrs[i], *imattr2 = &topo2->memattrs[i];
            unsigned j;
           if (strcmp(imattr1->name, imattr2->name)
                || imattr1->flags != imattr2->flags
                || imattr1->nr_targets != imattr2->nr_targets)
              goto roottoocomplex;
            if (i == HWLOC_MEMATTR_ID_CAPACITY
                || i == HWLOC_MEMATTR_ID_LOCALITY)
              /* no need to check virtual attributes, there were refreshed from other topology attributes, checked above */
              continue;
            for(j=0; j<imattr1->nr_targets; j++) {
              struct hwloc_internal_memattr_target_s *imtg1 = &imattr1->targets[j], *imtg2 = &imattr2->targets[j];
              if (imtg1->type != imtg2->type)
                goto roottoocomplex;
              if (imtg1->obj->logical_index != imtg2->obj->logical_index)
                goto roottoocomplex;
              if (imattr1->flags & HWLOC_MEMATTR_FLAG_NEED_INITIATOR) {
                unsigned k;
                for(k=0; k<imtg1->nr_initiators; k++) {
                  struct hwloc_internal_memattr_initiator_s *imi1 = &imtg1->initiators[k], *imi2 = &imtg2->initiators[k];
                  if (imi1->value != imi2->value
                      || imi1->initiator.type != imi2->initiator.type)
                    goto roottoocomplex;
                  if (imi1->initiator.type == HWLOC_LOCATION_TYPE_CPUSET) {
                    if (!hwloc_bitmap_isequal(imi1->initiator.location.cpuset, imi2->initiator.location.cpuset))
                      goto roottoocomplex;
                  } else if (imi1->initiator.type == HWLOC_LOCATION_TYPE_OBJECT) {
                    if (imi1->initiator.location.object.type != imi2->initiator.location.object.type)
                      goto roottoocomplex;
                    if (imi1->initiator.location.object.obj->logical_index != imi2->initiator.location.object.obj->logical_index)
                      goto roottoocomplex;
                  } else {
                    assert(0);
                  }
                }
              } else {
                if (imtg1->noinitiator_value != imtg2->noinitiator_value)
                  goto roottoocomplex;
              }
            }
          }
        }

	return err;

 roottoocomplex:
  hwloc_append_diff_too_complex(hwloc_get_root_obj(topo1), diffp, &lastdiff);
  return 1;
}

/********************
 * Applying diffs
 */

static int
hwloc_apply_diff_one(hwloc_topology_t topology,
		     hwloc_topology_diff_t diff,
		     unsigned long flags)
{
	int reverse = !!(flags & HWLOC_TOPOLOGY_DIFF_APPLY_REVERSE);

	switch (diff->generic.type) {
	case HWLOC_TOPOLOGY_DIFF_OBJ_ATTR: {
		struct hwloc_topology_diff_obj_attr_s *obj_attr = &diff->obj_attr;
		hwloc_obj_t obj = hwloc_get_obj_by_depth(topology, obj_attr->obj_depth, obj_attr->obj_index);
		if (!obj)
			return -1;

		switch (obj_attr->diff.generic.type) {
		case HWLOC_TOPOLOGY_DIFF_OBJ_ATTR_SIZE: {
			hwloc_obj_t tmpobj;
			hwloc_uint64_t oldvalue = reverse ? obj_attr->diff.uint64.newvalue : obj_attr->diff.uint64.oldvalue;
			hwloc_uint64_t newvalue = reverse ? obj_attr->diff.uint64.oldvalue : obj_attr->diff.uint64.newvalue;
			hwloc_uint64_t valuediff = newvalue - oldvalue;
			if (obj->type != HWLOC_OBJ_NUMANODE)
				return -1;
			if (obj->attr->numanode.local_memory != oldvalue)
				return -1;
			obj->attr->numanode.local_memory = newvalue;
			tmpobj = obj;
			while (tmpobj) {
				tmpobj->total_memory += valuediff;
				tmpobj = tmpobj->parent;
			}
			break;
		}
		case HWLOC_TOPOLOGY_DIFF_OBJ_ATTR_NAME: {
			const char *oldvalue = reverse ? obj_attr->diff.string.newvalue : obj_attr->diff.string.oldvalue;
			const char *newvalue = reverse ? obj_attr->diff.string.oldvalue : obj_attr->diff.string.newvalue;
			if (!obj->name || strcmp(obj->name, oldvalue))
				return -1;
			free(obj->name);
			obj->name = strdup(newvalue);
			break;
		}
		case HWLOC_TOPOLOGY_DIFF_OBJ_ATTR_INFO: {
			const char *name = obj_attr->diff.string.name;
			const char *oldvalue = reverse ? obj_attr->diff.string.newvalue : obj_attr->diff.string.oldvalue;
			const char *newvalue = reverse ? obj_attr->diff.string.oldvalue : obj_attr->diff.string.newvalue;
			unsigned i;
			int found = 0;
			for(i=0; i<obj->infos_count; i++) {
				struct hwloc_info_s *info = &obj->infos[i];
				if (!strcmp(info->name, name)
				    && !strcmp(info->value, oldvalue)) {
					free(info->value);
					info->value = strdup(newvalue);
					found = 1;
					break;
				}
			}
			if (!found)
				return -1;
			break;
		}
		default:
			return -1;
		}

		break;
	}
	default:
		return -1;
	}

	return 0;
}

int hwloc_topology_diff_apply(hwloc_topology_t topology,
			      hwloc_topology_diff_t diff,
			      unsigned long flags)
{
	hwloc_topology_diff_t tmpdiff, tmpdiff2;
	int err, nr;

	if (!topology->is_loaded) {
	  errno = EINVAL;
	  return -1;
	}
	if (topology->adopted_shmem_addr) {
	  errno = EPERM;
	  return -1;
	}

	if (flags & ~HWLOC_TOPOLOGY_DIFF_APPLY_REVERSE) {
		errno = EINVAL;
		return -1;
	}

	tmpdiff = diff;
	nr = 0;
	while (tmpdiff) {
		nr++;
		err = hwloc_apply_diff_one(topology, tmpdiff, flags);
		if (err < 0)
			goto cancel;
		tmpdiff = tmpdiff->generic.next;
	}
	return 0;

cancel:
	tmpdiff2 = tmpdiff;
	tmpdiff = diff;
	while (tmpdiff != tmpdiff2) {
		hwloc_apply_diff_one(topology, tmpdiff, flags ^ HWLOC_TOPOLOGY_DIFF_APPLY_REVERSE);
		tmpdiff = tmpdiff->generic.next;
	}
	errno = EINVAL;
	return -nr; /* return the index (starting at 1) of the first element that couldn't be applied */
}
