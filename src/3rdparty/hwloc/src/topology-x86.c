/*
 * Copyright © 2010-2021 Inria.  All rights reserved.
 * Copyright © 2010-2013 Université Bordeaux
 * Copyright © 2010-2011 Cisco Systems, Inc.  All rights reserved.
 * See COPYING in top-level directory.
 *
 *
 * This backend is only used when the operating system does not export
 * the necessary hardware topology information to user-space applications.
 * Currently, FreeBSD and NetBSD only add PUs and then fallback to this
 * backend for CPU/Cache discovery.
 *
 * Other backends such as Linux have their own way to retrieve various
 * pieces of hardware topology information from the operating system
 * on various architectures, without having to use this x86-specific code.
 * But this backend is still used after them to annotate some objects with
 * additional details (CPU info in Package, Inclusiveness in Caches).
 */

#include "private/autogen/config.h"
#include "hwloc.h"
#include "private/private.h"
#include "private/debug.h"
#include "private/misc.h"
#include "private/cpuid-x86.h"

#include <sys/types.h>
#ifdef HAVE_DIRENT_H
#include <dirent.h>
#endif
#ifdef HAVE_VALGRIND_VALGRIND_H
#include <valgrind/valgrind.h>
#endif

struct hwloc_x86_backend_data_s {
  unsigned nbprocs;
  hwloc_bitmap_t apicid_set;
  int apicid_unique;
  char *src_cpuiddump_path;
  int is_knl;
};

/************************************
 * Management of cpuid dump as input
 */

struct cpuiddump {
  unsigned nr;
  struct cpuiddump_entry {
    unsigned inmask; /* which of ine[abcd]x are set on input */
    unsigned ineax;
    unsigned inebx;
    unsigned inecx;
    unsigned inedx;
    unsigned outeax;
    unsigned outebx;
    unsigned outecx;
    unsigned outedx;
  } *entries;
};

static void
cpuiddump_free(struct cpuiddump *cpuiddump)
{
  if (cpuiddump->nr)
    free(cpuiddump->entries);
  free(cpuiddump);
}

static struct cpuiddump *
cpuiddump_read(const char *dirpath, unsigned idx)
{
  struct cpuiddump *cpuiddump;
  struct cpuiddump_entry *cur;
  size_t filenamelen;
  char *filename;
  FILE *file;
  char line[128];
  unsigned nr;

  cpuiddump = malloc(sizeof(*cpuiddump));
  if (!cpuiddump) {
    fprintf(stderr, "Failed to allocate cpuiddump for PU #%u, ignoring cpuiddump.\n", idx);
    goto out;
  }

  filenamelen = strlen(dirpath) + 15;
  filename = malloc(filenamelen);
  if (!filename)
    goto out_with_dump;
  snprintf(filename, filenamelen, "%s/pu%u", dirpath, idx);
  file = fopen(filename, "r");
  if (!file) {
    fprintf(stderr, "Could not read dumped cpuid file %s, ignoring cpuiddump.\n", filename);
    goto out_with_filename;
  }

  nr = 0;
  while (fgets(line, sizeof(line), file))
    nr++;
  cpuiddump->entries = malloc(nr * sizeof(struct cpuiddump_entry));
  if (!cpuiddump->entries) {
    fprintf(stderr, "Failed to allocate %u cpuiddump entries for PU #%u, ignoring cpuiddump.\n", nr, idx);
    goto out_with_file;
  }

  fseek(file, 0, SEEK_SET);
  cur = &cpuiddump->entries[0];
  nr = 0;
  while (fgets(line, sizeof(line), file)) {
    if (*line == '#')
      continue;
    if (sscanf(line, "%x %x %x %x %x => %x %x %x %x",
	      &cur->inmask,
	      &cur->ineax, &cur->inebx, &cur->inecx, &cur->inedx,
	      &cur->outeax, &cur->outebx, &cur->outecx, &cur->outedx) == 9) {
      cur++;
      nr++;
    }
  }

  cpuiddump->nr = nr;
  fclose(file);
  free(filename);
  return cpuiddump;

 out_with_file:
  fclose(file);
 out_with_filename:
  free(filename);
 out_with_dump:
  free(cpuiddump);
 out:
  return NULL;
}

static void
cpuiddump_find_by_input(unsigned *eax, unsigned *ebx, unsigned *ecx, unsigned *edx, struct cpuiddump *cpuiddump)
{
  unsigned i;

  for(i=0; i<cpuiddump->nr; i++) {
    struct cpuiddump_entry *entry = &cpuiddump->entries[i];
    if ((entry->inmask & 0x1) && *eax != entry->ineax)
      continue;
    if ((entry->inmask & 0x2) && *ebx != entry->inebx)
      continue;
    if ((entry->inmask & 0x4) && *ecx != entry->inecx)
      continue;
    if ((entry->inmask & 0x8) && *edx != entry->inedx)
      continue;
    *eax = entry->outeax;
    *ebx = entry->outebx;
    *ecx = entry->outecx;
    *edx = entry->outedx;
    return;
  }

  fprintf(stderr, "Couldn't find %x,%x,%x,%x in dumped cpuid, returning 0s.\n",
	  *eax, *ebx, *ecx, *edx);
  *eax = 0;
  *ebx = 0;
  *ecx = 0;
  *edx = 0;
}

static void cpuid_or_from_dump(unsigned *eax, unsigned *ebx, unsigned *ecx, unsigned *edx, struct cpuiddump *src_cpuiddump)
{
  if (src_cpuiddump) {
    cpuiddump_find_by_input(eax, ebx, ecx, edx, src_cpuiddump);
  } else {
    hwloc_x86_cpuid(eax, ebx, ecx, edx);
  }
}

/*******************************
 * Core detection routines and structures
 */

enum hwloc_x86_disc_flags {
  HWLOC_X86_DISC_FLAG_FULL = (1<<0), /* discover everything instead of only annotating */
  HWLOC_X86_DISC_FLAG_TOPOEXT_NUMANODES = (1<<1) /* use AMD topoext numanode information */
};

#define has_topoext(features) ((features)[6] & (1 << 22))
#define has_x2apic(features) ((features)[4] & (1 << 21))
#define has_hybrid(features) ((features)[18] & (1 << 15))

struct cacheinfo {
  hwloc_obj_cache_type_t type;
  unsigned level;
  unsigned nbthreads_sharing;
  unsigned cacheid;

  unsigned linesize;
  unsigned linepart;
  int inclusive;
  int ways;
  unsigned sets;
  unsigned long size;
};

struct procinfo {
  unsigned present;
  unsigned apicid;
#define PKG 0
#define CORE 1
#define NODE 2
#define UNIT 3
#define TILE 4
#define MODULE 5
#define DIE 6
#define HWLOC_X86_PROCINFO_ID_NR 7
  unsigned ids[HWLOC_X86_PROCINFO_ID_NR];
  unsigned *otherids;
  unsigned levels;
  unsigned numcaches;
  struct cacheinfo *cache;
  char cpuvendor[13];
  char cpumodel[3*4*4+1];
  unsigned cpustepping;
  unsigned cpumodelnumber;
  unsigned cpufamilynumber;

  unsigned hybridcoretype;
  unsigned hybridnativemodel;
};

enum cpuid_type {
  intel,
  amd,
  zhaoxin,
  hygon,
  unknown
};

/* AMD legacy cache information from specific CPUID 0x80000005-6 leaves */
static void setup__amd_cache_legacy(struct procinfo *infos, unsigned level, hwloc_obj_cache_type_t type, unsigned nbthreads_sharing, unsigned cpuid)
{
  struct cacheinfo *cache, *tmpcaches;
  unsigned cachenum;
  unsigned long size = 0;

  if (level == 1)
    size = ((cpuid >> 24)) << 10;
  else if (level == 2)
    size = ((cpuid >> 16)) << 10;
  else if (level == 3)
    size = ((cpuid >> 18)) << 19;
  if (!size)
    return;

  tmpcaches = realloc(infos->cache, (infos->numcaches+1)*sizeof(*infos->cache));
  if (!tmpcaches)
    /* failed to allocated, ignore that cache */
    return;
  infos->cache = tmpcaches;
  cachenum = infos->numcaches++;

  cache = &infos->cache[cachenum];

  cache->type = type;
  cache->level = level;
  cache->nbthreads_sharing = nbthreads_sharing;
  cache->linesize = cpuid & 0xff;
  cache->linepart = 0;
  cache->inclusive = 0; /* old AMD (K8-K10) supposed to have exclusive caches */

  if (level == 1) {
    cache->ways = (cpuid >> 16) & 0xff;
    if (cache->ways == 0xff)
      /* Fully associative */
      cache->ways = -1;
  } else {
    static const unsigned ways_tab[] = { 0, 1, 2, 0, 4, 0, 8, 0, 16, 0, 32, 48, 64, 96, 128, -1 };
    unsigned ways = (cpuid >> 12) & 0xf;
    cache->ways = ways_tab[ways];
  }
  cache->size = size;
  cache->sets = 0;

  hwloc_debug("cache L%u t%u linesize %u ways %d size %luKB\n", cache->level, cache->nbthreads_sharing, cache->linesize, cache->ways, cache->size >> 10);
}

/* AMD legacy cache information from CPUID 0x80000005-6 leaves */
static void read_amd_caches_legacy(struct procinfo *infos, struct cpuiddump *src_cpuiddump, unsigned legacy_max_log_proc)
{
  unsigned eax, ebx, ecx, edx;

  eax = 0x80000005;
  cpuid_or_from_dump(&eax, &ebx, &ecx, &edx, src_cpuiddump);
  setup__amd_cache_legacy(infos, 1, HWLOC_OBJ_CACHE_DATA, 1, ecx); /* private L1d */
  setup__amd_cache_legacy(infos, 1, HWLOC_OBJ_CACHE_INSTRUCTION, 1, edx); /* private L1i */

  eax = 0x80000006;
  cpuid_or_from_dump(&eax, &ebx, &ecx, &edx, src_cpuiddump);
  if (ecx & 0xf000)
    /* This is actually supported on Intel but LinePerTag isn't returned in bits 8-11.
     * Could be useful if some Intels (at least before Core micro-architecture)
     * support this leaf without leaf 0x4.
     */
    setup__amd_cache_legacy(infos, 2, HWLOC_OBJ_CACHE_UNIFIED, 1, ecx); /* private L2u */
  if (edx & 0xf000)
    setup__amd_cache_legacy(infos, 3, HWLOC_OBJ_CACHE_UNIFIED, legacy_max_log_proc, edx); /* package-wide L3u */
}

/* AMD caches from CPUID 0x8000001d leaf (topoext) */
static void read_amd_caches_topoext(struct procinfo *infos, struct cpuiddump *src_cpuiddump)
{
  unsigned eax, ebx, ecx, edx;
  unsigned cachenum;
  struct cacheinfo *cache;

  /* the code below doesn't want any other cache yet */
  assert(!infos->numcaches);

  for (cachenum = 0; ; cachenum++) {
    eax = 0x8000001d;
    ecx = cachenum;
    cpuid_or_from_dump(&eax, &ebx, &ecx, &edx, src_cpuiddump);
    if ((eax & 0x1f) == 0)
      break;
    infos->numcaches++;
  }

  cache = infos->cache = malloc(infos->numcaches * sizeof(*infos->cache));
  if (cache) {
    for (cachenum = 0; ; cachenum++) {
      unsigned long linesize, linepart, ways, sets;
      eax = 0x8000001d;
      ecx = cachenum;
      cpuid_or_from_dump(&eax, &ebx, &ecx, &edx, src_cpuiddump);

      if ((eax & 0x1f) == 0)
	break;
      switch (eax & 0x1f) {
      case 1: cache->type = HWLOC_OBJ_CACHE_DATA; break;
      case 2: cache->type = HWLOC_OBJ_CACHE_INSTRUCTION; break;
      default: cache->type = HWLOC_OBJ_CACHE_UNIFIED; break;
      }

      cache->level = (eax >> 5) & 0x7;
      /* Note: actually number of cores */
      cache->nbthreads_sharing = ((eax >> 14) &  0xfff) + 1;

      cache->linesize = linesize = (ebx & 0xfff) + 1;
      cache->linepart = linepart = ((ebx >> 12) & 0x3ff) + 1;
      ways = ((ebx >> 22) & 0x3ff) + 1;

      if (eax & (1 << 9))
	/* Fully associative */
	cache->ways = -1;
      else
	cache->ways = ways;
      cache->sets = sets = ecx + 1;
      cache->size = linesize * linepart * ways * sets;
      cache->inclusive = edx & 0x2;

      hwloc_debug("cache %u L%u%c t%u linesize %lu linepart %lu ways %lu sets %lu, size %luKB\n",
		  cachenum, cache->level,
		  cache->type == HWLOC_OBJ_CACHE_DATA ? 'd' : cache->type == HWLOC_OBJ_CACHE_INSTRUCTION ? 'i' : 'u',
		  cache->nbthreads_sharing, linesize, linepart, ways, sets, cache->size >> 10);

      cache++;
    }
  } else {
    infos->numcaches = 0;
  }
}

/* Intel cache info from CPUID 0x04 leaf */
static void read_intel_caches(struct hwloc_x86_backend_data_s *data, struct procinfo *infos, struct cpuiddump *src_cpuiddump)
{
  unsigned level;
  struct cacheinfo *tmpcaches;
  unsigned eax, ebx, ecx, edx;
  unsigned oldnumcaches = infos->numcaches; /* in case we got caches above */
  unsigned cachenum;
  struct cacheinfo *cache;

  for (cachenum = 0; ; cachenum++) {
    eax = 0x04;
    ecx = cachenum;
    cpuid_or_from_dump(&eax, &ebx, &ecx, &edx, src_cpuiddump);

    hwloc_debug("cache %u type %u\n", cachenum, eax & 0x1f);
    if ((eax & 0x1f) == 0)
      break;
    level = (eax >> 5) & 0x7;
    if (data->is_knl && level == 3)
      /* KNL reports wrong L3 information (size always 0, cpuset always the entire machine, ignore it */
      break;
    infos->numcaches++;
  }

  tmpcaches = realloc(infos->cache, infos->numcaches * sizeof(*infos->cache));
  if (!tmpcaches) {
    infos->numcaches = oldnumcaches;
  } else {
    infos->cache = tmpcaches;
    cache = &infos->cache[oldnumcaches];

    for (cachenum = 0; ; cachenum++) {
      unsigned long linesize, linepart, ways, sets;
      eax = 0x04;
      ecx = cachenum;
      cpuid_or_from_dump(&eax, &ebx, &ecx, &edx, src_cpuiddump);

      if ((eax & 0x1f) == 0)
	break;
      level = (eax >> 5) & 0x7;
      if (data->is_knl && level == 3)
	/* KNL reports wrong L3 information (size always 0, cpuset always the entire machine, ignore it */
	break;
      switch (eax & 0x1f) {
      case 1: cache->type = HWLOC_OBJ_CACHE_DATA; break;
      case 2: cache->type = HWLOC_OBJ_CACHE_INSTRUCTION; break;
      default: cache->type = HWLOC_OBJ_CACHE_UNIFIED; break;
      }

      cache->level = level;
      cache->nbthreads_sharing = ((eax >> 14) & 0xfff) + 1;

      cache->linesize = linesize = (ebx & 0xfff) + 1;
      cache->linepart = linepart = ((ebx >> 12) & 0x3ff) + 1;
      ways = ((ebx >> 22) & 0x3ff) + 1;
      if (eax & (1 << 9))
        /* Fully associative */
        cache->ways = -1;
      else
        cache->ways = ways;
      cache->sets = sets = ecx + 1;
      cache->size = linesize * linepart * ways * sets;
      cache->inclusive = edx & 0x2;

      hwloc_debug("cache %u L%u%c t%u linesize %lu linepart %lu ways %lu sets %lu, size %luKB\n",
		  cachenum, cache->level,
		  cache->type == HWLOC_OBJ_CACHE_DATA ? 'd' : cache->type == HWLOC_OBJ_CACHE_INSTRUCTION ? 'i' : 'u',
		  cache->nbthreads_sharing, linesize, linepart, ways, sets, cache->size >> 10);
      cache++;
    }
  }
}

/* AMD core/thread info from CPUID 0x80000008 leaf */
static void read_amd_cores_legacy(struct procinfo *infos, struct cpuiddump *src_cpuiddump)
{
  unsigned eax, ebx, ecx, edx;
  unsigned max_nbcores;
  unsigned max_nbthreads;
  unsigned coreidsize;
  unsigned logprocid;
  unsigned threadid __hwloc_attribute_unused;

  eax = 0x80000008;
  cpuid_or_from_dump(&eax, &ebx, &ecx, &edx, src_cpuiddump);

  coreidsize = (ecx >> 12) & 0xf;
  hwloc_debug("core ID size: %u\n", coreidsize);
  if (!coreidsize) {
    max_nbcores = (ecx & 0xff) + 1;
  } else
    max_nbcores = 1 << coreidsize;
  hwloc_debug("Thus max # of cores: %u\n", max_nbcores);

  /* No multithreaded AMD for this old CPUID leaf */
  max_nbthreads = 1 ;
  hwloc_debug("and max # of threads: %u\n", max_nbthreads);

  /* legacy_max_log_proc is deprecated, it can be smaller than max_nbcores,
   * which is the maximum number of cores that the processor could theoretically support
   * (see "Multiple Core Calculation" in the AMD CPUID specification).
   * Recompute packageid/coreid accordingly.
   */
  infos->ids[PKG] = infos->apicid / max_nbcores;
  logprocid = infos->apicid % max_nbcores;
  infos->ids[CORE] = logprocid / max_nbthreads;
  threadid = logprocid % max_nbthreads;
  hwloc_debug("this is thread %u of core %u\n", threadid, infos->ids[CORE]);
}

/* AMD unit/node from CPUID 0x8000001e leaf (topoext) */
static void read_amd_cores_topoext(struct procinfo *infos, unsigned long flags, struct cpuiddump *src_cpuiddump)
{
  unsigned apic_id, nodes_per_proc = 0;
  unsigned eax, ebx, ecx, edx;

  eax = 0x8000001e;
  cpuid_or_from_dump(&eax, &ebx, &ecx, &edx, src_cpuiddump);
  infos->apicid = apic_id = eax;

  if (flags & HWLOC_X86_DISC_FLAG_TOPOEXT_NUMANODES) {
    if (infos->cpufamilynumber == 0x16) {
      /* ecx is reserved */
      infos->ids[NODE] = 0;
      nodes_per_proc = 1;
    } else {
      /* AMD other families or Hygon family 18h */
      infos->ids[NODE] = ecx & 0xff;
      nodes_per_proc = ((ecx >> 8) & 7) + 1;
    }
    if ((infos->cpufamilynumber == 0x15 && nodes_per_proc > 2)
	|| ((infos->cpufamilynumber == 0x17 || infos->cpufamilynumber == 0x18) && nodes_per_proc > 4)) {
      hwloc_debug("warning: undefined nodes_per_proc value %u, assuming it means %u\n", nodes_per_proc, nodes_per_proc);
    }
  }

  if (infos->cpufamilynumber <= 0x16) { /* topoext appeared in 0x15 and compute-units were only used in 0x15 and 0x16 */
    unsigned cores_per_unit;
    /* coreid was obtained from read_amd_cores_legacy() earlier */
    infos->ids[UNIT] = ebx & 0xff;
    cores_per_unit = ((ebx >> 8) & 0xff) + 1;
    hwloc_debug("topoext %08x, %u nodes, node %u, %u cores in unit %u\n", apic_id, nodes_per_proc, infos->ids[NODE], cores_per_unit, infos->ids[UNIT]);
    /* coreid and unitid are package-wide (core 0-15 and unit 0-7 on 16-core 2-NUMAnode processor).
     * The Linux kernel reduces theses to NUMA-node-wide (by applying %core_per_node and %unit_per node respectively).
     * It's not clear if we should do this as well.
     */
  } else {
    unsigned threads_per_core;
    infos->ids[CORE] = ebx & 0xff;
    threads_per_core = ((ebx >> 8) & 0xff) + 1;
    hwloc_debug("topoext %08x, %u nodes, node %u, %u threads in core %u\n", apic_id, nodes_per_proc, infos->ids[NODE], threads_per_core, infos->ids[CORE]);
  }
}

/* Intel core/thread or even die/module/tile from CPUID 0x0b or 0x1f leaves (v1 and v2 extended topology enumeration) */
static void read_intel_cores_exttopoenum(struct procinfo *infos, unsigned leaf, struct cpuiddump *src_cpuiddump)
{
  unsigned level, apic_nextshift, apic_number, apic_type, apic_id = 0, apic_shift = 0, id;
  unsigned threadid __hwloc_attribute_unused = 0; /* shut-up compiler */
  unsigned eax, ebx, ecx = 0, edx;
  int apic_packageshift = 0;

  for (level = 0; ; level++) {
    ecx = level;
    eax = leaf;
    cpuid_or_from_dump(&eax, &ebx, &ecx, &edx, src_cpuiddump);
    if (!eax && !ebx)
      break;
    apic_packageshift = eax & 0x1f;
  }

  if (level) {
    infos->otherids = malloc(level * sizeof(*infos->otherids));
    if (infos->otherids) {
      infos->levels = level;
      for (level = 0; ; level++) {
	ecx = level;
	eax = leaf;
	cpuid_or_from_dump(&eax, &ebx, &ecx, &edx, src_cpuiddump);
	if (!eax && !ebx)
	  break;
	apic_nextshift = eax & 0x1f;
	apic_number = ebx & 0xffff;
	apic_type = (ecx & 0xff00) >> 8;
	apic_id = edx;
	id = (apic_id >> apic_shift) & ((1 << (apic_packageshift - apic_shift)) - 1);
	hwloc_debug("x2APIC %08x %u: nextshift %u num %2u type %u id %2u\n", apic_id, level, apic_nextshift, apic_number, apic_type, id);
	infos->apicid = apic_id;
	infos->otherids[level] = UINT_MAX;
	switch (apic_type) {
	case 1:
	  threadid = id;
	  /* apic_number is the actual number of threads per core */
	  break;
	case 2:
	  infos->ids[CORE] = id;
	  /* apic_number is the actual number of threads per die */
	  break;
	case 3:
	  infos->ids[MODULE] = id;
	  /* apic_number is the actual number of threads per tile */
	  break;
	case 4:
	  infos->ids[TILE] = id;
	  /* apic_number is the actual number of threads per die */
	  break;
	case 5:
	  infos->ids[DIE] = id;
	  /* apic_number is the actual number of threads per package */
	  break;
	default:
	  hwloc_debug("x2APIC %u: unknown type %u\n", level, apic_type);
	  infos->otherids[level] = apic_id >> apic_shift;
	  break;
	}
	apic_shift = apic_nextshift;
      }
      infos->apicid = apic_id;
      infos->ids[PKG] = apic_id >> apic_shift;
      hwloc_debug("x2APIC remainder: %u\n", infos->ids[PKG]);
      hwloc_debug("this is thread %u of core %u\n", threadid, infos->ids[CORE]);
    }
  }
}

/* Fetch information from the processor itself thanks to cpuid and store it in
 * infos for summarize to analyze them globally */
static void look_proc(struct hwloc_backend *backend, struct procinfo *infos, unsigned long flags, unsigned highest_cpuid, unsigned highest_ext_cpuid, unsigned *features, enum cpuid_type cpuid_type, struct cpuiddump *src_cpuiddump)
{
  struct hwloc_x86_backend_data_s *data = backend->private_data;
  unsigned eax, ebx, ecx = 0, edx;
  unsigned cachenum;
  struct cacheinfo *cache;
  unsigned regs[4];
  unsigned legacy_max_log_proc; /* not valid on Intel processors with > 256 threads, or when cpuid 0x80000008 is supported */
  unsigned legacy_log_proc_id;
  unsigned _model, _extendedmodel, _family, _extendedfamily;

  infos->present = 1;

  /* Get apicid, legacy_max_log_proc, packageid, legacy_log_proc_id from cpuid 0x01 */
  eax = 0x01;
  cpuid_or_from_dump(&eax, &ebx, &ecx, &edx, src_cpuiddump);
  infos->apicid = ebx >> 24;
  if (edx & (1 << 28))
    legacy_max_log_proc = 1 << hwloc_flsl(((ebx >> 16) & 0xff) - 1);
  else
    legacy_max_log_proc = 1;
  hwloc_debug("APIC ID 0x%02x legacy_max_log_proc %u\n", infos->apicid, legacy_max_log_proc);
  infos->ids[PKG] = infos->apicid / legacy_max_log_proc;
  legacy_log_proc_id = infos->apicid % legacy_max_log_proc;
  hwloc_debug("phys %u legacy thread %u\n", infos->ids[PKG], legacy_log_proc_id);

  /* Get cpu model/family/stepping numbers from same cpuid */
  _model          = (eax>>4) & 0xf;
  _extendedmodel  = (eax>>16) & 0xf;
  _family         = (eax>>8) & 0xf;
  _extendedfamily = (eax>>20) & 0xff;
  if ((cpuid_type == intel || cpuid_type == amd || cpuid_type == hygon) && _family == 0xf) {
    infos->cpufamilynumber = _family + _extendedfamily;
  } else {
    infos->cpufamilynumber = _family;
  }
  if ((cpuid_type == intel && (_family == 0x6 || _family == 0xf))
      || ((cpuid_type == amd || cpuid_type == hygon) && _family == 0xf)
      || (cpuid_type == zhaoxin && (_family == 0x6 || _family == 0x7))) {
    infos->cpumodelnumber = _model + (_extendedmodel << 4);
  } else {
    infos->cpumodelnumber = _model;
  }
  infos->cpustepping = eax & 0xf;

  if (cpuid_type == intel && infos->cpufamilynumber == 0x6 &&
      (infos->cpumodelnumber == 0x57 || infos->cpumodelnumber == 0x85))
    data->is_knl = 1; /* KNM is the same as KNL */

  /* Get cpu vendor string from cpuid 0x00 */
  memset(regs, 0, sizeof(regs));
  regs[0] = 0;
  cpuid_or_from_dump(&regs[0], &regs[1], &regs[3], &regs[2], src_cpuiddump);
  memcpy(infos->cpuvendor, regs+1, 4*3);
  /* infos was calloc'ed, already ends with \0 */

  /* Get cpu model string from cpuid 0x80000002-4 */
  if (highest_ext_cpuid >= 0x80000004) {
    memset(regs, 0, sizeof(regs));
    regs[0] = 0x80000002;
    cpuid_or_from_dump(&regs[0], &regs[1], &regs[2], &regs[3], src_cpuiddump);
    memcpy(infos->cpumodel, regs, 4*4);
    regs[0] = 0x80000003;
    cpuid_or_from_dump(&regs[0], &regs[1], &regs[2], &regs[3], src_cpuiddump);
    memcpy(infos->cpumodel + 4*4, regs, 4*4);
    regs[0] = 0x80000004;
    cpuid_or_from_dump(&regs[0], &regs[1], &regs[2], &regs[3], src_cpuiddump);
    memcpy(infos->cpumodel + 4*4*2, regs, 4*4);
    /* infos was calloc'ed, already ends with \0 */
  }

  if ((cpuid_type != amd && cpuid_type != hygon) && highest_cpuid >= 0x04) {
    /* Get core/thread information from first cache reported by cpuid 0x04
     * (not supported on AMD)
     */
    eax = 0x04;
    ecx = 0;
    cpuid_or_from_dump(&eax, &ebx, &ecx, &edx, src_cpuiddump);
    if ((eax & 0x1f) != 0) {
      /* cache looks valid */
      unsigned max_nbcores;
      unsigned max_nbthreads;
      unsigned threadid __hwloc_attribute_unused;
      max_nbcores = ((eax >> 26) & 0x3f) + 1;
      max_nbthreads = legacy_max_log_proc / max_nbcores;
      hwloc_debug("thus %u threads\n", max_nbthreads);
      threadid = legacy_log_proc_id % max_nbthreads;
      infos->ids[CORE] = legacy_log_proc_id / max_nbthreads;
      hwloc_debug("this is thread %u of core %u\n", threadid, infos->ids[CORE]);
    }
  }

  if (highest_cpuid >= 0x1a && has_hybrid(features)) {
    /* Get hybrid cpu information from cpuid 0x1a */
    eax = 0x1a;
    ecx = 0;
    cpuid_or_from_dump(&eax, &ebx, &ecx, &edx, src_cpuiddump);
    infos->hybridcoretype = eax >> 24;
    infos->hybridnativemodel = eax & 0xffffff;
  }

  /*********************************************************************************
   * Get the hierarchy of thread, core, die, package, etc. from CPU-specific leaves
   */

  if (cpuid_type != intel && cpuid_type != zhaoxin && highest_ext_cpuid >= 0x80000008 && !has_x2apic(features)) {
    /* Get core/thread information from cpuid 0x80000008
     * (not supported on Intel)
     * We could ignore this codepath when x2apic is supported, but we may need
     * nodeids if HWLOC_X86_TOPOEXT_NUMANODES is set.
     */
    read_amd_cores_legacy(infos, src_cpuiddump);
  }

  if (cpuid_type != intel && cpuid_type != zhaoxin && has_topoext(features)) {
    /* Get apicid, nodeid, unitid/coreid from cpuid 0x8000001e (AMD topology extension).
     * Requires read_amd_cores_legacy() for coreid on family 0x15-16.
     *
     * Only needed when x2apic supported if NUMA nodes are needed.
     */
    read_amd_cores_topoext(infos, flags, src_cpuiddump);
  }

  if ((cpuid_type == intel) && highest_cpuid >= 0x1f) {
    /* Get package/die/module/tile/core/thread information from cpuid 0x1f
     * (Intel v2 Extended Topology Enumeration)
     */
    read_intel_cores_exttopoenum(infos, 0x1f, src_cpuiddump);

  } else if ((cpuid_type == intel || cpuid_type == amd || cpuid_type == zhaoxin)
	     && highest_cpuid >= 0x0b && has_x2apic(features)) {
    /* Get package/core/thread information from cpuid 0x0b
     * (Intel v1 Extended Topology Enumeration)
     */
    read_intel_cores_exttopoenum(infos, 0x0b, src_cpuiddump);
  }

  /**************************************
   * Get caches from CPU-specific leaves
   */

  infos->numcaches = 0;
  infos->cache = NULL;

  if (cpuid_type != intel && cpuid_type != zhaoxin && has_topoext(features)) {
    /* Get cache information from cpuid 0x8000001d (AMD topology extension) */
    read_amd_caches_topoext(infos, src_cpuiddump);

  } else if (cpuid_type != intel && cpuid_type != zhaoxin && highest_ext_cpuid >= 0x80000006) {
    /* If there's no topoext,
     * get cache information from cpuid 0x80000005 and 0x80000006.
     * (not supported on Intel)
     * It looks like we cannot have 0x80000005 without 0x80000006.
     */
    read_amd_caches_legacy(infos, src_cpuiddump, legacy_max_log_proc);
  }

  if ((cpuid_type != amd && cpuid_type != hygon) && highest_cpuid >= 0x04) {
    /* Get cache information from cpuid 0x04
     * (not supported on AMD)
     */
    read_intel_caches(data, infos, src_cpuiddump);
  }

  /* Now that we have all info, compute cacheids and apply quirks */
  for (cachenum = 0; cachenum < infos->numcaches; cachenum++) {
    cache = &infos->cache[cachenum];

    /* default cacheid value */
    cache->cacheid = infos->apicid / cache->nbthreads_sharing;

    if (cpuid_type == intel) {
      /* round nbthreads_sharing to nearest power of two to build a mask (for clearing lower bits) */
      unsigned bits = hwloc_flsl(cache->nbthreads_sharing-1);
      unsigned mask = ~((1U<<bits) - 1);
      cache->cacheid = infos->apicid & mask;

    } else if (cpuid_type == amd) {
      /* AMD quirks */
      if (infos->cpufamilynumber == 0x17
	  && cache->level == 3 && cache->nbthreads_sharing == 6) {
	/* AMD family 0x17 always shares L3 between 8 APIC ids,
	 * even when only 6 APIC ids are enabled and reported in nbthreads_sharing
	 * (on 24-core CPUs).
	 */
	cache->cacheid = infos->apicid / 8;

      } else if (infos->cpufamilynumber== 0x10 && infos->cpumodelnumber == 0x9
	  && cache->level == 3
	  && (cache->ways == -1 || (cache->ways % 2 == 0)) && cache->nbthreads_sharing >= 8) {
	/* Fix AMD family 0x10 model 0x9 (Magny-Cours) with 8 or 12 cores.
	 * The L3 (and its associativity) is actually split into two halves).
	 */
	if (cache->nbthreads_sharing == 16)
	  cache->nbthreads_sharing = 12; /* nbthreads_sharing is a power of 2 but the processor actually has 8 or 12 cores */
	cache->nbthreads_sharing /= 2;
	cache->size /= 2;
	if (cache->ways != -1)
	  cache->ways /= 2;
	/* AMD Magny-Cours 12-cores processor reserve APIC ids as AAAAAABBBBBB....
	 * among first L3 (A), second L3 (B), and unexisting cores (.).
	 * On multi-socket servers, L3 in non-first sockets may have APIC id ranges
	 * such as [16-21] that are not aligned on multiple of nbthreads_sharing (6).
	 * That means, we can't just compare apicid/nbthreads_sharing to identify siblings.
	 */
	cache->cacheid = (infos->apicid % legacy_max_log_proc) / cache->nbthreads_sharing /* cacheid within the package */
	  + 2 * (infos->apicid / legacy_max_log_proc); /* add 2 caches per previous package */

      } else if (infos->cpufamilynumber == 0x15
		 && (infos->cpumodelnumber == 0x1 /* Bulldozer */ || infos->cpumodelnumber == 0x2 /* Piledriver */)
		 && cache->level == 3 && cache->nbthreads_sharing == 6) {
	/* AMD Bulldozer and Piledriver 12-core processors have same APIC ids as Magny-Cours below,
	 * but we can't merge the checks because the original nbthreads_sharing must be exactly 6 here.
	 */
	cache->cacheid = (infos->apicid % legacy_max_log_proc) / cache->nbthreads_sharing /* cacheid within the package */
	  + 2 * (infos->apicid / legacy_max_log_proc); /* add 2 cache per previous package */
      }
    } else if (cpuid_type == hygon) {
      if (infos->cpufamilynumber == 0x18
	  && cache->level == 3 && cache->nbthreads_sharing == 6) {
        /* Hygon family 0x18 always shares L3 between 8 APIC ids,
         * even when only 6 APIC ids are enabled and reported in nbthreads_sharing
         * (on 24-core CPUs).
         */
        cache->cacheid = infos->apicid / 8;
      }
    }
  }

  if (hwloc_bitmap_isset(data->apicid_set, infos->apicid))
    data->apicid_unique = 0;
  else
    hwloc_bitmap_set(data->apicid_set, infos->apicid);
}

static void
hwloc_x86_add_cpuinfos(hwloc_obj_t obj, struct procinfo *info, int replace)
{
  char number[12];
  if (info->cpuvendor[0])
    hwloc__add_info_nodup(&obj->infos, &obj->infos_count, "CPUVendor", info->cpuvendor, replace);
  snprintf(number, sizeof(number), "%u", info->cpufamilynumber);
  hwloc__add_info_nodup(&obj->infos, &obj->infos_count, "CPUFamilyNumber", number, replace);
  snprintf(number, sizeof(number), "%u", info->cpumodelnumber);
  hwloc__add_info_nodup(&obj->infos, &obj->infos_count, "CPUModelNumber", number, replace);
  if (info->cpumodel[0]) {
    const char *c = info->cpumodel;
    while (*c == ' ')
      c++;
    hwloc__add_info_nodup(&obj->infos, &obj->infos_count, "CPUModel", c, replace);
  }
  snprintf(number, sizeof(number), "%u", info->cpustepping);
  hwloc__add_info_nodup(&obj->infos, &obj->infos_count, "CPUStepping", number, replace);
}

static void
hwloc_x86_add_groups(hwloc_topology_t topology,
		     struct procinfo *infos,
		     unsigned nbprocs,
		     hwloc_bitmap_t remaining_cpuset,
		     unsigned type,
		     const char *subtype,
		     unsigned kind,
		     int dont_merge)
{
  hwloc_bitmap_t obj_cpuset;
  hwloc_obj_t obj;
  unsigned i, j;

  while ((i = hwloc_bitmap_first(remaining_cpuset)) != (unsigned) -1) {
    unsigned packageid = infos[i].ids[PKG];
    unsigned id = infos[i].ids[type];

    if (id == (unsigned)-1) {
      hwloc_bitmap_clr(remaining_cpuset, i);
      continue;
    }

    obj_cpuset = hwloc_bitmap_alloc();
    for (j = i; j < nbprocs; j++) {
      if (infos[j].ids[type] == (unsigned) -1) {
	hwloc_bitmap_clr(remaining_cpuset, j);
	continue;
      }

      if (infos[j].ids[PKG] == packageid && infos[j].ids[type] == id) {
	hwloc_bitmap_set(obj_cpuset, j);
	hwloc_bitmap_clr(remaining_cpuset, j);
      }
    }

    obj = hwloc_alloc_setup_object(topology, HWLOC_OBJ_GROUP, id);
    obj->cpuset = obj_cpuset;
    obj->subtype = strdup(subtype);
    obj->attr->group.kind = kind;
    obj->attr->group.dont_merge = dont_merge;
    hwloc_debug_2args_bitmap("os %s %u has cpuset %s\n",
			     subtype, id, obj_cpuset);
    hwloc__insert_object_by_cpuset(topology, NULL, obj, "x86:group");
  }
}

/* Analyse information stored in infos, and build/annotate topology levels accordingly */
static void summarize(struct hwloc_backend *backend, struct procinfo *infos, unsigned long flags)
{
  struct hwloc_topology *topology = backend->topology;
  struct hwloc_x86_backend_data_s *data = backend->private_data;
  unsigned nbprocs = data->nbprocs;
  hwloc_bitmap_t complete_cpuset = hwloc_bitmap_alloc();
  unsigned i, j, l, level;
  int one = -1;
  hwloc_bitmap_t remaining_cpuset;
  int gotnuma = 0;
  int fulldiscovery = (flags & HWLOC_X86_DISC_FLAG_FULL);

#ifdef HWLOC_DEBUG
  hwloc_debug("\nSummary of x86 CPUID topology:\n");
  for(i=0; i<nbprocs; i++) {
    hwloc_debug("PU %u present=%u apicid=%u on PKG %d CORE %d DIE %d NODE %d\n",
                i, infos[i].present, infos[i].apicid,
                infos[i].ids[PKG], infos[i].ids[CORE], infos[i].ids[DIE], infos[i].ids[NODE]);
  }
  hwloc_debug("\n");
#endif

  for (i = 0; i < nbprocs; i++)
    if (infos[i].present) {
      hwloc_bitmap_set(complete_cpuset, i);
      one = i;
    }

  if (one == -1) {
    hwloc_bitmap_free(complete_cpuset);
    return;
  }

  remaining_cpuset = hwloc_bitmap_alloc();

  /* Ideally, when fulldiscovery=0, we could add any object that doesn't exist yet.
   * But what if the x86 and the native backends disagree because one is buggy? Which one to trust?
   * We only add missing caches, and annotate other existing objects for now.
   */

  if (hwloc_filter_check_keep_object_type(topology, HWLOC_OBJ_PACKAGE)) {
    /* Look for packages */
    hwloc_obj_t package;

    hwloc_bitmap_copy(remaining_cpuset, complete_cpuset);
    while ((i = hwloc_bitmap_first(remaining_cpuset)) != (unsigned) -1) {
      if (fulldiscovery) {
	unsigned packageid = infos[i].ids[PKG];
	hwloc_bitmap_t package_cpuset = hwloc_bitmap_alloc();

	for (j = i; j < nbprocs; j++) {
	  if (infos[j].ids[PKG] == packageid) {
	    hwloc_bitmap_set(package_cpuset, j);
	    hwloc_bitmap_clr(remaining_cpuset, j);
	  }
	}
	package = hwloc_alloc_setup_object(topology, HWLOC_OBJ_PACKAGE, packageid);
	package->cpuset = package_cpuset;

	hwloc_x86_add_cpuinfos(package, &infos[i], 0);

	hwloc_debug_1arg_bitmap("os package %u has cpuset %s\n",
				packageid, package_cpuset);
	hwloc__insert_object_by_cpuset(topology, NULL, package, "x86:package");

      } else {
	/* Annotate packages previously-existing packages */
	hwloc_bitmap_t set = hwloc_bitmap_alloc();
	hwloc_bitmap_set(set, i);
	package = hwloc_get_next_obj_covering_cpuset_by_type(topology, set, HWLOC_OBJ_PACKAGE, NULL);
	hwloc_bitmap_free(set);
	if (package) {
	  /* Found package above that PU, annotate if no such attribute yet */
	  hwloc_x86_add_cpuinfos(package, &infos[i], 1);
	  hwloc_bitmap_andnot(remaining_cpuset, remaining_cpuset, package->cpuset);
	} else {
	  /* No package, annotate the root object */
	  hwloc_x86_add_cpuinfos(hwloc_get_root_obj(topology), &infos[i], 1);
	  break;
	}
      }
    }
  }

  /* Look for Numa nodes inside packages (cannot be filtered-out) */
  if (fulldiscovery && (flags & HWLOC_X86_DISC_FLAG_TOPOEXT_NUMANODES)) {
    hwloc_bitmap_t node_cpuset;
    hwloc_obj_t node;

    /* FIXME: if there's memory inside the root object, divide it into NUMA nodes? */

    hwloc_bitmap_copy(remaining_cpuset, complete_cpuset);
    while ((i = hwloc_bitmap_first(remaining_cpuset)) != (unsigned) -1) {
      unsigned packageid = infos[i].ids[PKG];
      unsigned nodeid = infos[i].ids[NODE];

      if (nodeid == (unsigned)-1) {
        hwloc_bitmap_clr(remaining_cpuset, i);
	continue;
      }

      node_cpuset = hwloc_bitmap_alloc();
      for (j = i; j < nbprocs; j++) {
	if (infos[j].ids[NODE] == (unsigned) -1) {
	  hwloc_bitmap_clr(remaining_cpuset, j);
	  continue;
	}

        if (infos[j].ids[PKG] == packageid && infos[j].ids[NODE] == nodeid) {
          hwloc_bitmap_set(node_cpuset, j);
          hwloc_bitmap_clr(remaining_cpuset, j);
        }
      }
      node = hwloc_alloc_setup_object(topology, HWLOC_OBJ_NUMANODE, nodeid);
      node->cpuset = node_cpuset;
      node->nodeset = hwloc_bitmap_alloc();
      hwloc_bitmap_set(node->nodeset, nodeid);
      hwloc_debug_1arg_bitmap("os node %u has cpuset %s\n",
          nodeid, node_cpuset);
      hwloc__insert_object_by_cpuset(topology, NULL, node, "x86:numa");
      gotnuma++;
    }
  }

  if (hwloc_filter_check_keep_object_type(topology, HWLOC_OBJ_GROUP)) {
    if (fulldiscovery) {
      /* Look for AMD Compute units inside packages */
      hwloc_bitmap_copy(remaining_cpuset, complete_cpuset);
      hwloc_x86_add_groups(topology, infos, nbprocs, remaining_cpuset,
			   UNIT, "Compute Unit",
			   HWLOC_GROUP_KIND_AMD_COMPUTE_UNIT, 0);
      /* Look for Intel Modules inside packages */
      hwloc_bitmap_copy(remaining_cpuset, complete_cpuset);
      hwloc_x86_add_groups(topology, infos, nbprocs, remaining_cpuset,
			   MODULE, "Module",
			   HWLOC_GROUP_KIND_INTEL_MODULE, 0);
      /* Look for Intel Tiles inside packages */
      hwloc_bitmap_copy(remaining_cpuset, complete_cpuset);
      hwloc_x86_add_groups(topology, infos, nbprocs, remaining_cpuset,
			   TILE, "Tile",
			   HWLOC_GROUP_KIND_INTEL_TILE, 0);

      /* Look for unknown objects */
      if (infos[one].otherids) {
	for (level = infos[one].levels-1; level <= infos[one].levels-1; level--) {
	  if (infos[one].otherids[level] != UINT_MAX) {
	    hwloc_bitmap_t unknown_cpuset;
	    hwloc_obj_t unknown_obj;

	    hwloc_bitmap_copy(remaining_cpuset, complete_cpuset);
	    while ((i = hwloc_bitmap_first(remaining_cpuset)) != (unsigned) -1) {
	      unsigned unknownid = infos[i].otherids[level];

	      unknown_cpuset = hwloc_bitmap_alloc();
	      for (j = i; j < nbprocs; j++) {
		if (infos[j].otherids[level] == unknownid) {
		  hwloc_bitmap_set(unknown_cpuset, j);
		  hwloc_bitmap_clr(remaining_cpuset, j);
		}
	      }
	      unknown_obj = hwloc_alloc_setup_object(topology, HWLOC_OBJ_GROUP, unknownid);
	      unknown_obj->cpuset = unknown_cpuset;
	      unknown_obj->attr->group.kind = HWLOC_GROUP_KIND_INTEL_EXTTOPOENUM_UNKNOWN;
	      unknown_obj->attr->group.subkind = level;
	      hwloc_debug_2args_bitmap("os unknown%u %u has cpuset %s\n",
				       level, unknownid, unknown_cpuset);
	      hwloc__insert_object_by_cpuset(topology, NULL, unknown_obj, "x86:group:unknown");
	    }
	  }
	}
      }
    }
  }

  if (hwloc_filter_check_keep_object_type(topology, HWLOC_OBJ_DIE)) {
    /* Look for Intel Dies inside packages */
    if (fulldiscovery) {
      hwloc_bitmap_t die_cpuset;
      hwloc_obj_t die;

      hwloc_bitmap_copy(remaining_cpuset, complete_cpuset);
      while ((i = hwloc_bitmap_first(remaining_cpuset)) != (unsigned) -1) {
	unsigned packageid = infos[i].ids[PKG];
	unsigned dieid = infos[i].ids[DIE];

	if (dieid == (unsigned) -1) {
	  hwloc_bitmap_clr(remaining_cpuset, i);
	  continue;
	}

	die_cpuset = hwloc_bitmap_alloc();
	for (j = i; j < nbprocs; j++) {
	  if (infos[j].ids[DIE] == (unsigned) -1) {
	    hwloc_bitmap_clr(remaining_cpuset, j);
	    continue;
	  }

	  if (infos[j].ids[PKG] == packageid && infos[j].ids[DIE] == dieid) {
	    hwloc_bitmap_set(die_cpuset, j);
	    hwloc_bitmap_clr(remaining_cpuset, j);
	  }
	}
	die = hwloc_alloc_setup_object(topology, HWLOC_OBJ_DIE, dieid);
	die->cpuset = die_cpuset;
	hwloc_debug_1arg_bitmap("os die %u has cpuset %s\n",
				dieid, die_cpuset);
	hwloc__insert_object_by_cpuset(topology, NULL, die, "x86:die");
      }
    }
  }

  if (hwloc_filter_check_keep_object_type(topology, HWLOC_OBJ_CORE)) {
    /* Look for cores */
    if (fulldiscovery) {
      hwloc_bitmap_t core_cpuset;
      hwloc_obj_t core;

      hwloc_bitmap_copy(remaining_cpuset, complete_cpuset);
      while ((i = hwloc_bitmap_first(remaining_cpuset)) != (unsigned) -1) {
	unsigned packageid = infos[i].ids[PKG];
	unsigned nodeid = infos[i].ids[NODE];
	unsigned coreid = infos[i].ids[CORE];

	if (coreid == (unsigned) -1) {
	  hwloc_bitmap_clr(remaining_cpuset, i);
	  continue;
	}

	core_cpuset = hwloc_bitmap_alloc();
	for (j = i; j < nbprocs; j++) {
	  if (infos[j].ids[CORE] == (unsigned) -1) {
	    hwloc_bitmap_clr(remaining_cpuset, j);
	    continue;
	  }

	  if (infos[j].ids[PKG] == packageid && infos[j].ids[NODE] == nodeid && infos[j].ids[CORE] == coreid) {
	    hwloc_bitmap_set(core_cpuset, j);
	    hwloc_bitmap_clr(remaining_cpuset, j);
	  }
	}
	core = hwloc_alloc_setup_object(topology, HWLOC_OBJ_CORE, coreid);
	core->cpuset = core_cpuset;
	hwloc_debug_1arg_bitmap("os core %u has cpuset %s\n",
				coreid, core_cpuset);
	hwloc__insert_object_by_cpuset(topology, NULL, core, "x86:core");
      }
    }
  }

  /* Look for PUs (cannot be filtered-out) */
  if (fulldiscovery) {
    hwloc_debug("%s", "\n\n * CPU cpusets *\n\n");
    for (i=0; i<nbprocs; i++)
      if(infos[i].present) { /* Only add present PU. We don't know if others actually exist */
       struct hwloc_obj *obj = hwloc_alloc_setup_object(topology, HWLOC_OBJ_PU, i);
       obj->cpuset = hwloc_bitmap_alloc();
       hwloc_bitmap_only(obj->cpuset, i);
       hwloc_debug_1arg_bitmap("PU %u has cpuset %s\n", i, obj->cpuset);
       hwloc__insert_object_by_cpuset(topology, NULL, obj, "x86:pu");
     }
  }

  /* Look for caches */
  /* First find max level */
  level = 0;
  for (i = 0; i < nbprocs; i++)
    for (j = 0; j < infos[i].numcaches; j++)
      if (infos[i].cache[j].level > level)
        level = infos[i].cache[j].level;
  while (level > 0) {
    hwloc_obj_cache_type_t type;
    HWLOC_BUILD_ASSERT(HWLOC_OBJ_CACHE_DATA == HWLOC_OBJ_CACHE_UNIFIED+1);
    HWLOC_BUILD_ASSERT(HWLOC_OBJ_CACHE_INSTRUCTION == HWLOC_OBJ_CACHE_DATA+1);
    for (type = HWLOC_OBJ_CACHE_UNIFIED; type <= HWLOC_OBJ_CACHE_INSTRUCTION; type++) {
      /* Look for caches of that type at level level */
      hwloc_obj_type_t otype;
      hwloc_obj_t cache;

      otype = hwloc_cache_type_by_depth_type(level, type);
      if (otype == HWLOC_OBJ_TYPE_NONE)
	continue;
      if (!hwloc_filter_check_keep_object_type(topology, otype))
	continue;

      hwloc_bitmap_copy(remaining_cpuset, complete_cpuset);
      while ((i = hwloc_bitmap_first(remaining_cpuset)) != (unsigned) -1) {
	hwloc_bitmap_t puset;

	for (l = 0; l < infos[i].numcaches; l++) {
	  if (infos[i].cache[l].level == level && infos[i].cache[l].type == type)
	    break;
	}
	if (l == infos[i].numcaches) {
	  /* no cache Llevel of that type in i */
	  hwloc_bitmap_clr(remaining_cpuset, i);
	  continue;
	}

	puset = hwloc_bitmap_alloc();
	hwloc_bitmap_set(puset, i);
	cache = hwloc_get_next_obj_covering_cpuset_by_type(topology, puset, otype, NULL);
	hwloc_bitmap_free(puset);

	if (cache) {
	  /* Found cache above that PU, annotate if no such attribute yet */
	  if (!hwloc_obj_get_info_by_name(cache, "Inclusive"))
	    hwloc_obj_add_info(cache, "Inclusive", infos[i].cache[l].inclusive ? "1" : "0");
	  hwloc_bitmap_andnot(remaining_cpuset, remaining_cpuset, cache->cpuset);
	} else {
	  /* Add the missing cache */
	  hwloc_bitmap_t cache_cpuset;
	  unsigned packageid = infos[i].ids[PKG];
	  unsigned cacheid = infos[i].cache[l].cacheid;
	  /* Now look for others sharing it */
	  cache_cpuset = hwloc_bitmap_alloc();
	  for (j = i; j < nbprocs; j++) {
	    unsigned l2;
	    for (l2 = 0; l2 < infos[j].numcaches; l2++) {
	      if (infos[j].cache[l2].level == level && infos[j].cache[l2].type == type)
		break;
	    }
	    if (l2 == infos[j].numcaches) {
	      /* no cache Llevel of that type in j */
	      hwloc_bitmap_clr(remaining_cpuset, j);
	      continue;
	    }
	    if (infos[j].ids[PKG] == packageid && infos[j].cache[l2].cacheid == cacheid) {
	      hwloc_bitmap_set(cache_cpuset, j);
	      hwloc_bitmap_clr(remaining_cpuset, j);
	    }
	  }
	  cache = hwloc_alloc_setup_object(topology, otype, HWLOC_UNKNOWN_INDEX);
	  cache->attr->cache.depth = level;
	  cache->attr->cache.size = infos[i].cache[l].size;
	  cache->attr->cache.linesize = infos[i].cache[l].linesize;
	  cache->attr->cache.associativity = infos[i].cache[l].ways;
	  cache->attr->cache.type = infos[i].cache[l].type;
	  cache->cpuset = cache_cpuset;
	  hwloc_obj_add_info(cache, "Inclusive", infos[i].cache[l].inclusive ? "1" : "0");
	  hwloc_debug_2args_bitmap("os L%u cache %u has cpuset %s\n",
				   level, cacheid, cache_cpuset);
	  hwloc__insert_object_by_cpuset(topology, NULL, cache, "x86:cache");
	}
      }
    }
    level--;
  }

  /* FIXME: if KNL and L2 disabled, add tiles instead of L2 */

  hwloc_bitmap_free(remaining_cpuset);
  hwloc_bitmap_free(complete_cpuset);

  if (gotnuma)
    topology->support.discovery->numa = 1;
}

static int
look_procs(struct hwloc_backend *backend, struct procinfo *infos, unsigned long flags,
	   unsigned highest_cpuid, unsigned highest_ext_cpuid, unsigned *features, enum cpuid_type cpuid_type,
	   int (*get_cpubind)(hwloc_topology_t topology, hwloc_cpuset_t set, int flags),
	   int (*set_cpubind)(hwloc_topology_t topology, hwloc_const_cpuset_t set, int flags),
           hwloc_bitmap_t restrict_set)
{
  struct hwloc_x86_backend_data_s *data = backend->private_data;
  struct hwloc_topology *topology = backend->topology;
  unsigned nbprocs = data->nbprocs;
  hwloc_bitmap_t orig_cpuset = NULL;
  hwloc_bitmap_t set = NULL;
  unsigned i;

  if (!data->src_cpuiddump_path) {
    orig_cpuset = hwloc_bitmap_alloc();
    if (get_cpubind(topology, orig_cpuset, HWLOC_CPUBIND_STRICT)) {
      hwloc_bitmap_free(orig_cpuset);
      return -1;
    }
    set = hwloc_bitmap_alloc();
  }

  for (i = 0; i < nbprocs; i++) {
    struct cpuiddump *src_cpuiddump = NULL;

    if (restrict_set && !hwloc_bitmap_isset(restrict_set, i)) {
      /* skip this CPU outside of the binding mask */
      continue;
    }

    if (data->src_cpuiddump_path) {
      src_cpuiddump = cpuiddump_read(data->src_cpuiddump_path, i);
      if (!src_cpuiddump)
	continue;
    } else {
      hwloc_bitmap_only(set, i);
      hwloc_debug("binding to CPU%u\n", i);
      if (set_cpubind(topology, set, HWLOC_CPUBIND_STRICT)) {
	hwloc_debug("could not bind to CPU%u: %s\n", i, strerror(errno));
	continue;
      }
    }

    look_proc(backend, &infos[i], flags, highest_cpuid, highest_ext_cpuid, features, cpuid_type, src_cpuiddump);

    if (data->src_cpuiddump_path) {
      cpuiddump_free(src_cpuiddump);
    }
  }

  if (!data->src_cpuiddump_path) {
    set_cpubind(topology, orig_cpuset, 0);
    hwloc_bitmap_free(set);
    hwloc_bitmap_free(orig_cpuset);
  }

  if (data->apicid_unique) {
    summarize(backend, infos, flags);

    if (has_hybrid(features)) {
      /* use hybrid info for cpukinds */
      hwloc_bitmap_t atomset = hwloc_bitmap_alloc();
      hwloc_bitmap_t coreset = hwloc_bitmap_alloc();
      for(i=0; i<nbprocs; i++) {
        if (infos[i].hybridcoretype == 0x20)
          hwloc_bitmap_set(atomset, i);
        else if (infos[i].hybridcoretype == 0x40)
          hwloc_bitmap_set(coreset, i);
      }
      /* register IntelAtom set if any */
      if (!hwloc_bitmap_iszero(atomset)) {
        struct hwloc_info_s infoattr;
        infoattr.name = (char *) "CoreType";
        infoattr.value = (char *) "IntelAtom";
        hwloc_internal_cpukinds_register(topology, atomset, HWLOC_CPUKIND_EFFICIENCY_UNKNOWN, &infoattr, 1, 0);
        /* the cpuset is given to the callee */
      } else {
        hwloc_bitmap_free(atomset);
      }
      /* register IntelCore set if any */
      if (!hwloc_bitmap_iszero(coreset)) {
        struct hwloc_info_s infoattr;
        infoattr.name = (char *) "CoreType";
        infoattr.value = (char *) "IntelCore";
        hwloc_internal_cpukinds_register(topology, coreset, HWLOC_CPUKIND_EFFICIENCY_UNKNOWN, &infoattr, 1, 0);
        /* the cpuset is given to the callee */
      } else {
        hwloc_bitmap_free(coreset);
      }
    }
  }
  /* if !data->apicid_unique, do nothing and return success, so that the caller does nothing either */

  return 0;
}

#if defined HWLOC_FREEBSD_SYS && defined HAVE_CPUSET_SETID
#include <sys/param.h>
#include <sys/cpuset.h>
typedef cpusetid_t hwloc_x86_os_state_t;
static void hwloc_x86_os_state_save(hwloc_x86_os_state_t *state, struct cpuiddump *src_cpuiddump)
{
  if (!src_cpuiddump) {
    /* temporary make all cpus available during discovery */
    cpuset_getid(CPU_LEVEL_CPUSET, CPU_WHICH_PID, -1, state);
    cpuset_setid(CPU_WHICH_PID, -1, 0);
  }
}
static void hwloc_x86_os_state_restore(hwloc_x86_os_state_t *state, struct cpuiddump *src_cpuiddump)
{
  if (!src_cpuiddump) {
    /* restore initial cpuset */
    cpuset_setid(CPU_WHICH_PID, -1, *state);
  }
}
#else /* !defined HWLOC_FREEBSD_SYS || !defined HAVE_CPUSET_SETID */
typedef void * hwloc_x86_os_state_t;
static void hwloc_x86_os_state_save(hwloc_x86_os_state_t *state __hwloc_attribute_unused, struct cpuiddump *src_cpuiddump __hwloc_attribute_unused) { }
static void hwloc_x86_os_state_restore(hwloc_x86_os_state_t *state __hwloc_attribute_unused, struct cpuiddump *src_cpuiddump __hwloc_attribute_unused) { }
#endif /* !defined HWLOC_FREEBSD_SYS || !defined HAVE_CPUSET_SETID */

/* GenuineIntel */
#define INTEL_EBX ('G' | ('e'<<8) | ('n'<<16) | ('u'<<24))
#define INTEL_EDX ('i' | ('n'<<8) | ('e'<<16) | ('I'<<24))
#define INTEL_ECX ('n' | ('t'<<8) | ('e'<<16) | ('l'<<24))

/* AuthenticAMD */
#define AMD_EBX ('A' | ('u'<<8) | ('t'<<16) | ('h'<<24))
#define AMD_EDX ('e' | ('n'<<8) | ('t'<<16) | ('i'<<24))
#define AMD_ECX ('c' | ('A'<<8) | ('M'<<16) | ('D'<<24))

/* HYGON "HygonGenuine" */
#define HYGON_EBX ('H' | ('y'<<8) | ('g'<<16) | ('o'<<24))
#define HYGON_EDX ('n' | ('G'<<8) | ('e'<<16) | ('n'<<24))
#define HYGON_ECX ('u' | ('i'<<8) | ('n'<<16) | ('e'<<24))

/* (Zhaoxin) CentaurHauls */
#define ZX_EBX ('C' | ('e'<<8) | ('n'<<16) | ('t'<<24))
#define ZX_EDX ('a' | ('u'<<8) | ('r'<<16) | ('H'<<24))
#define ZX_ECX ('a' | ('u'<<8) | ('l'<<16) | ('s'<<24))
/* (Zhaoxin) Shanghai */
#define SH_EBX (' ' | (' '<<8) | ('S'<<16) | ('h'<<24))
#define SH_EDX ('a' | ('n'<<8) | ('g'<<16) | ('h'<<24))
#define SH_ECX ('a' | ('i'<<8) | (' '<<16) | (' '<<24))

/* fake cpubind for when nbprocs=1 and no binding support */
static int fake_get_cpubind(hwloc_topology_t topology __hwloc_attribute_unused,
			    hwloc_cpuset_t set __hwloc_attribute_unused,
			    int flags __hwloc_attribute_unused)
{
  return 0;
}
static int fake_set_cpubind(hwloc_topology_t topology __hwloc_attribute_unused,
			    hwloc_const_cpuset_t set __hwloc_attribute_unused,
			    int flags __hwloc_attribute_unused)
{
  return 0;
}

static
int hwloc_look_x86(struct hwloc_backend *backend, unsigned long flags)
{
  struct hwloc_x86_backend_data_s *data = backend->private_data;
  struct hwloc_topology *topology = backend->topology;
  unsigned nbprocs = data->nbprocs;
  unsigned eax, ebx, ecx = 0, edx;
  unsigned i;
  unsigned highest_cpuid;
  unsigned highest_ext_cpuid;
  /* This stores cpuid features with the same indexing as Linux */
  unsigned features[19] = { 0 };
  struct procinfo *infos = NULL;
  enum cpuid_type cpuid_type = unknown;
  hwloc_x86_os_state_t os_state;
  struct hwloc_binding_hooks hooks;
  struct hwloc_topology_support support;
  struct hwloc_topology_membind_support memsupport __hwloc_attribute_unused;
  int (*get_cpubind)(hwloc_topology_t topology, hwloc_cpuset_t set, int flags) = NULL;
  int (*set_cpubind)(hwloc_topology_t topology, hwloc_const_cpuset_t set, int flags) = NULL;
  hwloc_bitmap_t restrict_set = NULL;
  struct cpuiddump *src_cpuiddump = NULL;
  int ret = -1;

  /* check if binding works */
  memset(&hooks, 0, sizeof(hooks));
  support.membind = &memsupport;
  /* We could just copy the main hooks (except in some corner cases),
   * but the current overhead is negligible, so just always reget them.
   */
  hwloc_set_native_binding_hooks(&hooks, &support);
  /* in theory, those are only needed if !data->src_cpuiddump_path || HWLOC_TOPOLOGY_FLAG_RESTRICT_TO_BINDING
   * but that's the vast majority of cases anyway, and the overhead is very small.
   */

  if (data->src_cpuiddump_path) {
    /* Just read cpuid from the dump (implies !topology->is_thissystem by default) */
    src_cpuiddump = cpuiddump_read(data->src_cpuiddump_path, 0);
    if (!src_cpuiddump)
      goto out;

  } else {
    /* Using real hardware.
     * However we don't enforce topology->is_thissystem so that
     * we may still force use this backend when debugging with !thissystem.
     */

    if (hooks.get_thisthread_cpubind && hooks.set_thisthread_cpubind) {
      get_cpubind = hooks.get_thisthread_cpubind;
      set_cpubind = hooks.set_thisthread_cpubind;
    } else if (hooks.get_thisproc_cpubind && hooks.set_thisproc_cpubind) {
      /* FIXME: if called by a multithreaded program, we will restore the original process binding
       * for each thread instead of their own original thread binding.
       * See issue #158.
       */
      get_cpubind = hooks.get_thisproc_cpubind;
      set_cpubind = hooks.set_thisproc_cpubind;
    } else {
      /* we need binding support if there are multiple PUs */
      if (nbprocs > 1)
	goto out;
      get_cpubind = fake_get_cpubind;
      set_cpubind = fake_set_cpubind;
    }
  }

  if (topology->flags & HWLOC_TOPOLOGY_FLAG_RESTRICT_TO_CPUBINDING) {
    restrict_set = hwloc_bitmap_alloc();
    if (!restrict_set)
      goto out;
    if (hooks.get_thisproc_cpubind)
      hooks.get_thisproc_cpubind(topology, restrict_set, 0);
    else if (hooks.get_thisthread_cpubind)
      hooks.get_thisthread_cpubind(topology, restrict_set, 0);
    if (hwloc_bitmap_iszero(restrict_set)) {
      hwloc_bitmap_free(restrict_set);
      restrict_set = NULL;
    }
  }

  if (!src_cpuiddump && !hwloc_have_x86_cpuid())
    goto out;

  infos = calloc(nbprocs, sizeof(struct procinfo));
  if (NULL == infos)
    goto out;
  for (i = 0; i < nbprocs; i++) {
    infos[i].ids[PKG] = (unsigned) -1;
    infos[i].ids[CORE] = (unsigned) -1;
    infos[i].ids[NODE] = (unsigned) -1;
    infos[i].ids[UNIT] = (unsigned) -1;
    infos[i].ids[TILE] = (unsigned) -1;
    infos[i].ids[MODULE] = (unsigned) -1;
    infos[i].ids[DIE] = (unsigned) -1;
  }

  eax = 0x00;
  cpuid_or_from_dump(&eax, &ebx, &ecx, &edx, src_cpuiddump);
  highest_cpuid = eax;
  if (ebx == INTEL_EBX && ecx == INTEL_ECX && edx == INTEL_EDX)
    cpuid_type = intel;
  else if (ebx == AMD_EBX && ecx == AMD_ECX && edx == AMD_EDX)
    cpuid_type = amd;
  else if ((ebx == ZX_EBX && ecx == ZX_ECX && edx == ZX_EDX)
	   || (ebx == SH_EBX && ecx == SH_ECX && edx == SH_EDX))
    cpuid_type = zhaoxin;
  else if (ebx == HYGON_EBX && ecx == HYGON_ECX && edx == HYGON_EDX)
    cpuid_type = hygon;

  hwloc_debug("highest cpuid %x, cpuid type %u\n", highest_cpuid, cpuid_type);
  if (highest_cpuid < 0x01) {
      goto out_with_infos;
  }

  eax = 0x01;
  cpuid_or_from_dump(&eax, &ebx, &ecx, &edx, src_cpuiddump);
  features[0] = edx;
  features[4] = ecx;

  eax = 0x80000000;
  cpuid_or_from_dump(&eax, &ebx, &ecx, &edx, src_cpuiddump);
  highest_ext_cpuid = eax;

  hwloc_debug("highest extended cpuid %x\n", highest_ext_cpuid);

  if (highest_cpuid >= 0x7) {
    eax = 0x7;
    ecx = 0;
    cpuid_or_from_dump(&eax, &ebx, &ecx, &edx, src_cpuiddump);
    features[9] = ebx;
    features[18] = edx;
  }

  if (cpuid_type != intel && highest_ext_cpuid >= 0x80000001) {
    eax = 0x80000001;
    cpuid_or_from_dump(&eax, &ebx, &ecx, &edx, src_cpuiddump);
    features[1] = edx;
    features[6] = ecx;
  }

  hwloc_x86_os_state_save(&os_state, src_cpuiddump);

  ret = look_procs(backend, infos, flags,
		   highest_cpuid, highest_ext_cpuid, features, cpuid_type,
		   get_cpubind, set_cpubind, restrict_set);
  if (!ret)
    /* success, we're done */
    goto out_with_os_state;

  if (nbprocs == 1) {
    /* only one processor, no need to bind */
    look_proc(backend, &infos[0], flags, highest_cpuid, highest_ext_cpuid, features, cpuid_type, src_cpuiddump);
    summarize(backend, infos, flags);
    ret = 0;
  }

out_with_os_state:
  hwloc_x86_os_state_restore(&os_state, src_cpuiddump);

out_with_infos:
  if (NULL != infos) {
    for (i = 0; i < nbprocs; i++) {
      free(infos[i].cache);
      free(infos[i].otherids);
    }
    free(infos);
  }

out:
  hwloc_bitmap_free(restrict_set);
  if (src_cpuiddump)
    cpuiddump_free(src_cpuiddump);
  return ret;
}

static int
hwloc_x86_discover(struct hwloc_backend *backend, struct hwloc_disc_status *dstatus)
{
  struct hwloc_x86_backend_data_s *data = backend->private_data;
  struct hwloc_topology *topology = backend->topology;
  unsigned long flags = 0;
  int alreadypus = 0;
  int ret;

  assert(dstatus->phase == HWLOC_DISC_PHASE_CPU);

  if (topology->flags & HWLOC_TOPOLOGY_FLAG_DONT_CHANGE_BINDING) {
    /* TODO: Things would work if there's a single PU, no need to rebind */
    return 0;
  }

  if (getenv("HWLOC_X86_TOPOEXT_NUMANODES")) {
    flags |= HWLOC_X86_DISC_FLAG_TOPOEXT_NUMANODES;
  }

#if HAVE_DECL_RUNNING_ON_VALGRIND
  if (RUNNING_ON_VALGRIND && !data->src_cpuiddump_path) {
    fprintf(stderr, "hwloc x86 backend cannot work under Valgrind, disabling.\n"
	    "May be reenabled by dumping CPUIDs with hwloc-gather-cpuid\n"
	    "and reloading them under Valgrind with HWLOC_CPUID_PATH.\n");
    return 0;
  }
#endif

  if (data->src_cpuiddump_path) {
    assert(data->nbprocs > 0); /* enforced by hwloc_x86_component_instantiate() */
    topology->support.discovery->pu = 1;
  } else {
    int nbprocs = hwloc_fallback_nbprocessors(HWLOC_FALLBACK_NBPROCESSORS_INCLUDE_OFFLINE);
    if (nbprocs >= 1)
      topology->support.discovery->pu = 1;
    else
      nbprocs = 1;
    data->nbprocs = (unsigned) nbprocs;
  }

  if (topology->levels[0][0]->cpuset) {
    /* somebody else discovered things, reconnect levels so that we can look at them */
    hwloc_topology_reconnect(topology, 0);
    if (topology->nb_levels == 2 && topology->level_nbobjects[1] == data->nbprocs) {
      /* only PUs were discovered, as much as we would, complete the topology with everything else */
      alreadypus = 1;
      goto fulldiscovery;
    }

    /* several object types were added, we can't easily complete, just do partial discovery */
    ret = hwloc_look_x86(backend, flags);
    if (ret)
      hwloc_obj_add_info(topology->levels[0][0], "Backend", "x86");
    return 0;
  } else {
    /* topology is empty, initialize it */
    hwloc_alloc_root_sets(topology->levels[0][0]);
  }

fulldiscovery:
  if (hwloc_look_x86(backend, flags | HWLOC_X86_DISC_FLAG_FULL) < 0) {
    /* if failed, create PUs */
    if (!alreadypus)
      hwloc_setup_pu_level(topology, data->nbprocs);
  }

  hwloc_obj_add_info(topology->levels[0][0], "Backend", "x86");

  if (!data->src_cpuiddump_path) { /* CPUID dump works for both x86 and x86_64 */
#ifdef HAVE_UNAME
    hwloc_add_uname_info(topology, NULL); /* we already know is_thissystem() is true */
#else
    /* uname isn't available, manually setup the "Architecture" info */
#ifdef HWLOC_X86_64_ARCH
    hwloc_obj_add_info(topology->levels[0][0], "Architecture", "x86_64");
#else
    hwloc_obj_add_info(topology->levels[0][0], "Architecture", "x86");
#endif
#endif
  }

  return 1;
}

static int
hwloc_x86_check_cpuiddump_input(const char *src_cpuiddump_path, hwloc_bitmap_t set)
{

#if !(defined HWLOC_WIN_SYS && !defined __MINGW32__ && !defined __CYGWIN__) /* needs a lot of work */
  struct dirent *dirent;
  DIR *dir;
  char *path;
  FILE *file;
  char line [32];

  dir = opendir(src_cpuiddump_path);
  if (!dir) 
    return -1;

  path = malloc(strlen(src_cpuiddump_path) + strlen("/hwloc-cpuid-info") + 1);
  if (!path)
    goto out_with_dir;
  sprintf(path, "%s/hwloc-cpuid-info", src_cpuiddump_path);
  file = fopen(path, "r");
  if (!file) {
    fprintf(stderr, "Couldn't open dumped cpuid summary %s\n", path);
    goto out_with_path;
  }
  if (!fgets(line, sizeof(line), file)) {
    fprintf(stderr, "Found read dumped cpuid summary in %s\n", path);
    fclose(file);
    goto out_with_path;
  }
  fclose(file);
  if (strcmp(line, "Architecture: x86\n")) {
    fprintf(stderr, "Found non-x86 dumped cpuid summary in %s: %s\n", path, line);
    goto out_with_path;
  }
  free(path);

  while ((dirent = readdir(dir)) != NULL) {
    if (!strncmp(dirent->d_name, "pu", 2)) {
      char *end;
      unsigned long idx = strtoul(dirent->d_name+2, &end, 10);
      if (!*end)
	hwloc_bitmap_set(set, idx);
      else
	fprintf(stderr, "Ignoring invalid dirent `%s' in dumped cpuid directory `%s'\n",
		dirent->d_name, src_cpuiddump_path);
    }
  }
  closedir(dir);

  if (hwloc_bitmap_iszero(set)) {
    fprintf(stderr, "Did not find any valid pu%%u entry in dumped cpuid directory `%s'\n",
	    src_cpuiddump_path);
    return -1;
  } else if (hwloc_bitmap_last(set) != hwloc_bitmap_weight(set) - 1) {
    /* The x86 backends enforces contigous set of PUs starting at 0 so far */
    fprintf(stderr, "Found non-contigous pu%%u range in dumped cpuid directory `%s'\n",
	    src_cpuiddump_path);
    return -1;
  }

  return 0;

 out_with_path:
  free(path);
 out_with_dir:
  closedir(dir);
#endif /* HWLOC_WIN_SYS & !__MINGW32__ needs a lot of work */
  return -1;
}

static void
hwloc_x86_backend_disable(struct hwloc_backend *backend)
{
  struct hwloc_x86_backend_data_s *data = backend->private_data;
  hwloc_bitmap_free(data->apicid_set);
  free(data->src_cpuiddump_path);
  free(data);
}

static struct hwloc_backend *
hwloc_x86_component_instantiate(struct hwloc_topology *topology,
				struct hwloc_disc_component *component,
				unsigned excluded_phases __hwloc_attribute_unused,
				const void *_data1 __hwloc_attribute_unused,
				const void *_data2 __hwloc_attribute_unused,
				const void *_data3 __hwloc_attribute_unused)
{
  struct hwloc_backend *backend;
  struct hwloc_x86_backend_data_s *data;
  const char *src_cpuiddump_path;

  backend = hwloc_backend_alloc(topology, component);
  if (!backend)
    goto out;

  data = malloc(sizeof(*data));
  if (!data) {
    errno = ENOMEM;
    goto out_with_backend;
  }

  backend->private_data = data;
  backend->discover = hwloc_x86_discover;
  backend->disable = hwloc_x86_backend_disable;

  /* default values */
  data->is_knl = 0;
  data->apicid_set = hwloc_bitmap_alloc();
  data->apicid_unique = 1;
  data->src_cpuiddump_path = NULL;

  src_cpuiddump_path = getenv("HWLOC_CPUID_PATH");
  if (src_cpuiddump_path) {
    hwloc_bitmap_t set = hwloc_bitmap_alloc();
    if (!hwloc_x86_check_cpuiddump_input(src_cpuiddump_path, set)) {
      backend->is_thissystem = 0;
      data->src_cpuiddump_path = strdup(src_cpuiddump_path);
      assert(!hwloc_bitmap_iszero(set)); /* enforced by hwloc_x86_check_cpuiddump_input() */
      data->nbprocs = hwloc_bitmap_weight(set);
    } else {
      fprintf(stderr, "Ignoring dumped cpuid directory.\n");
    }
    hwloc_bitmap_free(set);
  }

  return backend;

 out_with_backend:
  free(backend);
 out:
  return NULL;
}

static struct hwloc_disc_component hwloc_x86_disc_component = {
  "x86",
  HWLOC_DISC_PHASE_CPU,
  HWLOC_DISC_PHASE_GLOBAL,
  hwloc_x86_component_instantiate,
  45, /* between native and no_os */
  1,
  NULL
};

const struct hwloc_component hwloc_x86_component = {
  HWLOC_COMPONENT_ABI,
  NULL, NULL,
  HWLOC_COMPONENT_TYPE_DISC,
  0,
  &hwloc_x86_disc_component
};
