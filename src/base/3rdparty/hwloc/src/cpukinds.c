/*
 * Copyright © 2020-2021 Inria.  All rights reserved.
 * See COPYING in top-level directory.
 */

#include "private/autogen/config.h"
#include "hwloc.h"
#include "private/private.h"
#include "private/debug.h"


/*****************
 * Basics
 */

void
hwloc_internal_cpukinds_init(struct hwloc_topology *topology)
{
  topology->cpukinds = NULL;
  topology->nr_cpukinds = 0;
  topology->nr_cpukinds_allocated = 0;
}

void
hwloc_internal_cpukinds_destroy(struct hwloc_topology *topology)
{
  unsigned i;
  for(i=0; i<topology->nr_cpukinds; i++) {
    struct hwloc_internal_cpukind_s *kind = &topology->cpukinds[i];
    hwloc_bitmap_free(kind->cpuset);
    hwloc__free_infos(kind->infos, kind->nr_infos);
  }
  free(topology->cpukinds);
  topology->cpukinds = NULL;
  topology->nr_cpukinds = 0;
}

int
hwloc_internal_cpukinds_dup(hwloc_topology_t new, hwloc_topology_t old)
{
  struct hwloc_tma *tma = new->tma;
  struct hwloc_internal_cpukind_s *kinds;
  unsigned i;

  if (!old->nr_cpukinds)
    return 0;

  kinds = hwloc_tma_malloc(tma, old->nr_cpukinds * sizeof(*kinds));
  if (!kinds)
    return -1;
  new->cpukinds = kinds;
  new->nr_cpukinds = old->nr_cpukinds;
  memcpy(kinds, old->cpukinds, old->nr_cpukinds * sizeof(*kinds));

  for(i=0;i<old->nr_cpukinds; i++) {
    kinds[i].cpuset = hwloc_bitmap_tma_dup(tma, old->cpukinds[i].cpuset);
    if (!kinds[i].cpuset) {
      new->nr_cpukinds = i;
      goto failed;
    }
    if (hwloc__tma_dup_infos(tma,
                             &kinds[i].infos, &kinds[i].nr_infos,
                             old->cpukinds[i].infos, old->cpukinds[i].nr_infos) < 0) {
      assert(!tma || !tma->dontfree); /* this tma cannot fail to allocate */
      hwloc_bitmap_free(kinds[i].cpuset);
      new->nr_cpukinds = i;
      goto failed;
    }
  }

  return 0;

 failed:
  hwloc_internal_cpukinds_destroy(new);
  return -1;
}

void
hwloc_internal_cpukinds_restrict(hwloc_topology_t topology)
{
  unsigned i;
  int removed = 0;
  for(i=0; i<topology->nr_cpukinds; i++) {
    struct hwloc_internal_cpukind_s *kind = &topology->cpukinds[i];
    hwloc_bitmap_and(kind->cpuset, kind->cpuset, hwloc_get_root_obj(topology)->cpuset);
    if (hwloc_bitmap_iszero(kind->cpuset)) {
      hwloc_bitmap_free(kind->cpuset);
      hwloc__free_infos(kind->infos, kind->nr_infos);
      memmove(kind, kind+1, (topology->nr_cpukinds - i - 1)*sizeof(*kind));
      i--;
      topology->nr_cpukinds--;
      removed = 1;
    }
  }
  if (removed)
    hwloc_internal_cpukinds_rank(topology);
}


/********************
 * Registering
 */

static __hwloc_inline int
hwloc__cpukind_check_duplicate_info(struct hwloc_internal_cpukind_s *kind,
                                    const char *name, const char *value)
{
  unsigned i;
  for(i=0; i<kind->nr_infos; i++)
    if (!strcmp(kind->infos[i].name, name)
        && !strcmp(kind->infos[i].value, value))
      return 1;
  return 0;
}

static __hwloc_inline void
hwloc__cpukind_add_infos(struct hwloc_internal_cpukind_s *kind,
                         const struct hwloc_info_s *infos, unsigned nr_infos)
{
  unsigned i;
  for(i=0; i<nr_infos; i++) {
    if (hwloc__cpukind_check_duplicate_info(kind, infos[i].name, infos[i].value))
      continue;
    hwloc__add_info(&kind->infos, &kind->nr_infos, infos[i].name, infos[i].value);
  }
}

int
hwloc_internal_cpukinds_register(hwloc_topology_t topology, hwloc_cpuset_t cpuset,
                                 int forced_efficiency,
                                 const struct hwloc_info_s *infos, unsigned nr_infos,
                                 unsigned long flags)
{
  struct hwloc_internal_cpukind_s *kinds;
  unsigned i, max, bits, oldnr, newnr;

  if (hwloc_bitmap_iszero(cpuset)) {
    hwloc_bitmap_free(cpuset);
    errno = EINVAL;
    return -1;
  }

  if (flags & ~HWLOC_CPUKINDS_REGISTER_FLAG_OVERWRITE_FORCED_EFFICIENCY) {
    errno = EINVAL;
    return -1;
  }

  /* TODO: for now, only windows provides a forced efficiency.
   * if another backend ever provides a conflicting value, the first backend value will be kept.
   * (user-provided values are not an issue, they are meant to overwrite)
   */

  /* If we have N kinds currently, we may need 2N+1 kinds after inserting the new one:
   * - each existing kind may get split into which PUs are in the new kind and which aren't.
   * - some PUs might not have been in any kind yet.
   */
  max = 2 * topology->nr_cpukinds + 1;
  /* Allocate the power-of-two above 2N+1. */
  bits = hwloc_flsl(max-1) + 1;
  max = 1U<<bits;
  /* Allocate 8 minimum to avoid multiple reallocs */
  if (max < 8)
    max = 8;

  /* Create or enlarge the array of kinds if needed */
  kinds = topology->cpukinds;
  if (max > topology->nr_cpukinds_allocated) {
    kinds = realloc(kinds, max * sizeof(*kinds));
    if (!kinds) {
      hwloc_bitmap_free(cpuset);
      return -1;
    }
    memset(&kinds[topology->nr_cpukinds_allocated], 0, (max - topology->nr_cpukinds_allocated) * sizeof(*kinds));
    topology->nr_cpukinds_allocated = max;
    topology->cpukinds = kinds;
  }

  newnr = oldnr = topology->nr_cpukinds;
  for(i=0; i<oldnr; i++) {
    int res = hwloc_bitmap_compare_inclusion(cpuset, kinds[i].cpuset);
    if (res == HWLOC_BITMAP_INTERSECTS || res == HWLOC_BITMAP_INCLUDED) {
      /* new kind with intersection of cpusets and union of infos */
      kinds[newnr].cpuset = hwloc_bitmap_alloc();
      kinds[newnr].efficiency = HWLOC_CPUKIND_EFFICIENCY_UNKNOWN;
      kinds[newnr].forced_efficiency = forced_efficiency;
      hwloc_bitmap_and(kinds[newnr].cpuset, cpuset, kinds[i].cpuset);
      hwloc__cpukind_add_infos(&kinds[newnr], kinds[i].infos, kinds[i].nr_infos);
      hwloc__cpukind_add_infos(&kinds[newnr], infos, nr_infos);
      /* remove cpuset PUs from the existing kind that we just split */
      hwloc_bitmap_andnot(kinds[i].cpuset, kinds[i].cpuset, kinds[newnr].cpuset);
      /* clear cpuset PUs that were taken care of */
      hwloc_bitmap_andnot(cpuset, cpuset, kinds[newnr].cpuset);

      newnr++;

    } else if (res == HWLOC_BITMAP_CONTAINS
               || res == HWLOC_BITMAP_EQUAL) {
      /* append new info to existing smaller (or equal) kind */
      hwloc__cpukind_add_infos(&kinds[i], infos, nr_infos);
      if ((flags & HWLOC_CPUKINDS_REGISTER_FLAG_OVERWRITE_FORCED_EFFICIENCY)
          || kinds[i].forced_efficiency == HWLOC_CPUKIND_EFFICIENCY_UNKNOWN)
        kinds[i].forced_efficiency = forced_efficiency;
      /* clear cpuset PUs that were taken care of */
      hwloc_bitmap_andnot(cpuset, cpuset, kinds[i].cpuset);

    } else {
      assert(res == HWLOC_BITMAP_DIFFERENT);
      /* nothing to do */
    }

    /* don't compare with anything else if already empty */
    if (hwloc_bitmap_iszero(cpuset))
      break;
  }

  /* add a final kind with remaining PUs if any */
  if (!hwloc_bitmap_iszero(cpuset)) {
    kinds[newnr].cpuset = cpuset;
    kinds[newnr].efficiency = HWLOC_CPUKIND_EFFICIENCY_UNKNOWN;
    kinds[newnr].forced_efficiency = forced_efficiency;
    hwloc__cpukind_add_infos(&kinds[newnr], infos, nr_infos);
    newnr++;
  } else {
    hwloc_bitmap_free(cpuset);
  }

  topology->nr_cpukinds = newnr;
  return 0;
}

int
hwloc_cpukinds_register(hwloc_topology_t topology, hwloc_cpuset_t _cpuset,
                        int forced_efficiency,
                        unsigned nr_infos, struct hwloc_info_s *infos,
                        unsigned long flags)
{
  hwloc_bitmap_t cpuset;
  int err;

  if (flags) {
    errno = EINVAL;
    return -1;
  }

  if (!_cpuset || hwloc_bitmap_iszero(_cpuset)) {
    errno = EINVAL;
    return -1;
  }

  cpuset = hwloc_bitmap_dup(_cpuset);
  if (!cpuset)
    return -1;

  if (forced_efficiency < 0)
    forced_efficiency = HWLOC_CPUKIND_EFFICIENCY_UNKNOWN;

  err = hwloc_internal_cpukinds_register(topology, cpuset, forced_efficiency, infos, nr_infos, HWLOC_CPUKINDS_REGISTER_FLAG_OVERWRITE_FORCED_EFFICIENCY);
  if (err < 0)
    return err;

  hwloc_internal_cpukinds_rank(topology);
  return 0;
}


/*********************
 * Ranking
 */

static int
hwloc__cpukinds_check_duplicate_rankings(struct hwloc_topology *topology)
{
  unsigned i,j;
  for(i=0; i<topology->nr_cpukinds; i++)
    for(j=i+1; j<topology->nr_cpukinds; j++)
      if (topology->cpukinds[i].ranking_value == topology->cpukinds[j].ranking_value)
        /* if any duplicate, fail */
        return -1;
  return 0;
}

static int
hwloc__cpukinds_try_rank_by_forced_efficiency(struct hwloc_topology *topology)
{
  unsigned i;

  hwloc_debug("Trying to rank cpukinds by forced efficiency...\n");
  for(i=0; i<topology->nr_cpukinds; i++) {
    if (topology->cpukinds[i].forced_efficiency == HWLOC_CPUKIND_EFFICIENCY_UNKNOWN)
      /* if any unknown, fail */
      return -1;
    topology->cpukinds[i].ranking_value = topology->cpukinds[i].forced_efficiency;
  }

  return hwloc__cpukinds_check_duplicate_rankings(topology);
}

struct hwloc_cpukinds_info_summary {
  int have_max_freq;
  int have_base_freq;
  int have_intel_core_type;
  struct hwloc_cpukind_info_summary {
    unsigned intel_core_type; /* 1 for atom, 2 for core */
    unsigned max_freq, base_freq; /* MHz, hence < 100000 */
  } * summaries;
};

static void
hwloc__cpukinds_summarize_info(struct hwloc_topology *topology,
                               struct hwloc_cpukinds_info_summary *summary)
{
  unsigned i, j;

  summary->have_max_freq = 1;
  summary->have_base_freq = 1;
  summary->have_intel_core_type = 1;

  for(i=0; i<topology->nr_cpukinds; i++) {
    struct hwloc_internal_cpukind_s *kind = &topology->cpukinds[i];
    for(j=0; j<kind->nr_infos; j++) {
      struct hwloc_info_s *info = &kind->infos[j];
      if (!strcmp(info->name, "FrequencyMaxMHz")) {
        summary->summaries[i].max_freq = atoi(info->value);
      } else if (!strcmp(info->name, "FrequencyBaseMHz")) {
        summary->summaries[i].base_freq = atoi(info->value);
      } else if (!strcmp(info->name, "CoreType")) {
        if (!strcmp(info->value, "IntelAtom"))
          summary->summaries[i].intel_core_type = 1;
        else if (!strcmp(info->value, "IntelCore"))
          summary->summaries[i].intel_core_type = 2;
      }
    }
    hwloc_debug("cpukind #%u has intel_core_type %u max_freq %u base_freq %u\n",
                i, summary->summaries[i].intel_core_type,
                summary->summaries[i].max_freq, summary->summaries[i].base_freq);
    if (!summary->summaries[i].base_freq)
      summary->have_base_freq = 0;
    if (!summary->summaries[i].max_freq)
      summary->have_max_freq = 0;
    if (!summary->summaries[i].intel_core_type)
      summary->have_intel_core_type = 0;
  }
}

enum hwloc_cpukinds_ranking {
  HWLOC_CPUKINDS_RANKING_DEFAULT, /* forced + frequency on ARM, forced + coretype_frequency otherwise */
  HWLOC_CPUKINDS_RANKING_NO_FORCED_EFFICIENCY, /* default without forced */
  HWLOC_CPUKINDS_RANKING_FORCED_EFFICIENCY,
  HWLOC_CPUKINDS_RANKING_CORETYPE_FREQUENCY, /* either coretype or frequency or both */
  HWLOC_CPUKINDS_RANKING_CORETYPE_FREQUENCY_STRICT, /* both coretype and frequency are required */
  HWLOC_CPUKINDS_RANKING_CORETYPE,
  HWLOC_CPUKINDS_RANKING_FREQUENCY,
  HWLOC_CPUKINDS_RANKING_FREQUENCY_MAX,
  HWLOC_CPUKINDS_RANKING_FREQUENCY_BASE,
  HWLOC_CPUKINDS_RANKING_NONE
};

static int
hwloc__cpukinds_try_rank_by_info(struct hwloc_topology *topology,
                                 enum hwloc_cpukinds_ranking heuristics,
                                 struct hwloc_cpukinds_info_summary *summary)
{
  unsigned i;

  if (HWLOC_CPUKINDS_RANKING_CORETYPE_FREQUENCY_STRICT == heuristics) {
    hwloc_debug("Trying to rank cpukinds by coretype+frequency_strict...\n");
    /* we need intel_core_type AND (base or max freq) for all kinds */
    if (!summary->have_intel_core_type
        || (!summary->have_max_freq && !summary->have_base_freq))
      return -1;
    /* rank first by coretype (Core>>Atom) then by frequency, base if available, max otherwise */
    for(i=0; i<topology->nr_cpukinds; i++) {
      struct hwloc_internal_cpukind_s *kind = &topology->cpukinds[i];
      if (summary->have_base_freq)
        kind->ranking_value = (summary->summaries[i].intel_core_type << 20) + summary->summaries[i].base_freq;
      else
        kind->ranking_value = (summary->summaries[i].intel_core_type << 20) + summary->summaries[i].max_freq;
    }

  } else if (HWLOC_CPUKINDS_RANKING_CORETYPE_FREQUENCY == heuristics) {
    hwloc_debug("Trying to rank cpukinds by coretype+frequency...\n");
    /* we need intel_core_type OR (base or max freq) for all kinds */
    if (!summary->have_intel_core_type
        && (!summary->have_max_freq && !summary->have_base_freq))
      return -1;
    /* rank first by coretype (Core>>Atom) then by frequency, base if available, max otherwise */
    for(i=0; i<topology->nr_cpukinds; i++) {
      struct hwloc_internal_cpukind_s *kind = &topology->cpukinds[i];
      if (summary->have_base_freq)
        kind->ranking_value = (summary->summaries[i].intel_core_type << 20) + summary->summaries[i].base_freq;
      else
        kind->ranking_value = (summary->summaries[i].intel_core_type << 20) + summary->summaries[i].max_freq;
    }

  } else if (HWLOC_CPUKINDS_RANKING_CORETYPE == heuristics) {
    hwloc_debug("Trying to rank cpukinds by coretype...\n");
    /* we need intel_core_type */
    if (!summary->have_intel_core_type)
      return -1;
    /* rank by coretype (Core>>Atom) */
    for(i=0; i<topology->nr_cpukinds; i++) {
      struct hwloc_internal_cpukind_s *kind = &topology->cpukinds[i];
      kind->ranking_value = (summary->summaries[i].intel_core_type << 20);
    }

  } else if (HWLOC_CPUKINDS_RANKING_FREQUENCY == heuristics) {
    hwloc_debug("Trying to rank cpukinds by frequency...\n");
    /* we need base or max freq for all kinds */
    if (!summary->have_max_freq && !summary->have_base_freq)
      return -1;
    /* rank first by frequency, base if available, max otherwise */
    for(i=0; i<topology->nr_cpukinds; i++) {
      struct hwloc_internal_cpukind_s *kind = &topology->cpukinds[i];
      if (summary->have_base_freq)
        kind->ranking_value = summary->summaries[i].base_freq;
      else
        kind->ranking_value = summary->summaries[i].max_freq;
    }

  } else if (HWLOC_CPUKINDS_RANKING_FREQUENCY_MAX == heuristics) {
    hwloc_debug("Trying to rank cpukinds by frequency max...\n");
    /* we need max freq for all kinds */
    if (!summary->have_max_freq)
      return -1;
    /* rank first by frequency, base if available, max otherwise */
    for(i=0; i<topology->nr_cpukinds; i++) {
      struct hwloc_internal_cpukind_s *kind = &topology->cpukinds[i];
      kind->ranking_value = summary->summaries[i].max_freq;
    }

  } else if (HWLOC_CPUKINDS_RANKING_FREQUENCY_BASE == heuristics) {
    hwloc_debug("Trying to rank cpukinds by frequency base...\n");
    /* we need max freq for all kinds */
    if (!summary->have_base_freq)
      return -1;
    /* rank first by frequency, base if available, max otherwise */
    for(i=0; i<topology->nr_cpukinds; i++) {
      struct hwloc_internal_cpukind_s *kind = &topology->cpukinds[i];
      kind->ranking_value = summary->summaries[i].base_freq;
    }

  } else assert(0);

  return hwloc__cpukinds_check_duplicate_rankings(topology);
}

static int hwloc__cpukinds_compare_ranking_values(const void *_a, const void *_b)
{
  const struct hwloc_internal_cpukind_s *a = _a;
  const struct hwloc_internal_cpukind_s *b = _b;
  uint64_t arv = a->ranking_value;
  uint64_t brv = b->ranking_value;
  return arv < brv ? -1 : arv > brv ? 1 : 0;
}

/* this function requires ranking values to be unique */
static void
hwloc__cpukinds_finalize_ranking(struct hwloc_topology *topology)
{
  unsigned i;
  /* sort */
  qsort(topology->cpukinds, topology->nr_cpukinds, sizeof(*topology->cpukinds), hwloc__cpukinds_compare_ranking_values);
  /* define our own efficiency between 0 and N-1 */
  for(i=0; i<topology->nr_cpukinds; i++)
    topology->cpukinds[i].efficiency = i;
}

int
hwloc_internal_cpukinds_rank(struct hwloc_topology *topology)
{
  enum hwloc_cpukinds_ranking heuristics;
  char *env;
  unsigned i;
  int err;

  if (!topology->nr_cpukinds)
    return 0;

  if (topology->nr_cpukinds == 1) {
    topology->cpukinds[0].efficiency = 0;
    return 0;
  }

  heuristics = HWLOC_CPUKINDS_RANKING_DEFAULT;
  env = getenv("HWLOC_CPUKINDS_RANKING");
  if (env) {
    if (!strcmp(env, "default"))
      heuristics = HWLOC_CPUKINDS_RANKING_DEFAULT;
    else if (!strcmp(env, "none"))
      heuristics = HWLOC_CPUKINDS_RANKING_NONE;
    else if (!strcmp(env, "coretype+frequency"))
      heuristics = HWLOC_CPUKINDS_RANKING_CORETYPE_FREQUENCY;
    else if (!strcmp(env, "coretype+frequency_strict"))
      heuristics = HWLOC_CPUKINDS_RANKING_CORETYPE_FREQUENCY_STRICT;
    else if (!strcmp(env, "coretype"))
      heuristics = HWLOC_CPUKINDS_RANKING_CORETYPE;
    else if (!strcmp(env, "frequency"))
      heuristics = HWLOC_CPUKINDS_RANKING_FREQUENCY;
    else if (!strcmp(env, "frequency_max"))
      heuristics = HWLOC_CPUKINDS_RANKING_FREQUENCY_MAX;
    else if (!strcmp(env, "frequency_base"))
      heuristics = HWLOC_CPUKINDS_RANKING_FREQUENCY_BASE;
    else if (!strcmp(env, "forced_efficiency"))
      heuristics = HWLOC_CPUKINDS_RANKING_FORCED_EFFICIENCY;
    else if (!strcmp(env, "no_forced_efficiency"))
      heuristics = HWLOC_CPUKINDS_RANKING_NO_FORCED_EFFICIENCY;
    else if (hwloc_hide_errors() < 2)
      fprintf(stderr, "hwloc: Failed to recognize HWLOC_CPUKINDS_RANKING value %s\n", env);
  }

  if (heuristics == HWLOC_CPUKINDS_RANKING_DEFAULT
      || heuristics == HWLOC_CPUKINDS_RANKING_NO_FORCED_EFFICIENCY) {
    /* default is forced_efficiency first */
    struct hwloc_cpukinds_info_summary summary;

    if (heuristics == HWLOC_CPUKINDS_RANKING_DEFAULT)
      hwloc_debug("Using default ranking strategy...\n");
    else
      hwloc_debug("Using custom ranking strategy from HWLOC_CPUKINDS_RANKING=%s\n", env);

    if (heuristics != HWLOC_CPUKINDS_RANKING_NO_FORCED_EFFICIENCY) {
      err = hwloc__cpukinds_try_rank_by_forced_efficiency(topology);
      if (!err)
        goto ready;
    }

    summary.summaries = calloc(topology->nr_cpukinds, sizeof(*summary.summaries));
    if (!summary.summaries)
      goto failed;
    hwloc__cpukinds_summarize_info(topology, &summary);

    err = hwloc__cpukinds_try_rank_by_info(topology, HWLOC_CPUKINDS_RANKING_CORETYPE_FREQUENCY, &summary);
    free(summary.summaries);
    if (!err)
      goto ready;

  } else if (heuristics == HWLOC_CPUKINDS_RANKING_FORCED_EFFICIENCY) {
    hwloc_debug("Using custom ranking strategy from HWLOC_CPUKINDS_RANKING=%s\n", env);

    err = hwloc__cpukinds_try_rank_by_forced_efficiency(topology);
    if (!err)
      goto ready;

  } else if (heuristics != HWLOC_CPUKINDS_RANKING_NONE) {
    /* custom heuristics */
    struct hwloc_cpukinds_info_summary summary;

    hwloc_debug("Using custom ranking strategy from HWLOC_CPUKINDS_RANKING=%s\n", env);

    summary.summaries = calloc(topology->nr_cpukinds, sizeof(*summary.summaries));
    if (!summary.summaries)
      goto failed;
    hwloc__cpukinds_summarize_info(topology, &summary);

    err = hwloc__cpukinds_try_rank_by_info(topology, heuristics, &summary);
    free(summary.summaries);
    if (!err)
      goto ready;
  }

 failed:
  /* failed to rank, clear efficiencies */
  for(i=0; i<topology->nr_cpukinds; i++)
    topology->cpukinds[i].efficiency = HWLOC_CPUKIND_EFFICIENCY_UNKNOWN;
  hwloc_debug("Failed to rank cpukinds.\n\n");
  return 0;

 ready:
  for(i=0; i<topology->nr_cpukinds; i++)
    hwloc_debug("cpukind #%u got ranking value %llu\n", i, (unsigned long long) topology->cpukinds[i].ranking_value);
  hwloc__cpukinds_finalize_ranking(topology);
#ifdef HWLOC_DEBUG
  for(i=0; i<topology->nr_cpukinds; i++)
    assert(topology->cpukinds[i].efficiency == (int) i);
#endif
  hwloc_debug("\n");
  return 0;
}


/*****************
 * Consulting
 */

int
hwloc_cpukinds_get_nr(hwloc_topology_t topology, unsigned long flags)
{
  if (flags) {
    errno = EINVAL;
    return -1;
  }

  return topology->nr_cpukinds;
}

int
hwloc_cpukinds_get_info(hwloc_topology_t topology,
                        unsigned id,
                        hwloc_bitmap_t cpuset,
                        int *efficiencyp,
                        unsigned *nr_infosp, struct hwloc_info_s **infosp,
                        unsigned long flags)
{
  struct hwloc_internal_cpukind_s *kind;

  if (flags) {
    errno = EINVAL;
    return -1;
  }

  if (id >= topology->nr_cpukinds) {
    errno = ENOENT;
    return -1;
  }

  kind = &topology->cpukinds[id];

  if (cpuset)
    hwloc_bitmap_copy(cpuset, kind->cpuset);

  if (efficiencyp)
    *efficiencyp = kind->efficiency;

  if (nr_infosp && infosp) {
    *nr_infosp = kind->nr_infos;
    *infosp = kind->infos;
  }
  return 0;
}

int
hwloc_cpukinds_get_by_cpuset(hwloc_topology_t topology,
                             hwloc_const_bitmap_t cpuset,
                             unsigned long flags)
{
  unsigned id;

  if (flags) {
    errno = EINVAL;
    return -1;
  }

  if (!cpuset || hwloc_bitmap_iszero(cpuset)) {
    errno = EINVAL;
    return -1;
  }

  for(id=0; id<topology->nr_cpukinds; id++) {
    struct hwloc_internal_cpukind_s *kind = &topology->cpukinds[id];
    int res = hwloc_bitmap_compare_inclusion(cpuset, kind->cpuset);
    if (res == HWLOC_BITMAP_EQUAL || res == HWLOC_BITMAP_INCLUDED) {
      return (int) id;
    } else if (res == HWLOC_BITMAP_INTERSECTS || res == HWLOC_BITMAP_CONTAINS) {
      errno = EXDEV;
      return -1;
    }
  }

  errno = ENOENT;
  return -1;
}
