/*
 * Copyright © 2009 CNRS
 * Copyright © 2009-2020 Inria.  All rights reserved.
 * Copyright © 2009-2011 Université Bordeaux
 * Copyright © 2009-2011 Cisco Systems, Inc.  All rights reserved.
 * See COPYING in top-level directory.
 */

#include "private/autogen/config.h"
#include "hwloc/autogen/config.h"
#include "hwloc.h"
#include "private/misc.h"
#include "private/private.h"
#include "private/debug.h"
#include "hwloc/bitmap.h"

#include <stdarg.h>
#include <stdio.h>
#include <assert.h>
#include <errno.h>
#include <ctype.h>

/*
 * possible improvements:
 * - have a way to change the initial allocation size:
 *   add hwloc_bitmap_set_foo() to changes a global here,
 *   and make the hwloc core call based on the early number of PUs
 * - make HWLOC_BITMAP_PREALLOC_BITS configurable, and detectable
 *   by parsing /proc/cpuinfo during configure on Linux.
 * - preallocate inside the bitmap structure (so that the whole structure is a cacheline for instance)
 *   and allocate a dedicated array only later when reallocating larger
 * - add a bitmap->ulongs_empty_first which guarantees that some first ulongs are empty,
 *   making tests much faster for big bitmaps since there's no need to look at first ulongs.
 *   no need for ulongs_empty_first to be exactly the max number of empty ulongs,
 *   clearing bits that were set earlier isn't very common.
 */

/* magic number */
#define HWLOC_BITMAP_MAGIC 0x20091007

/* preallocated bits in every bitmap */
#define HWLOC_BITMAP_PREALLOC_BITS 512
#define HWLOC_BITMAP_PREALLOC_ULONGS (HWLOC_BITMAP_PREALLOC_BITS/HWLOC_BITS_PER_LONG)

/* actual opaque type internals */
struct hwloc_bitmap_s {
  unsigned ulongs_count; /* how many ulong bitmasks are valid, >= 1 */
  unsigned ulongs_allocated; /* how many ulong bitmasks are allocated, >= ulongs_count */
  unsigned long *ulongs;
  int infinite; /* set to 1 if all bits beyond ulongs are set */
#ifdef HWLOC_DEBUG
  int magic;
#endif
};

/* overzealous check in debug-mode, not as powerful as valgrind but still useful */
#ifdef HWLOC_DEBUG
#define HWLOC__BITMAP_CHECK(set) do {				\
  assert((set)->magic == HWLOC_BITMAP_MAGIC);			\
  assert((set)->ulongs_count >= 1);				\
  assert((set)->ulongs_allocated >= (set)->ulongs_count);	\
} while (0)
#else
#define HWLOC__BITMAP_CHECK(set)
#endif

/* extract a subset from a set using an index or a cpu */
#define HWLOC_SUBBITMAP_INDEX(cpu)		((cpu)/(HWLOC_BITS_PER_LONG))
#define HWLOC_SUBBITMAP_CPU_ULBIT(cpu)		((cpu)%(HWLOC_BITS_PER_LONG))
/* Read from a bitmap ulong without knowing whether x is valid.
 * Writers should make sure that x is valid and modify set->ulongs[x] directly.
 */
#define HWLOC_SUBBITMAP_READULONG(set,x)	((x) < (set)->ulongs_count ? (set)->ulongs[x] : (set)->infinite ? HWLOC_SUBBITMAP_FULL : HWLOC_SUBBITMAP_ZERO)

/* predefined subset values */
#define HWLOC_SUBBITMAP_ZERO			0UL
#define HWLOC_SUBBITMAP_FULL			(~0UL)
#define HWLOC_SUBBITMAP_ULBIT(bit)		(1UL<<(bit))
#define HWLOC_SUBBITMAP_CPU(cpu)		HWLOC_SUBBITMAP_ULBIT(HWLOC_SUBBITMAP_CPU_ULBIT(cpu))
#define HWLOC_SUBBITMAP_ULBIT_TO(bit)		(HWLOC_SUBBITMAP_FULL>>(HWLOC_BITS_PER_LONG-1-(bit)))
#define HWLOC_SUBBITMAP_ULBIT_FROM(bit)		(HWLOC_SUBBITMAP_FULL<<(bit))
#define HWLOC_SUBBITMAP_ULBIT_FROMTO(begin,end)	(HWLOC_SUBBITMAP_ULBIT_TO(end) & HWLOC_SUBBITMAP_ULBIT_FROM(begin))

struct hwloc_bitmap_s * hwloc_bitmap_alloc(void)
{
  struct hwloc_bitmap_s * set;

  set = malloc(sizeof(struct hwloc_bitmap_s));
  if (!set)
    return NULL;

  set->ulongs_count = 1;
  set->ulongs_allocated = HWLOC_BITMAP_PREALLOC_ULONGS;
  set->ulongs = malloc(HWLOC_BITMAP_PREALLOC_ULONGS * sizeof(unsigned long));
  if (!set->ulongs) {
    free(set);
    return NULL;
  }

  set->ulongs[0] = HWLOC_SUBBITMAP_ZERO;
  set->infinite = 0;
#ifdef HWLOC_DEBUG
  set->magic = HWLOC_BITMAP_MAGIC;
#endif
  return set;
}

struct hwloc_bitmap_s * hwloc_bitmap_alloc_full(void)
{
  struct hwloc_bitmap_s * set = hwloc_bitmap_alloc();
  if (set) {
    set->infinite = 1;
    set->ulongs[0] = HWLOC_SUBBITMAP_FULL;
  }
  return set;
}

void hwloc_bitmap_free(struct hwloc_bitmap_s * set)
{
  if (!set)
    return;

  HWLOC__BITMAP_CHECK(set);
#ifdef HWLOC_DEBUG
  set->magic = 0;
#endif

  free(set->ulongs);
  free(set);
}

/* enlarge until it contains at least needed_count ulongs.
 */
static int
hwloc_bitmap_enlarge_by_ulongs(struct hwloc_bitmap_s * set, unsigned needed_count) __hwloc_attribute_warn_unused_result;
static int
hwloc_bitmap_enlarge_by_ulongs(struct hwloc_bitmap_s * set, unsigned needed_count)
{
  unsigned tmp = 1U << hwloc_flsl((unsigned long) needed_count - 1);
  if (tmp > set->ulongs_allocated) {
    unsigned long *tmpulongs;
    tmpulongs = realloc(set->ulongs, tmp * sizeof(unsigned long));
    if (!tmpulongs)
      return -1;
    set->ulongs = tmpulongs;
    set->ulongs_allocated = tmp;
  }
  return 0;
}

/* enlarge until it contains at least needed_count ulongs,
 * and update new ulongs according to the infinite field.
 */
static int
hwloc_bitmap_realloc_by_ulongs(struct hwloc_bitmap_s * set, unsigned needed_count) __hwloc_attribute_warn_unused_result;
static int
hwloc_bitmap_realloc_by_ulongs(struct hwloc_bitmap_s * set, unsigned needed_count)
{
  unsigned i;

  HWLOC__BITMAP_CHECK(set);

  if (needed_count <= set->ulongs_count)
    return 0;

  /* realloc larger if needed */
  if (hwloc_bitmap_enlarge_by_ulongs(set, needed_count) < 0)
    return -1;

  /* fill the newly allocated subset depending on the infinite flag */
  for(i=set->ulongs_count; i<needed_count; i++)
    set->ulongs[i] = set->infinite ? HWLOC_SUBBITMAP_FULL : HWLOC_SUBBITMAP_ZERO;
  set->ulongs_count = needed_count;
  return 0;
}

/* realloc until it contains at least cpu+1 bits */
#define hwloc_bitmap_realloc_by_cpu_index(set, cpu) hwloc_bitmap_realloc_by_ulongs(set, ((cpu)/HWLOC_BITS_PER_LONG)+1)

/* reset a bitmap to exactely the needed size.
 * the caller must reinitialize all ulongs and the infinite flag later.
 */
static int
hwloc_bitmap_reset_by_ulongs(struct hwloc_bitmap_s * set, unsigned needed_count) __hwloc_attribute_warn_unused_result;
static int
hwloc_bitmap_reset_by_ulongs(struct hwloc_bitmap_s * set, unsigned needed_count)
{
  if (hwloc_bitmap_enlarge_by_ulongs(set, needed_count))
    return -1;
  set->ulongs_count = needed_count;
  return 0;
}

/* reset until it contains exactly cpu+1 bits (roundup to a ulong).
 * the caller must reinitialize all ulongs and the infinite flag later.
 */
#define hwloc_bitmap_reset_by_cpu_index(set, cpu) hwloc_bitmap_reset_by_ulongs(set, ((cpu)/HWLOC_BITS_PER_LONG)+1)

struct hwloc_bitmap_s * hwloc_bitmap_tma_dup(struct hwloc_tma *tma, const struct hwloc_bitmap_s * old)
{
  struct hwloc_bitmap_s * new;

  if (!old)
    return NULL;

  HWLOC__BITMAP_CHECK(old);

  new = hwloc_tma_malloc(tma, sizeof(struct hwloc_bitmap_s));
  if (!new)
    return NULL;

  new->ulongs = hwloc_tma_malloc(tma, old->ulongs_allocated * sizeof(unsigned long));
  if (!new->ulongs) {
    free(new);
    return NULL;
  }
  new->ulongs_allocated = old->ulongs_allocated;
  new->ulongs_count = old->ulongs_count;
  memcpy(new->ulongs, old->ulongs, new->ulongs_count * sizeof(unsigned long));
  new->infinite = old->infinite;
#ifdef HWLOC_DEBUG
  new->magic = HWLOC_BITMAP_MAGIC;
#endif
  return new;
}

struct hwloc_bitmap_s * hwloc_bitmap_dup(const struct hwloc_bitmap_s * old)
{
  return hwloc_bitmap_tma_dup(NULL, old);
}

int hwloc_bitmap_copy(struct hwloc_bitmap_s * dst, const struct hwloc_bitmap_s * src)
{
  HWLOC__BITMAP_CHECK(dst);
  HWLOC__BITMAP_CHECK(src);

  if (hwloc_bitmap_reset_by_ulongs(dst, src->ulongs_count) < 0)
    return -1;

  memcpy(dst->ulongs, src->ulongs, src->ulongs_count * sizeof(unsigned long));
  dst->infinite = src->infinite;
  return 0;
}

/* Strings always use 32bit groups */
#define HWLOC_PRIxSUBBITMAP		"%08lx"
#define HWLOC_BITMAP_SUBSTRING_SIZE	32
#define HWLOC_BITMAP_SUBSTRING_LENGTH	(HWLOC_BITMAP_SUBSTRING_SIZE/4)
#define HWLOC_BITMAP_STRING_PER_LONG	(HWLOC_BITS_PER_LONG/HWLOC_BITMAP_SUBSTRING_SIZE)

int hwloc_bitmap_snprintf(char * __hwloc_restrict buf, size_t buflen, const struct hwloc_bitmap_s * __hwloc_restrict set)
{
  ssize_t size = buflen;
  char *tmp = buf;
  int res, ret = 0;
  int needcomma = 0;
  int i;
  unsigned long accum = 0;
  int accumed = 0;
#if HWLOC_BITS_PER_LONG == HWLOC_BITMAP_SUBSTRING_SIZE
  const unsigned long accum_mask = ~0UL;
#else /* HWLOC_BITS_PER_LONG != HWLOC_BITMAP_SUBSTRING_SIZE */
  const unsigned long accum_mask = ((1UL << HWLOC_BITMAP_SUBSTRING_SIZE) - 1) << (HWLOC_BITS_PER_LONG - HWLOC_BITMAP_SUBSTRING_SIZE);
#endif /* HWLOC_BITS_PER_LONG != HWLOC_BITMAP_SUBSTRING_SIZE */

  HWLOC__BITMAP_CHECK(set);

  /* mark the end in case we do nothing later */
  if (buflen > 0)
    tmp[0] = '\0';

  if (set->infinite) {
    res = hwloc_snprintf(tmp, size, "0xf...f");
    needcomma = 1;
    if (res < 0)
      return -1;
    ret += res;
    if (res >= size)
      res = size>0 ? (int)size - 1 : 0;
    tmp += res;
    size -= res;
  }

  i=(int) set->ulongs_count-1;

  if (set->infinite) {
    /* ignore starting FULL since we have 0xf...f already */
    while (i>=0 && set->ulongs[i] == HWLOC_SUBBITMAP_FULL)
      i--;
  } else {
    /* ignore starting ZERO except the last one */
    while (i>=0 && set->ulongs[i] == HWLOC_SUBBITMAP_ZERO)
      i--;
  }

  while (i>=0 || accumed) {
    /* Refill accumulator */
    if (!accumed) {
      accum = set->ulongs[i--];
      accumed = HWLOC_BITS_PER_LONG;
    }

    if (accum & accum_mask) {
      /* print the whole subset if not empty */
        res = hwloc_snprintf(tmp, size, needcomma ? ",0x" HWLOC_PRIxSUBBITMAP : "0x" HWLOC_PRIxSUBBITMAP,
		     (accum & accum_mask) >> (HWLOC_BITS_PER_LONG - HWLOC_BITMAP_SUBSTRING_SIZE));
      needcomma = 1;
    } else if (i == -1 && accumed == HWLOC_BITMAP_SUBSTRING_SIZE) {
      /* print a single 0 to mark the last subset */
      res = hwloc_snprintf(tmp, size, needcomma ? ",0x0" : "0x0");
    } else if (needcomma) {
      res = hwloc_snprintf(tmp, size, ",");
    } else {
      res = 0;
    }
    if (res < 0)
      return -1;
    ret += res;

#if HWLOC_BITS_PER_LONG == HWLOC_BITMAP_SUBSTRING_SIZE
    accum = 0;
    accumed = 0;
#else
    accum <<= HWLOC_BITMAP_SUBSTRING_SIZE;
    accumed -= HWLOC_BITMAP_SUBSTRING_SIZE;
#endif

    if (res >= size)
      res = size>0 ? (int)size - 1 : 0;

    tmp += res;
    size -= res;
  }

  /* if didn't display anything, display 0x0 */
  if (!ret) {
    res = hwloc_snprintf(tmp, size, "0x0");
    if (res < 0)
      return -1;
    ret += res;
  }

  return ret;
}

int hwloc_bitmap_asprintf(char ** strp, const struct hwloc_bitmap_s * __hwloc_restrict set)
{
  int len;
  char *buf;

  HWLOC__BITMAP_CHECK(set);

  len = hwloc_bitmap_snprintf(NULL, 0, set);
  buf = malloc(len+1);
  if (!buf)
    return -1;
  *strp = buf;
  return hwloc_bitmap_snprintf(buf, len+1, set);
}

int hwloc_bitmap_sscanf(struct hwloc_bitmap_s *set, const char * __hwloc_restrict string)
{
  const char * current = string;
  unsigned long accum = 0;
  int count=0;
  int infinite = 0;

  /* count how many substrings there are */
  count++;
  while ((current = strchr(current+1, ',')) != NULL)
    count++;

  current = string;
  if (!strncmp("0xf...f", current, 7)) {
    current += 7;
    if (*current != ',') {
      /* special case for infinite/full bitmap */
      hwloc_bitmap_fill(set);
      return 0;
    }
    current++;
    infinite = 1;
    count--;
  }

  if (hwloc_bitmap_reset_by_ulongs(set, (count + HWLOC_BITMAP_STRING_PER_LONG - 1) / HWLOC_BITMAP_STRING_PER_LONG) < 0)
    return -1;
  set->infinite = 0;

  while (*current != '\0') {
    unsigned long val;
    char *next;
    val = strtoul(current, &next, 16);

    assert(count > 0);
    count--;

    accum |= (val << ((count * HWLOC_BITMAP_SUBSTRING_SIZE) % HWLOC_BITS_PER_LONG));
    if (!(count % HWLOC_BITMAP_STRING_PER_LONG)) {
      set->ulongs[count / HWLOC_BITMAP_STRING_PER_LONG] = accum;
      accum = 0;
    }

    if (*next != ',') {
      if (*next || count > 0)
	goto failed;
      else
	break;
    }
    current = (const char*) next+1;
  }

  set->infinite = infinite; /* set at the end, to avoid spurious realloc with filled new ulongs */

  return 0;

 failed:
  /* failure to parse */
  hwloc_bitmap_zero(set);
  return -1;
}

int hwloc_bitmap_list_snprintf(char * __hwloc_restrict buf, size_t buflen, const struct hwloc_bitmap_s * __hwloc_restrict set)
{
  int prev = -1;
  ssize_t size = buflen;
  char *tmp = buf;
  int res, ret = 0;
  int needcomma = 0;

  HWLOC__BITMAP_CHECK(set);

  /* mark the end in case we do nothing later */
  if (buflen > 0)
    tmp[0] = '\0';

  while (1) {
    int begin, end;

    begin = hwloc_bitmap_next(set, prev);
    if (begin == -1)
      break;
    end = hwloc_bitmap_next_unset(set, begin);

    if (end == begin+1) {
      res = hwloc_snprintf(tmp, size, needcomma ? ",%d" : "%d", begin);
    } else if (end == -1) {
      res = hwloc_snprintf(tmp, size, needcomma ? ",%d-" : "%d-", begin);
    } else {
      res = hwloc_snprintf(tmp, size, needcomma ? ",%d-%d" : "%d-%d", begin, end-1);
    }
    if (res < 0)
      return -1;
    ret += res;

    if (res >= size)
      res = size>0 ? (int)size - 1 : 0;

    tmp += res;
    size -= res;
    needcomma = 1;

    if (end == -1)
      break;
    else
      prev = end - 1;
  }

  return ret;
}

int hwloc_bitmap_list_asprintf(char ** strp, const struct hwloc_bitmap_s * __hwloc_restrict set)
{
  int len;
  char *buf;

  HWLOC__BITMAP_CHECK(set);

  len = hwloc_bitmap_list_snprintf(NULL, 0, set);
  buf = malloc(len+1);
  if (!buf)
    return -1;
  *strp = buf;
  return hwloc_bitmap_list_snprintf(buf, len+1, set);
}

int hwloc_bitmap_list_sscanf(struct hwloc_bitmap_s *set, const char * __hwloc_restrict string)
{
  const char * current = string;
  char *next;
  long begin = -1, val;

  hwloc_bitmap_zero(set);

  while (*current != '\0') {

    /* ignore empty ranges */
    while (*current == ',' || *current == ' ')
      current++;

    val = strtoul(current, &next, 0);
    /* make sure we got at least one digit */
    if (next == current)
      goto failed;

    if (begin != -1) {
      /* finishing a range */
      if (hwloc_bitmap_set_range(set, begin, val) < 0)
        goto failed;
      begin = -1;

    } else if (*next == '-') {
      /* starting a new range */
      if (*(next+1) == '\0') {
	/* infinite range */
	if (hwloc_bitmap_set_range(set, val, -1) < 0)
	  goto failed;
        break;
      } else {
	/* normal range */
	begin = val;
      }

    } else if (*next == ',' || *next == ' ' || *next == '\0') {
      /* single digit */
      hwloc_bitmap_set(set, val);
    }

    if (*next == '\0')
      break;
    current = next+1;
  }

  return 0;

 failed:
  /* failure to parse */
  hwloc_bitmap_zero(set);
  return -1;
}

int hwloc_bitmap_taskset_snprintf(char * __hwloc_restrict buf, size_t buflen, const struct hwloc_bitmap_s * __hwloc_restrict set)
{
  ssize_t size = buflen;
  char *tmp = buf;
  int res, ret = 0;
  int started = 0;
  int i;

  HWLOC__BITMAP_CHECK(set);

  /* mark the end in case we do nothing later */
  if (buflen > 0)
    tmp[0] = '\0';

  if (set->infinite) {
    res = hwloc_snprintf(tmp, size, "0xf...f");
    started = 1;
    if (res < 0)
      return -1;
    ret += res;
    if (res >= size)
      res = size>0 ? (int)size - 1 : 0;
    tmp += res;
    size -= res;
  }

  i=set->ulongs_count-1;

  if (set->infinite) {
    /* ignore starting FULL since we have 0xf...f already */
    while (i>=0 && set->ulongs[i] == HWLOC_SUBBITMAP_FULL)
      i--;
  } else {
    /* ignore starting ZERO except the last one */
    while (i>=1 && set->ulongs[i] == HWLOC_SUBBITMAP_ZERO)
      i--;
  }

  while (i>=0) {
    unsigned long val = set->ulongs[i--];
    if (started) {
      /* print the whole subset */
#if HWLOC_BITS_PER_LONG == 64
      res = hwloc_snprintf(tmp, size, "%016lx", val);
#else
      res = hwloc_snprintf(tmp, size, "%08lx", val);
#endif
    } else if (val || i == -1) {
      res = hwloc_snprintf(tmp, size, "0x%lx", val);
      started = 1;
    } else {
      res = 0;
    }
    if (res < 0)
      return -1;
    ret += res;
    if (res >= size)
      res = size>0 ? (int)size - 1 : 0;
    tmp += res;
    size -= res;
  }

  /* if didn't display anything, display 0x0 */
  if (!ret) {
    res = hwloc_snprintf(tmp, size, "0x0");
    if (res < 0)
      return -1;
    ret += res;
  }

  return ret;
}

int hwloc_bitmap_taskset_asprintf(char ** strp, const struct hwloc_bitmap_s * __hwloc_restrict set)
{
  int len;
  char *buf;

  HWLOC__BITMAP_CHECK(set);

  len = hwloc_bitmap_taskset_snprintf(NULL, 0, set);
  buf = malloc(len+1);
  if (!buf)
    return -1;
  *strp = buf;
  return hwloc_bitmap_taskset_snprintf(buf, len+1, set);
}

int hwloc_bitmap_taskset_sscanf(struct hwloc_bitmap_s *set, const char * __hwloc_restrict string)
{
  const char * current = string;
  int chars;
  int count;
  int infinite = 0;

  if (!strncmp("0xf...f", current, 7)) {
    /* infinite bitmap */
    infinite = 1;
    current += 7;
    if (*current == '\0') {
      /* special case for infinite/full bitmap */
      hwloc_bitmap_fill(set);
      return 0;
    }
  } else {
    /* finite bitmap */
    if (!strncmp("0x", current, 2))
      current += 2;
    if (*current == '\0') {
      /* special case for empty bitmap */
      hwloc_bitmap_zero(set);
      return 0;
    }
  }
  /* we know there are other characters now */

  chars = (int)strlen(current);
  count = (chars * 4 + HWLOC_BITS_PER_LONG - 1) / HWLOC_BITS_PER_LONG;

  if (hwloc_bitmap_reset_by_ulongs(set, count) < 0)
    return -1;
  set->infinite = 0;

  while (*current != '\0') {
    int tmpchars;
    char ustr[17];
    unsigned long val;
    char *next;

    tmpchars = chars % (HWLOC_BITS_PER_LONG/4);
    if (!tmpchars)
      tmpchars = (HWLOC_BITS_PER_LONG/4);

    memcpy(ustr, current, tmpchars);
    ustr[tmpchars] = '\0';
    val = strtoul(ustr, &next, 16);
    if (*next != '\0')
      goto failed;

    set->ulongs[count-1] = val;

    current += tmpchars;
    chars -= tmpchars;
    count--;
  }

  set->infinite = infinite; /* set at the end, to avoid spurious realloc with filled new ulongs */

  return 0;

 failed:
  /* failure to parse */
  hwloc_bitmap_zero(set);
  return -1;
}

static void hwloc_bitmap__zero(struct hwloc_bitmap_s *set)
{
	unsigned i;
	for(i=0; i<set->ulongs_count; i++)
		set->ulongs[i] = HWLOC_SUBBITMAP_ZERO;
	set->infinite = 0;
}

void hwloc_bitmap_zero(struct hwloc_bitmap_s * set)
{
	HWLOC__BITMAP_CHECK(set);

	HWLOC_BUILD_ASSERT(HWLOC_BITMAP_PREALLOC_ULONGS >= 1);
	if (hwloc_bitmap_reset_by_ulongs(set, 1) < 0) {
		/* cannot fail since we preallocate some ulongs.
		 * if we ever preallocate nothing, we'll reset to 0 ulongs.
		 */
	}
	hwloc_bitmap__zero(set);
}

static void hwloc_bitmap__fill(struct hwloc_bitmap_s * set)
{
	unsigned i;
	for(i=0; i<set->ulongs_count; i++)
		set->ulongs[i] = HWLOC_SUBBITMAP_FULL;
	set->infinite = 1;
}

void hwloc_bitmap_fill(struct hwloc_bitmap_s * set)
{
	HWLOC__BITMAP_CHECK(set);

	HWLOC_BUILD_ASSERT(HWLOC_BITMAP_PREALLOC_ULONGS >= 1);
	if (hwloc_bitmap_reset_by_ulongs(set, 1) < 0) {
		/* cannot fail since we pre-allocate some ulongs.
		 * if we ever pre-allocate nothing, we'll reset to 0 ulongs.
		 */
	}
	hwloc_bitmap__fill(set);
}

int hwloc_bitmap_from_ulong(struct hwloc_bitmap_s *set, unsigned long mask)
{
	HWLOC__BITMAP_CHECK(set);

	HWLOC_BUILD_ASSERT(HWLOC_BITMAP_PREALLOC_ULONGS >= 1);
	if (hwloc_bitmap_reset_by_ulongs(set, 1) < 0) {
		/* cannot fail since we pre-allocate some ulongs.
		 * if ever pre-allocate nothing, we may have to return a failure.
		 */
	}
	set->ulongs[0] = mask; /* there's always at least one ulong allocated */
	set->infinite = 0;
	return 0;
}

int hwloc_bitmap_from_ith_ulong(struct hwloc_bitmap_s *set, unsigned i, unsigned long mask)
{
	unsigned j;

	HWLOC__BITMAP_CHECK(set);

	if (hwloc_bitmap_reset_by_ulongs(set, i+1) < 0)
		return -1;

	set->ulongs[i] = mask;
	for(j=0; j<i; j++)
		set->ulongs[j] = HWLOC_SUBBITMAP_ZERO;
	set->infinite = 0;
	return 0;
}

int hwloc_bitmap_from_ulongs(struct hwloc_bitmap_s *set, unsigned nr, const unsigned long *masks)
{
	unsigned j;

	HWLOC__BITMAP_CHECK(set);

	if (hwloc_bitmap_reset_by_ulongs(set, nr) < 0)
		return -1;

	for(j=0; j<nr; j++)
		set->ulongs[j] = masks[j];
	set->infinite = 0;
	return 0;
}

unsigned long hwloc_bitmap_to_ulong(const struct hwloc_bitmap_s *set)
{
	HWLOC__BITMAP_CHECK(set);

	return set->ulongs[0]; /* there's always at least one ulong allocated */
}

unsigned long hwloc_bitmap_to_ith_ulong(const struct hwloc_bitmap_s *set, unsigned i)
{
	HWLOC__BITMAP_CHECK(set);

	return HWLOC_SUBBITMAP_READULONG(set, i);
}

int hwloc_bitmap_to_ulongs(const struct hwloc_bitmap_s *set, unsigned nr, unsigned long *masks)
{
	unsigned j;

	HWLOC__BITMAP_CHECK(set);

	for(j=0; j<nr; j++)
		masks[j] = HWLOC_SUBBITMAP_READULONG(set, j);
	return 0;
}

int hwloc_bitmap_nr_ulongs(const struct hwloc_bitmap_s *set)
{
	unsigned last;

	HWLOC__BITMAP_CHECK(set);

	if (set->infinite)
		return -1;

	last = hwloc_bitmap_last(set);
	return (last + HWLOC_BITS_PER_LONG)/HWLOC_BITS_PER_LONG;
}

int hwloc_bitmap_only(struct hwloc_bitmap_s * set, unsigned cpu)
{
	unsigned index_ = HWLOC_SUBBITMAP_INDEX(cpu);

	HWLOC__BITMAP_CHECK(set);

	if (hwloc_bitmap_reset_by_cpu_index(set, cpu) < 0)
		return -1;

	hwloc_bitmap__zero(set);
	set->ulongs[index_] |= HWLOC_SUBBITMAP_CPU(cpu);
	return 0;
}

int hwloc_bitmap_allbut(struct hwloc_bitmap_s * set, unsigned cpu)
{
	unsigned index_ = HWLOC_SUBBITMAP_INDEX(cpu);

	HWLOC__BITMAP_CHECK(set);

	if (hwloc_bitmap_reset_by_cpu_index(set, cpu) < 0)
		return -1;

	hwloc_bitmap__fill(set);
	set->ulongs[index_] &= ~HWLOC_SUBBITMAP_CPU(cpu);
	return 0;
}

int hwloc_bitmap_set(struct hwloc_bitmap_s * set, unsigned cpu)
{
	unsigned index_ = HWLOC_SUBBITMAP_INDEX(cpu);

	HWLOC__BITMAP_CHECK(set);

	/* nothing to do if setting inside the infinite part of the bitmap */
	if (set->infinite && cpu >= set->ulongs_count * HWLOC_BITS_PER_LONG)
		return 0;

	if (hwloc_bitmap_realloc_by_cpu_index(set, cpu) < 0)
		return -1;

	set->ulongs[index_] |= HWLOC_SUBBITMAP_CPU(cpu);
	return 0;
}

int hwloc_bitmap_set_range(struct hwloc_bitmap_s * set, unsigned begincpu, int _endcpu)
{
	unsigned i;
	unsigned beginset,endset;
	unsigned endcpu = (unsigned) _endcpu;

	HWLOC__BITMAP_CHECK(set);

	if (endcpu < begincpu)
		return 0;
	if (set->infinite && begincpu >= set->ulongs_count * HWLOC_BITS_PER_LONG)
		/* setting only in the already-set infinite part, nothing to do */
		return 0;

	if (_endcpu == -1) {
		/* infinite range */

		/* make sure we can play with the ulong that contains begincpu */
		if (hwloc_bitmap_realloc_by_cpu_index(set, begincpu) < 0)
			return -1;

		/* update the ulong that contains begincpu */
		beginset = HWLOC_SUBBITMAP_INDEX(begincpu);
		set->ulongs[beginset] |= HWLOC_SUBBITMAP_ULBIT_FROM(HWLOC_SUBBITMAP_CPU_ULBIT(begincpu));
		/* set ulongs after begincpu if any already allocated */
		for(i=beginset+1; i<set->ulongs_count; i++)
			set->ulongs[i] = HWLOC_SUBBITMAP_FULL;
		/* mark the infinity as set */
		set->infinite = 1;
	} else {
		/* finite range */

		/* ignore the part of the range that overlaps with the already-set infinite part */
		if (set->infinite && endcpu >= set->ulongs_count * HWLOC_BITS_PER_LONG)
			endcpu = set->ulongs_count * HWLOC_BITS_PER_LONG - 1;
		/* make sure we can play with the ulongs that contain begincpu and endcpu */
		if (hwloc_bitmap_realloc_by_cpu_index(set, endcpu) < 0)
			return -1;

		/* update first and last ulongs */
		beginset = HWLOC_SUBBITMAP_INDEX(begincpu);
		endset = HWLOC_SUBBITMAP_INDEX(endcpu);
		if (beginset == endset) {
			set->ulongs[beginset] |= HWLOC_SUBBITMAP_ULBIT_FROMTO(HWLOC_SUBBITMAP_CPU_ULBIT(begincpu), HWLOC_SUBBITMAP_CPU_ULBIT(endcpu));
		} else {
			set->ulongs[beginset] |= HWLOC_SUBBITMAP_ULBIT_FROM(HWLOC_SUBBITMAP_CPU_ULBIT(begincpu));
			set->ulongs[endset] |= HWLOC_SUBBITMAP_ULBIT_TO(HWLOC_SUBBITMAP_CPU_ULBIT(endcpu));
		}
		/* set ulongs in the middle of the range */
		for(i=beginset+1; i<endset; i++)
			set->ulongs[i] = HWLOC_SUBBITMAP_FULL;
	}

	return 0;
}

int hwloc_bitmap_set_ith_ulong(struct hwloc_bitmap_s *set, unsigned i, unsigned long mask)
{
	HWLOC__BITMAP_CHECK(set);

	if (hwloc_bitmap_realloc_by_ulongs(set, i+1) < 0)
		return -1;

	set->ulongs[i] = mask;
	return 0;
}

int hwloc_bitmap_clr(struct hwloc_bitmap_s * set, unsigned cpu)
{
	unsigned index_ = HWLOC_SUBBITMAP_INDEX(cpu);

	HWLOC__BITMAP_CHECK(set);

	/* nothing to do if clearing inside the infinitely-unset part of the bitmap */
	if (!set->infinite && cpu >= set->ulongs_count * HWLOC_BITS_PER_LONG)
		return 0;

	if (hwloc_bitmap_realloc_by_cpu_index(set, cpu) < 0)
		return -1;

	set->ulongs[index_] &= ~HWLOC_SUBBITMAP_CPU(cpu);
	return 0;
}

int hwloc_bitmap_clr_range(struct hwloc_bitmap_s * set, unsigned begincpu, int _endcpu)
{
	unsigned i;
	unsigned beginset,endset;
	unsigned endcpu = (unsigned) _endcpu;

	HWLOC__BITMAP_CHECK(set);

	if (endcpu < begincpu)
		return 0;

	if (!set->infinite && begincpu >= set->ulongs_count * HWLOC_BITS_PER_LONG)
		/* clearing only in the already-unset infinite part, nothing to do */
		return 0;

	if (_endcpu == -1) {
		/* infinite range */

		/* make sure we can play with the ulong that contains begincpu */
		if (hwloc_bitmap_realloc_by_cpu_index(set, begincpu) < 0)
			return -1;

		/* update the ulong that contains begincpu */
		beginset = HWLOC_SUBBITMAP_INDEX(begincpu);
		set->ulongs[beginset] &= ~HWLOC_SUBBITMAP_ULBIT_FROM(HWLOC_SUBBITMAP_CPU_ULBIT(begincpu));
		/* clear ulong after begincpu if any already allocated */
		for(i=beginset+1; i<set->ulongs_count; i++)
			set->ulongs[i] = HWLOC_SUBBITMAP_ZERO;
		/* mark the infinity as unset */
		set->infinite = 0;
	} else {
		/* finite range */

		/* ignore the part of the range that overlaps with the already-unset infinite part */
		if (!set->infinite && endcpu >= set->ulongs_count * HWLOC_BITS_PER_LONG)
			endcpu = set->ulongs_count * HWLOC_BITS_PER_LONG - 1;
		/* make sure we can play with the ulongs that contain begincpu and endcpu */
		if (hwloc_bitmap_realloc_by_cpu_index(set, endcpu) < 0)
			return -1;

		/* update first and last ulongs */
		beginset = HWLOC_SUBBITMAP_INDEX(begincpu);
		endset = HWLOC_SUBBITMAP_INDEX(endcpu);
		if (beginset == endset) {
			set->ulongs[beginset] &= ~HWLOC_SUBBITMAP_ULBIT_FROMTO(HWLOC_SUBBITMAP_CPU_ULBIT(begincpu), HWLOC_SUBBITMAP_CPU_ULBIT(endcpu));
		} else {
			set->ulongs[beginset] &= ~HWLOC_SUBBITMAP_ULBIT_FROM(HWLOC_SUBBITMAP_CPU_ULBIT(begincpu));
			set->ulongs[endset] &= ~HWLOC_SUBBITMAP_ULBIT_TO(HWLOC_SUBBITMAP_CPU_ULBIT(endcpu));
		}
		/* clear ulongs in the middle of the range */
		for(i=beginset+1; i<endset; i++)
			set->ulongs[i] = HWLOC_SUBBITMAP_ZERO;
	}

	return 0;
}

int hwloc_bitmap_isset(const struct hwloc_bitmap_s * set, unsigned cpu)
{
	unsigned index_ = HWLOC_SUBBITMAP_INDEX(cpu);

	HWLOC__BITMAP_CHECK(set);

	return (HWLOC_SUBBITMAP_READULONG(set, index_) & HWLOC_SUBBITMAP_CPU(cpu)) != 0;
}

int hwloc_bitmap_iszero(const struct hwloc_bitmap_s *set)
{
	unsigned i;

	HWLOC__BITMAP_CHECK(set);

	if (set->infinite)
		return 0;
	for(i=0; i<set->ulongs_count; i++)
		if (set->ulongs[i] != HWLOC_SUBBITMAP_ZERO)
			return 0;
	return 1;
}

int hwloc_bitmap_isfull(const struct hwloc_bitmap_s *set)
{
	unsigned i;

	HWLOC__BITMAP_CHECK(set);

	if (!set->infinite)
		return 0;
	for(i=0; i<set->ulongs_count; i++)
		if (set->ulongs[i] != HWLOC_SUBBITMAP_FULL)
			return 0;
	return 1;
}

int hwloc_bitmap_isequal (const struct hwloc_bitmap_s *set1, const struct hwloc_bitmap_s *set2)
{
	unsigned count1 = set1->ulongs_count;
	unsigned count2 = set2->ulongs_count;
	unsigned min_count = count1 < count2 ? count1 : count2;
	unsigned i;

	HWLOC__BITMAP_CHECK(set1);
	HWLOC__BITMAP_CHECK(set2);

	for(i=0; i<min_count; i++)
		if (set1->ulongs[i] != set2->ulongs[i])
			return 0;

	if (count1 != count2) {
		unsigned long w1 = set1->infinite ? HWLOC_SUBBITMAP_FULL : HWLOC_SUBBITMAP_ZERO;
		unsigned long w2 = set2->infinite ? HWLOC_SUBBITMAP_FULL : HWLOC_SUBBITMAP_ZERO;
		for(i=min_count; i<count1; i++) {
			if (set1->ulongs[i] != w2)
				return 0;
		}
		for(i=min_count; i<count2; i++) {
			if (set2->ulongs[i] != w1)
				return 0;
		}
	}

	if (set1->infinite != set2->infinite)
		return 0;

	return 1;
}

int hwloc_bitmap_intersects (const struct hwloc_bitmap_s *set1, const struct hwloc_bitmap_s *set2)
{
	unsigned count1 = set1->ulongs_count;
	unsigned count2 = set2->ulongs_count;
	unsigned min_count = count1 < count2 ? count1 : count2;
	unsigned i;

	HWLOC__BITMAP_CHECK(set1);
	HWLOC__BITMAP_CHECK(set2);

	for(i=0; i<min_count; i++)
		if (set1->ulongs[i] & set2->ulongs[i])
			return 1;

	if (count1 != count2) {
		if (set2->infinite) {
			for(i=min_count; i<set1->ulongs_count; i++)
				if (set1->ulongs[i])
					return 1;
		}
		if (set1->infinite) {
			for(i=min_count; i<set2->ulongs_count; i++)
				if (set2->ulongs[i])
					return 1;
		}
	}

	if (set1->infinite && set2->infinite)
		return 1;

	return 0;
}

int hwloc_bitmap_isincluded (const struct hwloc_bitmap_s *sub_set, const struct hwloc_bitmap_s *super_set)
{
	unsigned super_count = super_set->ulongs_count;
	unsigned sub_count = sub_set->ulongs_count;
	unsigned min_count = super_count < sub_count ? super_count : sub_count;
	unsigned i;

	HWLOC__BITMAP_CHECK(sub_set);
	HWLOC__BITMAP_CHECK(super_set);

	for(i=0; i<min_count; i++)
		if (super_set->ulongs[i] != (super_set->ulongs[i] | sub_set->ulongs[i]))
			return 0;

	if (super_count != sub_count) {
		if (!super_set->infinite)
			for(i=min_count; i<sub_count; i++)
				if (sub_set->ulongs[i])
					return 0;
		if (sub_set->infinite)
			for(i=min_count; i<super_count; i++)
				if (super_set->ulongs[i] != HWLOC_SUBBITMAP_FULL)
					return 0;
	}

	if (sub_set->infinite && !super_set->infinite)
		return 0;

	return 1;
}

int hwloc_bitmap_or (struct hwloc_bitmap_s *res, const struct hwloc_bitmap_s *set1, const struct hwloc_bitmap_s *set2)
{
	/* cache counts so that we can reset res even if it's also set1 or set2 */
	unsigned count1 = set1->ulongs_count;
	unsigned count2 = set2->ulongs_count;
	unsigned max_count = count1 > count2 ? count1 : count2;
	unsigned min_count = count1 + count2 - max_count;
	unsigned i;

	HWLOC__BITMAP_CHECK(res);
	HWLOC__BITMAP_CHECK(set1);
	HWLOC__BITMAP_CHECK(set2);

	if (hwloc_bitmap_reset_by_ulongs(res, max_count) < 0)
		return -1;

	for(i=0; i<min_count; i++)
		res->ulongs[i] = set1->ulongs[i] | set2->ulongs[i];

	if (count1 != count2) {
		if (min_count < count1) {
			if (set2->infinite) {
				res->ulongs_count = min_count;
			} else {
				for(i=min_count; i<max_count; i++)
					res->ulongs[i] = set1->ulongs[i];
			}
		} else {
			if (set1->infinite) {
				res->ulongs_count = min_count;
			} else {
				for(i=min_count; i<max_count; i++)
					res->ulongs[i] = set2->ulongs[i];
			}
		}
	}

	res->infinite = set1->infinite || set2->infinite;
	return 0;
}

int hwloc_bitmap_and (struct hwloc_bitmap_s *res, const struct hwloc_bitmap_s *set1, const struct hwloc_bitmap_s *set2)
{
	/* cache counts so that we can reset res even if it's also set1 or set2 */
	unsigned count1 = set1->ulongs_count;
	unsigned count2 = set2->ulongs_count;
	unsigned max_count = count1 > count2 ? count1 : count2;
	unsigned min_count = count1 + count2 - max_count;
	unsigned i;

	HWLOC__BITMAP_CHECK(res);
	HWLOC__BITMAP_CHECK(set1);
	HWLOC__BITMAP_CHECK(set2);

	if (hwloc_bitmap_reset_by_ulongs(res, max_count) < 0)
		return -1;

	for(i=0; i<min_count; i++)
		res->ulongs[i] = set1->ulongs[i] & set2->ulongs[i];

	if (count1 != count2) {
		if (min_count < count1) {
			if (set2->infinite) {
				for(i=min_count; i<max_count; i++)
					res->ulongs[i] = set1->ulongs[i];
			} else {
				res->ulongs_count = min_count;
			}
		} else {
			if (set1->infinite) {
				for(i=min_count; i<max_count; i++)
					res->ulongs[i] = set2->ulongs[i];
			} else {
				res->ulongs_count = min_count;
			}
		}
	}

	res->infinite = set1->infinite && set2->infinite;
	return 0;
}

int hwloc_bitmap_andnot (struct hwloc_bitmap_s *res, const struct hwloc_bitmap_s *set1, const struct hwloc_bitmap_s *set2)
{
	/* cache counts so that we can reset res even if it's also set1 or set2 */
	unsigned count1 = set1->ulongs_count;
	unsigned count2 = set2->ulongs_count;
	unsigned max_count = count1 > count2 ? count1 : count2;
	unsigned min_count = count1 + count2 - max_count;
	unsigned i;

	HWLOC__BITMAP_CHECK(res);
	HWLOC__BITMAP_CHECK(set1);
	HWLOC__BITMAP_CHECK(set2);

	if (hwloc_bitmap_reset_by_ulongs(res, max_count) < 0)
		return -1;

	for(i=0; i<min_count; i++)
		res->ulongs[i] = set1->ulongs[i] & ~set2->ulongs[i];

	if (count1 != count2) {
		if (min_count < count1) {
			if (!set2->infinite) {
				for(i=min_count; i<max_count; i++)
					res->ulongs[i] = set1->ulongs[i];
			} else {
				res->ulongs_count = min_count;
			}
		} else {
			if (set1->infinite) {
				for(i=min_count; i<max_count; i++)
					res->ulongs[i] = ~set2->ulongs[i];
			} else {
				res->ulongs_count = min_count;
			}
		}
	}

	res->infinite = set1->infinite && !set2->infinite;
	return 0;
}

int hwloc_bitmap_xor (struct hwloc_bitmap_s *res, const struct hwloc_bitmap_s *set1, const struct hwloc_bitmap_s *set2)
{
	/* cache counts so that we can reset res even if it's also set1 or set2 */
	unsigned count1 = set1->ulongs_count;
	unsigned count2 = set2->ulongs_count;
	unsigned max_count = count1 > count2 ? count1 : count2;
	unsigned min_count = count1 + count2 - max_count;
	unsigned i;

	HWLOC__BITMAP_CHECK(res);
	HWLOC__BITMAP_CHECK(set1);
	HWLOC__BITMAP_CHECK(set2);

	if (hwloc_bitmap_reset_by_ulongs(res, max_count) < 0)
		return -1;

	for(i=0; i<min_count; i++)
		res->ulongs[i] = set1->ulongs[i] ^ set2->ulongs[i];

	if (count1 != count2) {
		if (min_count < count1) {
			unsigned long w2 = set2->infinite ? HWLOC_SUBBITMAP_FULL : HWLOC_SUBBITMAP_ZERO;
			for(i=min_count; i<max_count; i++)
				res->ulongs[i] = set1->ulongs[i] ^ w2;
		} else {
			unsigned long w1 = set1->infinite ? HWLOC_SUBBITMAP_FULL : HWLOC_SUBBITMAP_ZERO;
			for(i=min_count; i<max_count; i++)
				res->ulongs[i] = set2->ulongs[i] ^ w1;
		}
	}

	res->infinite = (!set1->infinite) != (!set2->infinite);
	return 0;
}

int hwloc_bitmap_not (struct hwloc_bitmap_s *res, const struct hwloc_bitmap_s *set)
{
	unsigned count = set->ulongs_count;
	unsigned i;

	HWLOC__BITMAP_CHECK(res);
	HWLOC__BITMAP_CHECK(set);

	if (hwloc_bitmap_reset_by_ulongs(res, count) < 0)
		return -1;

	for(i=0; i<count; i++)
		res->ulongs[i] = ~set->ulongs[i];

	res->infinite = !set->infinite;
	return 0;
}

int hwloc_bitmap_first(const struct hwloc_bitmap_s * set)
{
	unsigned i;

	HWLOC__BITMAP_CHECK(set);

	for(i=0; i<set->ulongs_count; i++) {
		/* subsets are unsigned longs, use ffsl */
		unsigned long w = set->ulongs[i];
		if (w)
			return hwloc_ffsl(w) - 1 + HWLOC_BITS_PER_LONG*i;
	}

	if (set->infinite)
		return set->ulongs_count * HWLOC_BITS_PER_LONG;

	return -1;
}

int hwloc_bitmap_first_unset(const struct hwloc_bitmap_s * set)
{
	unsigned i;

	HWLOC__BITMAP_CHECK(set);

	for(i=0; i<set->ulongs_count; i++) {
		/* subsets are unsigned longs, use ffsl */
		unsigned long w = ~set->ulongs[i];
		if (w)
			return hwloc_ffsl(w) - 1 + HWLOC_BITS_PER_LONG*i;
	}

	if (!set->infinite)
		return set->ulongs_count * HWLOC_BITS_PER_LONG;

	return -1;
}

int hwloc_bitmap_last(const struct hwloc_bitmap_s * set)
{
	int i;

	HWLOC__BITMAP_CHECK(set);

	if (set->infinite)
		return -1;

	for(i=(int)set->ulongs_count-1; i>=0; i--) {
		/* subsets are unsigned longs, use flsl */
		unsigned long w = set->ulongs[i];
		if (w)
			return hwloc_flsl(w) - 1 + HWLOC_BITS_PER_LONG*i;
	}

	return -1;
}

int hwloc_bitmap_last_unset(const struct hwloc_bitmap_s * set)
{
	int i;

	HWLOC__BITMAP_CHECK(set);

	if (!set->infinite)
		return -1;

	for(i=(int)set->ulongs_count-1; i>=0; i--) {
		/* subsets are unsigned longs, use flsl */
		unsigned long w = ~set->ulongs[i];
		if (w)
			return hwloc_flsl(w) - 1 + HWLOC_BITS_PER_LONG*i;
	}

	return -1;
}

int hwloc_bitmap_next(const struct hwloc_bitmap_s * set, int prev_cpu)
{
	unsigned i = HWLOC_SUBBITMAP_INDEX(prev_cpu + 1);

	HWLOC__BITMAP_CHECK(set);

	if (i >= set->ulongs_count) {
		if (set->infinite)
			return prev_cpu + 1;
		else
			return -1;
	}

	for(; i<set->ulongs_count; i++) {
		/* subsets are unsigned longs, use ffsl */
		unsigned long w = set->ulongs[i];

		/* if the prev cpu is in the same word as the possible next one,
		   we need to mask out previous cpus */
		if (prev_cpu >= 0 && HWLOC_SUBBITMAP_INDEX((unsigned) prev_cpu) == i)
			w &= ~HWLOC_SUBBITMAP_ULBIT_TO(HWLOC_SUBBITMAP_CPU_ULBIT(prev_cpu));

		if (w)
			return hwloc_ffsl(w) - 1 + HWLOC_BITS_PER_LONG*i;
	}

	if (set->infinite)
		return set->ulongs_count * HWLOC_BITS_PER_LONG;

	return -1;
}

int hwloc_bitmap_next_unset(const struct hwloc_bitmap_s * set, int prev_cpu)
{
	unsigned i = HWLOC_SUBBITMAP_INDEX(prev_cpu + 1);

	HWLOC__BITMAP_CHECK(set);

	if (i >= set->ulongs_count) {
		if (!set->infinite)
			return prev_cpu + 1;
		else
			return -1;
	}

	for(; i<set->ulongs_count; i++) {
		/* subsets are unsigned longs, use ffsl */
		unsigned long w = ~set->ulongs[i];

		/* if the prev cpu is in the same word as the possible next one,
		   we need to mask out previous cpus */
		if (prev_cpu >= 0 && HWLOC_SUBBITMAP_INDEX((unsigned) prev_cpu) == i)
			w &= ~HWLOC_SUBBITMAP_ULBIT_TO(HWLOC_SUBBITMAP_CPU_ULBIT(prev_cpu));

		if (w)
			return hwloc_ffsl(w) - 1 + HWLOC_BITS_PER_LONG*i;
	}

	if (!set->infinite)
		return set->ulongs_count * HWLOC_BITS_PER_LONG;

	return -1;
}

int hwloc_bitmap_singlify(struct hwloc_bitmap_s * set)
{
	unsigned i;
	int found = 0;

	HWLOC__BITMAP_CHECK(set);

	for(i=0; i<set->ulongs_count; i++) {
		if (found) {
			set->ulongs[i] = HWLOC_SUBBITMAP_ZERO;
			continue;
		} else {
			/* subsets are unsigned longs, use ffsl */
			unsigned long w = set->ulongs[i];
			if (w) {
				int _ffs = hwloc_ffsl(w);
				set->ulongs[i] = HWLOC_SUBBITMAP_CPU(_ffs-1);
				found = 1;
			}
		}
	}

	if (set->infinite) {
		if (found) {
			set->infinite = 0;
		} else {
			/* set the first non allocated bit */
			unsigned first = set->ulongs_count * HWLOC_BITS_PER_LONG;
			set->infinite = 0; /* do not let realloc fill the newly allocated sets */
			return hwloc_bitmap_set(set, first);
		}
	}

	return 0;
}

int hwloc_bitmap_compare_first(const struct hwloc_bitmap_s * set1, const struct hwloc_bitmap_s * set2)
{
	unsigned count1 = set1->ulongs_count;
	unsigned count2 = set2->ulongs_count;
	unsigned max_count = count1 > count2 ? count1 : count2;
	unsigned min_count = count1 + count2 - max_count;
	unsigned i;

	HWLOC__BITMAP_CHECK(set1);
	HWLOC__BITMAP_CHECK(set2);

	for(i=0; i<min_count; i++) {
		unsigned long w1 = set1->ulongs[i];
		unsigned long w2 = set2->ulongs[i];
		if (w1 || w2) {
			int _ffs1 = hwloc_ffsl(w1);
			int _ffs2 = hwloc_ffsl(w2);
			/* if both have a bit set, compare for real */
			if (_ffs1 && _ffs2)
				return _ffs1-_ffs2;
			/* one is empty, and it is considered higher, so reverse-compare them */
			return _ffs2-_ffs1;
		}
	}

	if (count1 != count2) {
		if (min_count < count2) {
			for(i=min_count; i<count2; i++) {
				unsigned long w2 = set2->ulongs[i];
				if (set1->infinite)
					return -!(w2 & 1);
				else if (w2)
					return 1;
			}
		} else {
			for(i=min_count; i<count1; i++) {
				unsigned long w1 = set1->ulongs[i];
				if (set2->infinite)
					return !(w1 & 1);
				else if (w1)
					return -1;
			}
		}
	}

	return !!set1->infinite - !!set2->infinite;
}

int hwloc_bitmap_compare(const struct hwloc_bitmap_s * set1, const struct hwloc_bitmap_s * set2)
{
	unsigned count1 = set1->ulongs_count;
	unsigned count2 = set2->ulongs_count;
	unsigned max_count = count1 > count2 ? count1 : count2;
	unsigned min_count = count1 + count2 - max_count;
	int i;

	HWLOC__BITMAP_CHECK(set1);
	HWLOC__BITMAP_CHECK(set2);

	if ((!set1->infinite) != (!set2->infinite))
		return !!set1->infinite - !!set2->infinite;

	if (count1 != count2) {
		if (min_count < count2) {
			unsigned long val1 = set1->infinite ? HWLOC_SUBBITMAP_FULL :  HWLOC_SUBBITMAP_ZERO;
			for(i=(int)max_count-1; i>=(int) min_count; i--) {
				unsigned long val2 = set2->ulongs[i];
				if (val1 == val2)
					continue;
				return val1 < val2 ? -1 : 1;
			}
		} else {
			unsigned long val2 = set2->infinite ? HWLOC_SUBBITMAP_FULL :  HWLOC_SUBBITMAP_ZERO;
			for(i=(int)max_count-1; i>=(int) min_count; i--) {
				unsigned long val1 = set1->ulongs[i];
				if (val1 == val2)
					continue;
				return val1 < val2 ? -1 : 1;
			}
		}
	}

	for(i=(int)min_count-1; i>=0; i--) {
		unsigned long val1 = set1->ulongs[i];
		unsigned long val2 = set2->ulongs[i];
		if (val1 == val2)
			continue;
		return val1 < val2 ? -1 : 1;
	}

	return 0;
}

int hwloc_bitmap_weight(const struct hwloc_bitmap_s * set)
{
	int weight = 0;
	unsigned i;

	HWLOC__BITMAP_CHECK(set);

	if (set->infinite)
		return -1;

	for(i=0; i<set->ulongs_count; i++)
		weight += hwloc_weight_long(set->ulongs[i]);
	return weight;
}

int hwloc_bitmap_compare_inclusion(const struct hwloc_bitmap_s * set1, const struct hwloc_bitmap_s * set2)
{
	unsigned max_count = set1->ulongs_count > set2->ulongs_count ? set1->ulongs_count : set2->ulongs_count;
	int result = HWLOC_BITMAP_EQUAL; /* means empty sets return equal */
	int empty1 = 1;
	int empty2 = 1;
	unsigned i;

	HWLOC__BITMAP_CHECK(set1);
	HWLOC__BITMAP_CHECK(set2);

	for(i=0; i<max_count; i++) {
	  unsigned long val1 = HWLOC_SUBBITMAP_READULONG(set1, (unsigned) i);
	  unsigned long val2 = HWLOC_SUBBITMAP_READULONG(set2, (unsigned) i);

	  if (!val1) {
	    if (!val2)
	      /* both empty, no change */
	      continue;

	    /* val1 empty, val2 not */
	    if (result == HWLOC_BITMAP_CONTAINS) {
	      if (!empty2)
		return HWLOC_BITMAP_INTERSECTS;
	      result = HWLOC_BITMAP_DIFFERENT;
	    } else if (result == HWLOC_BITMAP_EQUAL) {
	      result = HWLOC_BITMAP_INCLUDED;
	    }
	    /* no change otherwise */

	  } else if (!val2) {
	    /* val2 empty, val1 not */
	    if (result == HWLOC_BITMAP_INCLUDED) {
	      if (!empty1)
		return HWLOC_BITMAP_INTERSECTS;
	      result = HWLOC_BITMAP_DIFFERENT;
	    } else if (result == HWLOC_BITMAP_EQUAL) {
	      result = HWLOC_BITMAP_CONTAINS;
	    }
	    /* no change otherwise */

	  } else if (val1 == val2) {
	    /* equal and not empty */
	    if (result == HWLOC_BITMAP_DIFFERENT)
	      return HWLOC_BITMAP_INTERSECTS;
	    /* equal/contains/included unchanged */

	  } else if ((val1 & val2) == val1) {
	    /* included and not empty */
	    if (result == HWLOC_BITMAP_CONTAINS || result == HWLOC_BITMAP_DIFFERENT)
	      return HWLOC_BITMAP_INTERSECTS;
	    /* equal/included unchanged */
	    result = HWLOC_BITMAP_INCLUDED;

	  } else if ((val1 & val2) == val2) {
	    /* contains and not empty */
	    if (result == HWLOC_BITMAP_INCLUDED || result == HWLOC_BITMAP_DIFFERENT)
	      return HWLOC_BITMAP_INTERSECTS;
	    /* equal/contains unchanged */
	    result = HWLOC_BITMAP_CONTAINS;

	  } else if ((val1 & val2) != 0) {
	    /* intersects and not empty */
	    return HWLOC_BITMAP_INTERSECTS;

	  } else {
	    /* different and not empty */

	    /* equal/included/contains with non-empty sets means intersects */
	    if (result == HWLOC_BITMAP_EQUAL && !empty1 /* implies !empty2 */)
	      return HWLOC_BITMAP_INTERSECTS;
	    if (result == HWLOC_BITMAP_INCLUDED && !empty1)
	      return HWLOC_BITMAP_INTERSECTS;
	    if (result == HWLOC_BITMAP_CONTAINS && !empty2)
	      return HWLOC_BITMAP_INTERSECTS;
	    /* otherwise means different */
	    result = HWLOC_BITMAP_DIFFERENT;
	  }

	  empty1 &= !val1;
	  empty2 &= !val2;
	}

	if (!set1->infinite) {
	  if (set2->infinite) {
	    /* set2 infinite only */
	    if (result == HWLOC_BITMAP_CONTAINS) {
	      if (!empty2)
		return HWLOC_BITMAP_INTERSECTS;
	      result = HWLOC_BITMAP_DIFFERENT;
	    } else if (result == HWLOC_BITMAP_EQUAL) {
	      result = HWLOC_BITMAP_INCLUDED;
	    }
	    /* no change otherwise */
	  }
	} else if (!set2->infinite) {
	  /* set1 infinite only */
	  if (result == HWLOC_BITMAP_INCLUDED) {
	    if (!empty1)
	      return HWLOC_BITMAP_INTERSECTS;
	    result = HWLOC_BITMAP_DIFFERENT;
	  } else if (result == HWLOC_BITMAP_EQUAL) {
	    result = HWLOC_BITMAP_CONTAINS;
	  }
	  /* no change otherwise */
	} else {
	  /* both infinite */
	  if (result == HWLOC_BITMAP_DIFFERENT)
	    return HWLOC_BITMAP_INTERSECTS;
	  /* equal/contains/included unchanged */
	}

	return result;
}
