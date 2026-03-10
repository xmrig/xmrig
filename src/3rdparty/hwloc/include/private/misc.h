/*
 * Copyright © 2009 CNRS
 * Copyright © 2009-2024 Inria.  All rights reserved.
 * Copyright © 2009-2012 Université Bordeaux
 * Copyright © 2011 Cisco Systems, Inc.  All rights reserved.
 * See COPYING in top-level directory.
 */

/* Misc macros and inlines.  */

#ifndef HWLOC_PRIVATE_MISC_H
#define HWLOC_PRIVATE_MISC_H

#include "hwloc/autogen/config.h"
#include "private/autogen/config.h"
#include "hwloc.h"

#ifdef HWLOC_HAVE_DECL_STRNCASECMP
#ifdef HAVE_STRINGS_H
#include <strings.h>
#endif
#else
#ifdef HAVE_CTYPE_H
#include <ctype.h>
#endif
#endif

#define HWLOC_BITS_PER_LONG (HWLOC_SIZEOF_UNSIGNED_LONG * 8)
#define HWLOC_BITS_PER_INT (HWLOC_SIZEOF_UNSIGNED_INT * 8)

#if (HWLOC_BITS_PER_LONG != 32) && (HWLOC_BITS_PER_LONG != 64)
#error "unknown size for unsigned long."
#endif

#if (HWLOC_BITS_PER_INT != 16) && (HWLOC_BITS_PER_INT != 32) && (HWLOC_BITS_PER_INT != 64)
#error "unknown size for unsigned int."
#endif

/* internal-use-only value for when we don't know the type or don't have any value */
#define HWLOC_OBJ_TYPE_NONE ((hwloc_obj_type_t) -1)

/**
 * ffsl helpers.
 */

#if defined(HWLOC_HAVE_BROKEN_FFS)

/* System has a broken ffs().
 * We must check the before __GNUC__ or HWLOC_HAVE_FFSL
 */
#    define HWLOC_NO_FFS

#elif defined(__GNUC__)

#  if (__GNUC__ >= 4) || ((__GNUC__ == 3) && (__GNUC_MINOR__ >= 4))
     /* Starting from 3.4, gcc has a long variant.  */
#    define hwloc_ffsl(x) __builtin_ffsl(x)
#  else
#    define hwloc_ffs(x) __builtin_ffs(x)
#    define HWLOC_NEED_FFSL
#  endif

#elif defined(HWLOC_HAVE_FFSL)

#  ifndef HWLOC_HAVE_DECL_FFSL
extern int ffsl(long) __hwloc_attribute_const;
#  endif

#  define hwloc_ffsl(x) ffsl(x)

#elif defined(HWLOC_HAVE_FFS)

#  ifndef HWLOC_HAVE_DECL_FFS
extern int ffs(int) __hwloc_attribute_const;
#  endif

#  define hwloc_ffs(x) ffs(x)
#  define HWLOC_NEED_FFSL

#else /* no ffs implementation */

#    define HWLOC_NO_FFS

#endif

#ifdef HWLOC_NO_FFS

/* no ffs or it is known to be broken */
static __hwloc_inline int
hwloc_ffsl_manual(unsigned long x) __hwloc_attribute_const;
static __hwloc_inline int
hwloc_ffsl_manual(unsigned long x)
{
	int i;

	if (!x)
		return 0;

	i = 1;
#if HWLOC_BITS_PER_LONG >= 64
	if (!(x & 0xfffffffful)) {
		x >>= 32;
		i += 32;
	}
#endif
	if (!(x & 0xffffu)) {
		x >>= 16;
		i += 16;
	}
	if (!(x & 0xff)) {
		x >>= 8;
		i += 8;
	}
	if (!(x & 0xf)) {
		x >>= 4;
		i += 4;
	}
	if (!(x & 0x3)) {
		x >>= 2;
		i += 2;
	}
	if (!(x & 0x1)) {
		x >>= 1;
		i += 1;
	}

	return i;
}
/* always define hwloc_ffsl as a macro, to avoid renaming breakage */
#define hwloc_ffsl hwloc_ffsl_manual

#elif defined(HWLOC_NEED_FFSL)

/* We only have an int ffs(int) implementation, build a long one.  */

/* First make it 32 bits if it was only 16.  */
static __hwloc_inline int
hwloc_ffs32(unsigned long x) __hwloc_attribute_const;
static __hwloc_inline int
hwloc_ffs32(unsigned long x)
{
#if HWLOC_BITS_PER_INT == 16
	int low_ffs, hi_ffs;

	low_ffs = hwloc_ffs(x & 0xfffful);
	if (low_ffs)
		return low_ffs;

	hi_ffs = hwloc_ffs(x >> 16);
	if (hi_ffs)
		return hi_ffs + 16;

	return 0;
#else
	return hwloc_ffs(x);
#endif
}

/* Then make it 64 bit if longs are.  */
static __hwloc_inline int
hwloc_ffsl_from_ffs32(unsigned long x) __hwloc_attribute_const;
static __hwloc_inline int
hwloc_ffsl_from_ffs32(unsigned long x)
{
#if HWLOC_BITS_PER_LONG == 64
	int low_ffs, hi_ffs;

	low_ffs = hwloc_ffs32(x & 0xfffffffful);
	if (low_ffs)
		return low_ffs;

	hi_ffs = hwloc_ffs32(x >> 32);
	if (hi_ffs)
		return hi_ffs + 32;

	return 0;
#else
	return hwloc_ffs32(x);
#endif
}
/* always define hwloc_ffsl as a macro, to avoid renaming breakage */
#define hwloc_ffsl hwloc_ffsl_from_ffs32

#endif

/**
 * flsl helpers.
 */
#ifdef __GNUC__

#  if (__GNUC__ >= 4) || ((__GNUC__ == 3) && (__GNUC_MINOR__ >= 4))
#    define hwloc_flsl(x) ((x) ? (8*sizeof(long) - __builtin_clzl(x)) : 0)
#  else
#    define hwloc_fls(x) ((x) ? (8*sizeof(int) - __builtin_clz(x)) : 0)
#    define HWLOC_NEED_FLSL
#  endif

#elif defined(HWLOC_HAVE_FLSL)

#  ifndef HWLOC_HAVE_DECL_FLSL
extern int flsl(long) __hwloc_attribute_const;
#  endif

#  define hwloc_flsl(x) flsl(x)

#elif defined(HWLOC_HAVE_CLZL)

#  ifndef HWLOC_HAVE_DECL_CLZL
extern int clzl(long) __hwloc_attribute_const;
#  endif

#  define hwloc_flsl(x) ((x) ? (8*sizeof(long) - clzl(x)) : 0)

#elif defined(HWLOC_HAVE_FLS)

#  ifndef HWLOC_HAVE_DECL_FLS
extern int fls(int) __hwloc_attribute_const;
#  endif

#  define hwloc_fls(x) fls(x)
#  define HWLOC_NEED_FLSL

#elif defined(HWLOC_HAVE_CLZ)

#  ifndef HWLOC_HAVE_DECL_CLZ
extern int clz(int) __hwloc_attribute_const;
#  endif

#  define hwloc_fls(x) ((x) ? (8*sizeof(int) - clz(x)) : 0)
#  define HWLOC_NEED_FLSL

#else /* no fls implementation */

static __hwloc_inline int
hwloc_flsl_manual(unsigned long x) __hwloc_attribute_const;
static __hwloc_inline int
hwloc_flsl_manual(unsigned long x)
{
	int i = 0;

	if (!x)
		return 0;

	i = 1;
#if HWLOC_BITS_PER_LONG >= 64
	if ((x & 0xffffffff00000000ul)) {
		x >>= 32;
		i += 32;
	}
#endif
	if ((x & 0xffff0000u)) {
		x >>= 16;
		i += 16;
	}
	if ((x & 0xff00)) {
		x >>= 8;
		i += 8;
	}
	if ((x & 0xf0)) {
		x >>= 4;
		i += 4;
	}
	if ((x & 0xc)) {
		x >>= 2;
		i += 2;
	}
	if ((x & 0x2)) {
		x >>= 1;
		i += 1;
	}

	return i;
}
/* always define hwloc_flsl as a macro, to avoid renaming breakage */
#define hwloc_flsl hwloc_flsl_manual

#endif

#ifdef HWLOC_NEED_FLSL

/* We only have an int fls(int) implementation, build a long one.  */

/* First make it 32 bits if it was only 16.  */
static __hwloc_inline int
hwloc_fls32(unsigned long x) __hwloc_attribute_const;
static __hwloc_inline int
hwloc_fls32(unsigned long x)
{
#if HWLOC_BITS_PER_INT == 16
	int low_fls, hi_fls;

	hi_fls = hwloc_fls(x >> 16);
	if (hi_fls)
		return hi_fls + 16;

	low_fls = hwloc_fls(x & 0xfffful);
	if (low_fls)
		return low_fls;

	return 0;
#else
	return hwloc_fls(x);
#endif
}

/* Then make it 64 bit if longs are.  */
static __hwloc_inline int
hwloc_flsl_from_fls32(unsigned long x) __hwloc_attribute_const;
static __hwloc_inline int
hwloc_flsl_from_fls32(unsigned long x)
{
#if HWLOC_BITS_PER_LONG == 64
	int low_fls, hi_fls;

	hi_fls = hwloc_fls32(x >> 32);
	if (hi_fls)
		return hi_fls + 32;

	low_fls = hwloc_fls32(x & 0xfffffffful);
	if (low_fls)
		return low_fls;

	return 0;
#else
	return hwloc_fls32(x);
#endif
}
/* always define hwloc_flsl as a macro, to avoid renaming breakage */
#define hwloc_flsl hwloc_flsl_from_fls32

#endif

static __hwloc_inline int
hwloc_weight_long(unsigned long w) __hwloc_attribute_const;
static __hwloc_inline int
hwloc_weight_long(unsigned long w)
{
#if HWLOC_BITS_PER_LONG == 32
#if (__GNUC__ >= 4) || ((__GNUC__ == 3) && (__GNUC_MINOR__) >= 4)
	return __builtin_popcount(w);
#else
	unsigned int res = (w & 0x55555555) + ((w >> 1) & 0x55555555);
	res = (res & 0x33333333) + ((res >> 2) & 0x33333333);
	res = (res & 0x0F0F0F0F) + ((res >> 4) & 0x0F0F0F0F);
	res = (res & 0x00FF00FF) + ((res >> 8) & 0x00FF00FF);
	return (res & 0x0000FFFF) + ((res >> 16) & 0x0000FFFF);
#endif
#else /* HWLOC_BITS_PER_LONG == 32 */
#if (__GNUC__ >= 4) || ((__GNUC__ == 3) && (__GNUC_MINOR__) >= 4)
	return __builtin_popcountll(w);
#else
	unsigned long res;
	res = (w & 0x5555555555555555ul) + ((w >> 1) & 0x5555555555555555ul);
	res = (res & 0x3333333333333333ul) + ((res >> 2) & 0x3333333333333333ul);
	res = (res & 0x0F0F0F0F0F0F0F0Ful) + ((res >> 4) & 0x0F0F0F0F0F0F0F0Ful);
	res = (res & 0x00FF00FF00FF00FFul) + ((res >> 8) & 0x00FF00FF00FF00FFul);
	res = (res & 0x0000FFFF0000FFFFul) + ((res >> 16) & 0x0000FFFF0000FFFFul);
	return (res & 0x00000000FFFFFFFFul) + ((res >> 32) & 0x00000000FFFFFFFFul);
#endif
#endif /* HWLOC_BITS_PER_LONG == 64 */
}

#if !HAVE_DECL_STRTOULL && defined(HAVE_STRTOULL)
unsigned long long int strtoull(const char *nptr, char **endptr, int base);
#endif

static __hwloc_inline int hwloc_strncasecmp(const char *s1, const char *s2, size_t n)
{
#ifdef HWLOC_HAVE_DECL_STRNCASECMP
  return strncasecmp(s1, s2, n);
#else
  while (n) {
    char c1 = tolower(*s1), c2 = tolower(*s2);
    if (!c1 || !c2 || c1 != c2)
      return c1-c2;
    n--; s1++; s2++;
  }
  return 0;
#endif
}

static __hwloc_inline hwloc_obj_type_t hwloc_cache_type_by_depth_type(unsigned depth, hwloc_obj_cache_type_t type)
{
  if (type == HWLOC_OBJ_CACHE_INSTRUCTION) {
    if (depth >= 1 && depth <= 3)
      return HWLOC_OBJ_L1ICACHE + depth-1;
    else
      return HWLOC_OBJ_TYPE_NONE;
  } else {
    if (depth >= 1 && depth <= 5)
      return HWLOC_OBJ_L1CACHE + depth-1;
    else
      return HWLOC_OBJ_TYPE_NONE;
  }
}

#define HWLOC_BITMAP_EQUAL 0       /* Bitmaps are equal */
#define HWLOC_BITMAP_INCLUDED 1    /* First bitmap included in second */
#define HWLOC_BITMAP_CONTAINS 2    /* First bitmap contains second */
#define HWLOC_BITMAP_INTERSECTS 3  /* Bitmaps intersect without any inclusion */
#define HWLOC_BITMAP_DIFFERENT  4  /* Bitmaps do not intersect */

/* Compare bitmaps \p bitmap1 and \p bitmap2 from an inclusion point of view. */
HWLOC_DECLSPEC int hwloc_bitmap_compare_inclusion(hwloc_const_bitmap_t bitmap1, hwloc_const_bitmap_t bitmap2) __hwloc_attribute_pure;

/* Return a stringified PCI class. */
HWLOC_DECLSPEC extern const char * hwloc_pci_class_string(unsigned short class_id);

/* Parse a PCI link speed (GT/s) string from Linux sysfs */
#ifdef HWLOC_LINUX_SYS
#include <stdlib.h> /* for atof() */
static __hwloc_inline float
hwloc_linux_pci_link_speed_from_string(const char *string)
{
  /* don't parse Gen1 with atof() since it expects a localized string
   * while the kernel sysfs files aren't.
   */
  if (!strncmp(string, "2.5 ", 4))
    /* "2.5 GT/s" is Gen1 with 8/10 encoding */
    return 2.5 * .8;

  /* also hardwire Gen2 since it also has a specific encoding */
  if (!strncmp(string, "5 ", 2))
    /* "5 GT/s" is Gen2 with 8/10 encoding */
    return 5 * .8;

  /* handle Gen3+ in a generic way */
  return atof(string) * 128./130; /* Gen3+ encoding is 128/130 */
}
#endif

/* Traverse children of a parent */
#define for_each_child(child, parent) for(child = parent->first_child; child; child = child->next_sibling)
#define for_each_memory_child(child, parent) for(child = parent->memory_first_child; child; child = child->next_sibling)
#define for_each_io_child(child, parent) for(child = parent->io_first_child; child; child = child->next_sibling)
#define for_each_misc_child(child, parent) for(child = parent->misc_first_child; child; child = child->next_sibling)

/* Any object attached to normal children */
static __hwloc_inline int hwloc__obj_type_is_normal (hwloc_obj_type_t type)
{
  /* type contiguity is asserted in topology_check() */
  return type <= HWLOC_OBJ_GROUP || type == HWLOC_OBJ_DIE;
}

/* Any object attached to memory children, currently NUMA nodes or Memory-side caches */
static __hwloc_inline int hwloc__obj_type_is_memory (hwloc_obj_type_t type)
{
  /* type contiguity is asserted in topology_check() */
  return type == HWLOC_OBJ_NUMANODE || type == HWLOC_OBJ_MEMCACHE;
}

/* I/O or Misc object, without cpusets or nodesets. */
static __hwloc_inline int hwloc__obj_type_is_special (hwloc_obj_type_t type)
{
  /* type contiguity is asserted in topology_check() */
  return type >= HWLOC_OBJ_BRIDGE && type <= HWLOC_OBJ_MISC;
}

/* Any object attached to io children */
static __hwloc_inline int hwloc__obj_type_is_io (hwloc_obj_type_t type)
{
  /* type contiguity is asserted in topology_check() */
  return type >= HWLOC_OBJ_BRIDGE && type <= HWLOC_OBJ_OS_DEVICE;
}

/* Any CPU caches (not Memory-side caches) */
static __hwloc_inline int
hwloc__obj_type_is_cache(hwloc_obj_type_t type)
{
  /* type contiguity is asserted in topology_check() */
  return (type >= HWLOC_OBJ_L1CACHE && type <= HWLOC_OBJ_L3ICACHE);
}

static __hwloc_inline int
hwloc__obj_type_is_dcache(hwloc_obj_type_t type)
{
  /* type contiguity is asserted in topology_check() */
  return (type >= HWLOC_OBJ_L1CACHE && type <= HWLOC_OBJ_L5CACHE);
}

/** \brief Check whether an object is a Instruction Cache. */
static __hwloc_inline int
hwloc__obj_type_is_icache(hwloc_obj_type_t type)
{
  /* type contiguity is asserted in topology_check() */
  return (type >= HWLOC_OBJ_L1ICACHE && type <= HWLOC_OBJ_L3ICACHE);
}

#ifdef HAVE_USELOCALE
#include "locale.h"
#ifdef HAVE_XLOCALE_H
#include "xlocale.h"
#endif
#define hwloc_localeswitch_declare locale_t __old_locale = (locale_t)0, __new_locale
#define hwloc_localeswitch_init() do {                     \
  __new_locale = newlocale(LC_ALL_MASK, "C", (locale_t)0); \
  if (__new_locale != (locale_t)0)                         \
    __old_locale = uselocale(__new_locale);                \
} while (0)
#define hwloc_localeswitch_fini() do { \
  if (__new_locale != (locale_t)0) {   \
    uselocale(__old_locale);           \
    freelocale(__new_locale);          \
  }                                    \
} while(0)
#else /* HAVE_USELOCALE */
#if HWLOC_HAVE_ATTRIBUTE_UNUSED
#define hwloc_localeswitch_declare int __dummy_nolocale __hwloc_attribute_unused
#define hwloc_localeswitch_init()
#else
#define hwloc_localeswitch_declare int __dummy_nolocale
#define hwloc_localeswitch_init() (void)__dummy_nolocale
#endif
#define hwloc_localeswitch_fini()
#endif /* HAVE_USELOCALE */

#if !HAVE_DECL_FABSF
#define fabsf(f) fabs((double)(f))
#endif

#if !HAVE_DECL_MODFF
#define modff(x,iptr) (float)modf((double)x,(double *)iptr)
#endif

#if HAVE_DECL__SC_PAGE_SIZE
#define hwloc_getpagesize() sysconf(_SC_PAGE_SIZE)
#elif HAVE_DECL__SC_PAGESIZE
#define hwloc_getpagesize() sysconf(_SC_PAGESIZE)
#elif defined HAVE_GETPAGESIZE
#define hwloc_getpagesize() getpagesize()
#else
#undef hwloc_getpagesize
#endif

#if HWLOC_HAVE_ATTRIBUTE_FORMAT
#  define __hwloc_attribute_format(type, str, arg)  __attribute__((__format__(type, str, arg)))
#else
#  define __hwloc_attribute_format(type, str, arg)
#endif

#define hwloc_memory_size_printf_value(_size, _verbose) \
  ((_size) < (10ULL<<20) || (_verbose) ? (((_size)>>9)+1)>>1 : (_size) < (10ULL<<30) ? (((_size)>>19)+1)>>1 : (_size) < (10ULL<<40) ? (((_size)>>29)+1)>>1 : (((_size)>>39)+1)>>1)
#define hwloc_memory_size_printf_unit(_size, _verbose) \
  ((_size) < (10ULL<<20) || (_verbose) ? "KB" : (_size) < (10ULL<<30) ? "MB" : (_size) < (10ULL<<40) ? "GB" : "TB")

#ifdef HWLOC_WIN_SYS
#  ifndef HAVE_SSIZE_T
typedef SSIZE_T ssize_t;
#  endif
#  if !HAVE_DECL_STRTOULL && !defined(HAVE_STRTOULL)
#    define strtoull _strtoui64
#  endif
#  ifndef S_ISREG
#    define S_ISREG(m) ((m) & S_IFREG)
#  endif
#  ifndef S_ISDIR
#    define S_ISDIR(m) (((m) & S_IFMT) == S_IFDIR)
#  endif
#  ifndef S_IRWXU
#    define S_IRWXU 00700
#  endif
#  ifndef HWLOC_HAVE_DECL_STRCASECMP
#    define strcasecmp _stricmp
#  endif
#  if !HAVE_DECL_SNPRINTF
#    define snprintf _snprintf
#  endif
#  if HAVE_DECL__STRDUP
#    define strdup _strdup
#  endif
#  if HAVE_DECL__PUTENV
#    define putenv _putenv
#  endif
#endif

static __inline float
hwloc__pci_link_speed(unsigned generation, unsigned lanes)
{
  float lanespeed;
  /*
   * These are single-direction bandwidths only.
   *
   * Gen1 used NRZ with 8/10 encoding.
   * PCIe Gen1 = 2.5GT/s signal-rate per lane x 8/10    =  0.25GB/s data-rate per lane
   * PCIe Gen2 = 5  GT/s signal-rate per lane x 8/10    =  0.5 GB/s data-rate per lane
   * Gen3 switched to NRZ with 128/130 encoding.
   * PCIe Gen3 = 8  GT/s signal-rate per lane x 128/130 =  1   GB/s data-rate per lane
   * PCIe Gen4 = 16 GT/s signal-rate per lane x 128/130 =  2   GB/s data-rate per lane
   * PCIe Gen5 = 32 GT/s signal-rate per lane x 128/130 =  4   GB/s data-rate per lane
   * Gen6 switched to PAM with with 242/256 FLIT (242B payload protected by 8B CRC + 6B FEC).
   * PCIe Gen6 = 64 GT/s signal-rate per lane x 242/256 =  8   GB/s data-rate per lane
   * PCIe Gen7 = 128GT/s signal-rate per lane x 242/256 = 16   GB/s data-rate per lane
   */

  /* lanespeed in Gbit/s */
  if (generation <= 2)
    lanespeed = 2.5f * generation * 0.8f;
  else if (generation <= 5)
    lanespeed = 8.0f * (1<<(generation-3)) * 128/130;
  else
    lanespeed = 8.0f * (1<<(generation-3)) * 242/256; /* assume Gen8 will be 256 GT/s and so on */

  /* linkspeed in GB/s */
  return lanespeed * lanes / 8;
}

#endif /* HWLOC_PRIVATE_MISC_H */
