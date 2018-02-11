#ifndef __ALIGN_H__
#define __ALIGN_H__

#ifdef _MSC_VER
#   define VAR_ALIGN(x, decl) __declspec(align(x)) decl
#else
#   define VAR_ALIGN(x, decl) decl __attribute__ ((aligned(x)))
#endif

#endif /* __ALIGN_H__ */
