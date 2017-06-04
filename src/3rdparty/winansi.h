/*
 * ANSI emulation wrappers
 */

#include <windows.h>
#include <stddef.h>
#include <stdio.h>

#define fileno(fd) _fileno(fd)

#ifdef __cplusplus
extern "C" {
#endif
    int winansi_fputs(const char *str, FILE *stream);
    int winansi_printf(const char *format, ...);
    int winansi_fprintf(FILE *stream, const char *format, ...);
    int winansi_vfprintf(FILE *stream, const char *format, va_list list);
#ifdef __cplusplus
}
#endif

#undef fputs
#undef fprintf
#undef vfprintf

#define fputs winansi_fputs
#define printf winansi_printf
#define fprintf winansi_fprintf
#define vfprintf winansi_vfprintf
