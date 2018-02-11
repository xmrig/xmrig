#ifndef __GETOPT_H__

#pragma warning(disable:4996)

#define __GETOPT_H__

#include <crtdefs.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stdio.h>
#include <windows.h>

#ifdef __cplusplus
extern "C" {
#endif

#define	REPLACE_GETOPT		

#ifdef REPLACE_GETOPT
int	opterr = 1;		
int	optind = 1;		
int	optopt = '?';		
#undef	optreset		
#define	optreset		__mingw_optreset
int	optreset;		
char    *optarg;		
#endif

#define PRINT_ERROR	((opterr) && (*options != ':'))

#define FLAG_PERMUTE	0x01	
#define FLAG_ALLARGS	0x02	
#define FLAG_LONGONLY	0x04	


#define	BADCH		(int)'?'
#define	BADARG		((*options == ':') ? (int)':' : (int)'?')
#define	INORDER 	(int)1

#ifndef __CYGWIN__
#define __progname __argv[0]
#else
extern char __declspec(dllimport) *__progname;
#endif

#ifdef __CYGWIN__
static char EMSG[] = "";
#else
#define	EMSG		""
#endif

static int getopt_internal(int, char * const *, const char *,
			   const struct option *, int *, int);
static int parse_long_options(char * const *, const char *,
			      const struct option *, int *, int);
static int gcd(int, int);
static void permute_args(int, int, int, char * const *);

static char *place = EMSG; 


static int nonopt_start = -1; 
static int nonopt_end = -1;   


static const char recargchar[] = "option requires an argument -- %c";
static const char recargstring[] = "option requires an argument -- %s";
static const char ambig[] = "ambiguous option -- %.*s";
static const char noarg[] = "option doesn't take an argument -- %.*s";
static const char illoptchar[] = "unknown option -- %c";
static const char illoptstring[] = "unknown option -- %s";

static void
_vwarnx(const char *fmt,va_list ap)
{
  (void)fprintf(stderr,"%s: ",__progname);
  if (fmt != NULL)
    (void)vfprintf(stderr,fmt,ap);
  (void)fprintf(stderr,"\n");
}

static void
warnx(const char *fmt,...)
{
  va_list ap;
  va_start(ap,fmt);
  _vwarnx(fmt,ap);
  va_end(ap);
}

static int
gcd(int a, int b)
{
	int c;

	c = a % b;
	while (c != 0) {
		a = b;
		b = c;
		c = a % b;
	}

	return (b);
}


static void
permute_args(int panonopt_start, int panonopt_end, int opt_end,
	char * const *nargv)
{
	int cstart, cyclelen, i, j, ncycle, nnonopts, nopts, pos;
	char *swap;

	nnonopts = panonopt_end - panonopt_start;
	nopts = opt_end - panonopt_end;
	ncycle = gcd(nnonopts, nopts);
	cyclelen = (opt_end - panonopt_start);

	for (i = 0; i < ncycle; i++) {
		cstart = panonopt_end+i;
		pos = cstart;
		for (j = 0; j < cyclelen; j++) {
			if (pos >= panonopt_end)
				pos -= nnonopts;
			else
				pos += nopts;
			swap = nargv[pos];
			
			((char **) nargv)[pos] = nargv[cstart];
			
			((char **)nargv)[cstart] = swap;
		}
	}
}

#ifdef REPLACE_GETOPT
int
getopt(int nargc, char * const *nargv, const char *options)
{
	return (getopt_internal(nargc, nargv, options, NULL, NULL, 0));
}
#endif 

#ifdef _BSD_SOURCE
# define optreset  __mingw_optreset
extern int optreset;
#endif
#ifdef __cplusplus
}
#endif
#endif 

#if !defined(__UNISTD_H_SOURCED__) && !defined(__GETOPT_LONG_H__)
#define __GETOPT_LONG_H__

#ifdef __cplusplus
extern "C" {
#endif

struct option		
{
  const char *name;		
  int         has_arg;		
  int        *flag;		
  int         val;		
};

enum    		
{
  no_argument = 0,      	
  required_argument,		
  optional_argument		
};
static int
parse_long_options(char * const *nargv, const char *options,
	const struct option *long_options, int *idx, int short_too)
{
	char *current_argv, *has_equal;
	size_t current_argv_len;
	int i, ambiguous, match;

#define IDENTICAL_INTERPRETATION(_x, _y)                                \
	(long_options[(_x)].has_arg == long_options[(_y)].has_arg &&    \
	 long_options[(_x)].flag == long_options[(_y)].flag &&          \
	 long_options[(_x)].val == long_options[(_y)].val)

	current_argv = place;
	match = -1;
	ambiguous = 0;

	optind++;

	if ((has_equal = strchr(current_argv, '=')) != NULL) {
		
		current_argv_len = has_equal - current_argv;
		has_equal++;
	} else
		current_argv_len = strlen(current_argv);

	for (i = 0; long_options[i].name; i++) {
		
		if (strncmp(current_argv, long_options[i].name,
		    current_argv_len))
			continue;

		if (strlen(long_options[i].name) == current_argv_len) {
			
			match = i;
			ambiguous = 0;
			break;
		}
		if (short_too && current_argv_len == 1)
			continue;

		if (match == -1)	
			match = i;
		else if (!IDENTICAL_INTERPRETATION(i, match))
			ambiguous = 1;
	}
	if (ambiguous) {
		
		if (PRINT_ERROR)
			warnx(ambig, (int)current_argv_len,
			     current_argv);
		optopt = 0;
		return (BADCH);
	}
	if (match != -1) {		
		if (long_options[match].has_arg == no_argument
		    && has_equal) {
			if (PRINT_ERROR)
				warnx(noarg, (int)current_argv_len,
				     current_argv);
			if (long_options[match].flag == NULL)
				optopt = long_options[match].val;
			else
				optopt = 0;
			return (BADARG);
		}
		if (long_options[match].has_arg == required_argument ||
		    long_options[match].has_arg == optional_argument) {
			if (has_equal)
				optarg = has_equal;
			else if (long_options[match].has_arg ==
			    required_argument) {
				optarg = nargv[optind++];
			}
		}
		if ((long_options[match].has_arg == required_argument)
		    && (optarg == NULL)) {
			if (PRINT_ERROR)
				warnx(recargstring,
				    current_argv);
			if (long_options[match].flag == NULL)
				optopt = long_options[match].val;
			else
				optopt = 0;
			--optind;
			return (BADARG);
		}
	} else {			
		if (short_too) {
			--optind;
			return (-1);
		}
		if (PRINT_ERROR)
			warnx(illoptstring, current_argv);
		optopt = 0;
		return (BADCH);
	}
	if (idx)
		*idx = match;
	if (long_options[match].flag) {
		*long_options[match].flag = long_options[match].val;
		return (0);
	} else
		return (long_options[match].val);
#undef IDENTICAL_INTERPRETATION
}
static int
getopt_internal(int nargc, char * const *nargv, const char *options,
	const struct option *long_options, int *idx, int flags)
{
	char *oli;				
	int optchar, short_too;
	static int posixly_correct = -1;

	if (options == NULL)
		return (-1);

	if (optind == 0)
		optind = optreset = 1;
	if (posixly_correct == -1 || optreset != 0)
		posixly_correct = (getenv("POSIXLY_CORRECT") != NULL);
	if (*options == '-')
		flags |= FLAG_ALLARGS;
	else if (posixly_correct || *options == '+')
		flags &= ~FLAG_PERMUTE;
	if (*options == '+' || *options == '-')
		options++;

	optarg = NULL;
	if (optreset)
		nonopt_start = nonopt_end = -1;
start:
	if (optreset || !*place) {		
		optreset = 0;
		if (optind >= nargc) {          
			place = EMSG;
			if (nonopt_end != -1) {
				
				permute_args(nonopt_start, nonopt_end,
				    optind, nargv);
				optind -= nonopt_end - nonopt_start;
			}
			else if (nonopt_start != -1) {
				optind = nonopt_start;
			}
			nonopt_start = nonopt_end = -1;
			return (-1);
		}
		if (*(place = nargv[optind]) != '-' ||
		    (place[1] == '\0' && strchr(options, '-') == NULL)) {
			place = EMSG;		
			if (flags & FLAG_ALLARGS) {
				optarg = nargv[optind++];
				return (INORDER);
			}
			if (!(flags & FLAG_PERMUTE)) {
				return (-1);
			}
			
			if (nonopt_start == -1)
				nonopt_start = optind;
			else if (nonopt_end != -1) {
				permute_args(nonopt_start, nonopt_end,
				    optind, nargv);
				nonopt_start = optind -
				    (nonopt_end - nonopt_start);
				nonopt_end = -1;
			}
			optind++;
			
			goto start;
		}
		if (nonopt_start != -1 && nonopt_end == -1)
			nonopt_end = optind;
		if (place[1] != '\0' && *++place == '-' && place[1] == '\0') {
			optind++;
			place = EMSG;
			if (nonopt_end != -1) {
				permute_args(nonopt_start, nonopt_end,
				    optind, nargv);
				optind -= nonopt_end - nonopt_start;
			}
			nonopt_start = nonopt_end = -1;
			return (-1);
		}
	}
	if (long_options != NULL && place != nargv[optind] &&
	    (*place == '-' || (flags & FLAG_LONGONLY))) {
		short_too = 0;
		if (*place == '-')
			place++;		
		else if (*place != ':' && strchr(options, *place) != NULL)
			short_too = 1;		

		optchar = parse_long_options(nargv, options, long_options,
		    idx, short_too);
		if (optchar != -1) {
			place = EMSG;
			return (optchar);
		}
	}

	if ((optchar = (int)*place++) == (int)':' ||
	    (optchar == (int)'-' && *place != '\0') ||
	    (oli = (char*)strchr(options, optchar)) == NULL) {
		if (optchar == (int)'-' && *place == '\0')
			return (-1);
		if (!*place)
			++optind;
		if (PRINT_ERROR)
			warnx(illoptchar, optchar);
		optopt = optchar;
		return (BADCH);
	}
	if (long_options != NULL && optchar == 'W' && oli[1] == ';') {
		
		if (*place) {}
		else if (++optind >= nargc) {	
			place = EMSG;
			if (PRINT_ERROR)
				warnx(recargchar, optchar);
			optopt = optchar;
			return (BADARG);
		} else				
			place = nargv[optind];
		optchar = parse_long_options(nargv, options, long_options,
		    idx, 0);
		place = EMSG;
		return (optchar);
	}
	if (*++oli != ':') {			
		if (!*place)
			++optind;
	} else {				
		optarg = NULL;
		if (*place)			
			optarg = place;
		else if (oli[1] != ':') {	
			if (++optind >= nargc) {	
				place = EMSG;
				if (PRINT_ERROR)
					warnx(recargchar, optchar);
				optopt = optchar;
				return (BADARG);
			} else
				optarg = nargv[optind];
		}
		place = EMSG;
		++optind;
	}
	
	return (optchar);
}

int
getopt_long(int nargc, char * const *nargv, const char *options,
    const struct option *long_options, int *idx)
{

	return (getopt_internal(nargc, nargv, options, long_options, idx,
	    FLAG_PERMUTE));
}

int
getopt_long_only(int nargc, char * const *nargv, const char *options,
    const struct option *long_options, int *idx)
{

	return (getopt_internal(nargc, nargv, options, long_options, idx,
	    FLAG_PERMUTE|FLAG_LONGONLY));
}

#ifndef HAVE_DECL_GETOPT
# define HAVE_DECL_GETOPT	1
#endif

#ifdef __cplusplus
}
#endif

#endif 
