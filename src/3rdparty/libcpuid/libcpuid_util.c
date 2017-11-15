/*
 * Copyright 2008  Veselin Georgiev,
 * anrieffNOSPAM @ mgail_DOT.com (convert to gmail)
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <ctype.h>
#include "libcpuid.h"
#include "libcpuid_util.h"

void match_features(const struct feature_map_t* matchtable, int count, uint32_t reg, struct cpu_id_t* data)
{
	int i;
	for (i = 0; i < count; i++)
		if (reg & (1u << matchtable[i].bit))
			data->flags[matchtable[i].feature] = 1;
}

static int xmatch_entry(char c, const char* p)
{
	int i, j;
	if (c == 0) return -1;
	if (c == p[0]) return 1;
	if (p[0] == '.') return 1;
	if (p[0] == '#' && isdigit(c)) return 1;
	if (p[0] == '[') {
		j = 1;
		while (p[j] && p[j] != ']') j++;
		if (!p[j]) return -1;
		for (i = 1; i < j; i++)
			if (p[i] == c) return j + 1;
	}
	return -1;
}

int match_pattern(const char* s, const char* p)
{
	int i, j, dj, k, n, m;
	n = (int) strlen(s);
	m = (int) strlen(p);
	for (i = 0; i < n; i++) {
		if (xmatch_entry(s[i], p) != -1) {
			j = 0;
			k = 0;
			while (j < m && ((dj = xmatch_entry(s[i + k], p + j)) != -1)) {
				k++;
				j += dj;
			}
			if (j == m) return i + 1;
		}
	}
	return 0;
}

struct cpu_id_t* get_cached_cpuid(void)
{
	static int initialized = 0;
	static struct cpu_id_t id;
	if (initialized) return &id;
	if (cpu_identify(NULL, &id))
		memset(&id, 0, sizeof(id));
	initialized = 1;
	return &id;
}

int match_all(uint64_t bits, uint64_t mask)
{
	return (bits & mask) == mask;
}
