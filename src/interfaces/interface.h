/* XMRig - enWILLYado
 *
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program. If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef __INTERFACE_H__
#define __INTERFACE_H__

#ifdef _WIN32
#if defined(_MSC_VER) && _MSC_VER < 1900

// C++-11
#define override

// VS
#include <vadefs.h>
#include <stdio.h>
#include <stdarg.h>
#define snprintf c99_snprintf
#define vsnprintf c99_vsnprintf

__inline int c99_vsnprintf(char* outBuf, size_t size, const char* format, va_list ap)
{
	int count = -1;

	if(size != 0)
	{
		count = _vsnprintf_s(outBuf, size, _TRUNCATE, format, ap);
	}
	if(count == -1)
	{
		count = _vscprintf(format, ap);
	}

	return count;
}

__inline int c99_snprintf(char* outBuf, size_t size, const char* format, ...)
{
	int count;
	va_list ap;

	va_start(ap, format);
	count = c99_vsnprintf(outBuf, size, format, ap);
	va_end(ap);

	return count;
}
#endif
#endif

#endif // __INTERFACE_H__
