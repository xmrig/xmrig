/*
  Copyright (c) 2024, RESBI <https://github.com/RESBI>

  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:
	* Redistributions of source code must retain the above
	  copyright notice, this list of conditions and the following
	  disclaimer.
	* Redistributions in binary form must reproduce the above
	  copyright notice, this list of conditions and the following
	  disclaimer in the documentation and/or other materials provided
	  with the distribution.
	* Neither the name of the copyright holder nor the
	  names of its contributors may be used to endorse or promote products
	  derived from this source without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
  ANY EXPRESS OR IMPLIED WARRANTIES ARE DISCLAIMED. IN NO EVENT SHALL THE
  COPYRIGHT HOLDER BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
  EXEMPLARY, DAMAGES, OR OTHER LIABILITY, WHETHER IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include <cstddef>
#include <cstdint>

#include "crypto/randomx/jit_compiler_la64_lasx_static.hpp"

namespace randomx {

	class SuperscalarProgram;

	/*
	 * The 256 bit ( LASX ) half of the LoongArch dataset item function.
	 *
	 * The 128 bit function in jit_compiler_la64.cpp calculates ONE item
	 * per call and holds one item in $vr.  LASX is a wider vector unit of
	 * the very same shape, so one $xr holds FOUR items and the raw item
	 * function walks the dataset four items at a time:
	 *
	 *	the item function of this layer, called with
	 *	  $r4 = the cache memory
	 *	  $r5 = the dataset at the first of the four items
	 *	  $r6 = the first of the four item numbers
	 *	computes all four items.
	 *
	 * Because of that the dataset has to be walked in whole steps of
	 * four, see the split in src/crypto/rx/RxDataset.cpp.  One 256 bit
	 * item function is 24 bytes per superscalar instruction, the
	 * 20 byte budget of the 128 bit layer is not enough.
	 */
	namespace la64_lasx {

		/* The number of items one call of the item function calculates. */
		constexpr size_t ItemsPerCall = 4;

		/* The cache line index mask of the item function. */
		uint64_t getCacheLineMask();

		/*
		 * The upper bound of the bytes of the static blob: the init
		 * loop followed by ONE item function for "accesses" cache
		 * accesses.
		 */
		size_t getBlobSize(size_t accesses);

		/*
		 * Copies the four item wide init loop, whose " bl " already
		 * points at the item function slot right behind it, and
		 * returns its size.
		 */
		size_t copyInitLoop(uint8_t* code);

		/*
		 * Writes the body of the item function at "code" and
		 * returns the bytes it took.
		 */
		size_t generateSuperscalarHash(SuperscalarProgram* programs, size_t accesses, uint8_t* code);

	}

}
