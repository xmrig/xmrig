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
  EXEMPLARY, DAMAGES, OR OTHER LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
  OR TORT OR OTHERWISE ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
  EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

/*
 * The 256 bit dataset item function of the LoongArch build.
 * See jit_compiler_la64_lasx_static.S.
 */
extern "C" {
	extern void randomx_init_dataset_la64_lasx();
	extern void randomx_init_dataset_la64_lasx_end();
	extern void randomx_calc_dataset_item_la64_lasx();
	extern void randomx_calc_dataset_item_la64_lasx_prefetch();
	extern void randomx_calc_dataset_item_la64_lasx_mix();
	extern void randomx_calc_dataset_item_la64_lasx_store_result();
	extern void randomx_calc_dataset_item_la64_lasx_end();

	/*
	 * The constants table and the lane offsets of the
	 * body, exported so a rig can read the table
	 * back out of the linked image and the
	 * copy the JIT compiler makes
	 * of it.
	 */
	extern void randomx_calc_dataset_item_la64_lasx_constants();
	extern void randomx_calc_dataset_item_la64_lasx_lanes();
}
