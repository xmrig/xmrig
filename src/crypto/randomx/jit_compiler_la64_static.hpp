/*
  Copyright (c) 2024, RESBI <https://github.com/RESBI>

  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:
	* Redistributions of source code must retain the above copyright
	  notice, this list of conditions and the following disclaimer.
	* Redistributions in binary form must reproduce the above copyright
	  notice, this list of conditions and the following disclaimer in the
	  documentation and/or other materials provided with the distribution.
	* Neither the name of the copyright holder nor the
	  names of its contributors may be used to endorse or promote products
	  derived from this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY
  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
  THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

extern "C" {
	extern void randomx_program_la64();
	extern void randomx_program_la64_main_loop();
	extern void randomx_program_la64_main_loop_mask0();
	extern void randomx_program_la64_main_loop_mask1();
	extern void randomx_program_la64_vm_instructions();
	extern void randomx_program_la64_vm_instructions_end();
	extern void randomx_program_la64_cacheline_align_mask1();
	extern void randomx_program_la64_cacheline_align_mask2();
	extern void randomx_program_la64_update_spMix1();
	extern void randomx_program_la64_v2_FE_mix();
	extern void randomx_program_la64_v1_FE_mix();
	extern void randomx_program_la64_v2_FE_mix_soft_aes();
	extern void randomx_program_la64_aes_lut_pointers();
	extern void randomx_program_la64_vm_instructions_end_light();
	extern void randomx_program_la64_vm_instructions_end_light_tweak();
	extern void randomx_program_la64_light_cacheline_align_mask();
	extern void randomx_program_la64_light_dataset_offset();
	extern void randomx_program_la64_vm_instructions_end_v1();
	extern void randomx_program_la64_vm_instructions_end_v2();
	extern void randomx_program_la64_vm_instructions_end_light_v1();
	extern void randomx_program_la64_vm_instructions_end_light_v2();
	extern void randomx_init_dataset_la64();
	extern void randomx_init_dataset_la64_end();
	extern void randomx_calc_dataset_item_la64();
	extern void randomx_calc_dataset_item_la64_prefetch();
	extern void randomx_calc_dataset_item_la64_mix();
	extern void randomx_calc_dataset_item_la64_store_result();
	extern void randomx_calc_dataset_item_la64_end();
}
