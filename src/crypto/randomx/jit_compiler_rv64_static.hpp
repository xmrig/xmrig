/*
Copyright (c) 2023 tevador <tevador@gmail.com>

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
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

extern "C" {
	void randomx_riscv64_literals();
	void randomx_riscv64_literals_end();
	void randomx_riscv64_data_init();
	void randomx_riscv64_fix_data_call();
	void randomx_riscv64_prologue();
	void randomx_riscv64_loop_begin();
	void randomx_riscv64_data_read();
	void randomx_riscv64_data_read_light();
	void randomx_riscv64_fix_loop_call();
	void randomx_riscv64_spad_store();
	void randomx_riscv64_spad_store_hardaes();
	void randomx_riscv64_spad_store_softaes();
	void randomx_riscv64_loop_end();
	void randomx_riscv64_fix_continue_loop();
	void randomx_riscv64_epilogue();
	void randomx_riscv64_softaes();
	void randomx_riscv64_program_end();
	void randomx_riscv64_ssh_init();
	void randomx_riscv64_ssh_load();
	void randomx_riscv64_ssh_prefetch();
	void randomx_riscv64_ssh_end();
}
