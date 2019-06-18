#include <string>
#include <iostream>
#include "utility.hpp"
#include "../randomx.h"
#include "../virtual_machine.hpp"
#include "../blake2/endian.h"

/*
	Writes final scratchpads to disk as files with .spad extension, each file is 2048 KiB.
	Command line parameters:
		--count N         number of files to generate (default = 1)
		--seed S          different seed will give different outputs (default = 0)

	Entropy can be estimated by compressing the files using 7zip in Ultra mode:

	    7z.exe a -t7z -m0=lzma2 -mx=9 scratchpads.7z *.spad
*/

int main(int argc, char** argv) {
	int count, seedValue;

	readIntOption("--count", argc, argv, count, 1);
	readIntOption("--seed", argc, argv, seedValue, 0);

	std::cout << "Generating " << count << " scratchpad(s) using seed " << seedValue << " ..." << std::endl;

	char seed[4];
	char input[4];
	char hash[RANDOMX_HASH_SIZE];

	store32(&seed, seedValue);

	randomx_cache *cache = randomx_alloc_cache(RANDOMX_FLAG_DEFAULT);
	randomx_init_cache(cache, &seed, sizeof seed);
	randomx_vm *vm = randomx_create_vm(RANDOMX_FLAG_DEFAULT, cache, NULL);

	for (int i = 0; i < count; ++i) {
		store32(&input, i);
		randomx_calculate_hash(vm, &input, sizeof input, hash);
		std::string filename("test-");
		filename += std::to_string(i);
		filename += ".spad";
		dump((const char*)vm->getScratchpad(), randomx::ScratchpadSize, filename.c_str());
	}

	randomx_destroy_vm(vm);
	randomx_release_cache(cache);

	return 0;
}
