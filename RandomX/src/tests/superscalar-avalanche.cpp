#include <iostream>
#include <cstdint>
#include <vector>
#include "../superscalar.hpp"
#include "../intrin_portable.h"

const uint8_t seed[32] = { 191, 182, 222, 175, 249, 89, 134, 104, 241, 68, 191, 62, 162, 166, 61, 64, 123, 191, 227, 193, 118, 60, 188, 53, 223, 133, 175, 24, 123, 230, 55, 74 };

int main() {

	int insensitiveProgCount[64] = { 0 };
	std::vector<uint64_t> dummy;
	for (int bit = 0; bit < 64; ++bit) {
		for (int i = 0; i < 10000; ++i) {
			uint64_t ra[8] = {
				6364136223846793005ULL,
				9298410992540426748ULL,
				12065312585734608966ULL,
				9306329213124610396ULL,
				5281919268842080866ULL,
				10536153434571861004ULL,
				3398623926847679864ULL,
				9549104520008361294ULL,
			};
			uint64_t rb[8];
			memcpy(rb, ra, sizeof rb);
			rb[0] ^= (1ULL << bit);
			randomx::SuperscalarProgram p;
			randomx::Blake2Generator gen(seed, sizeof seed, i);
			randomx::generateSuperscalar(p, gen);
			randomx::executeSuperscalar(ra, p, nullptr);
			randomx::executeSuperscalar(rb, p, nullptr);
			uint64_t diff = 0;
			for (int j = 0; j < 8; ++j) {
				diff += __popcnt64(ra[j] ^ rb[j]);
			}
			if (diff < 192 || diff > 320) {
				std::cout << "Seed: " << i << " diff = " << diff << std::endl;
				insensitiveProgCount[bit]++;
			}
		}
	}
	for (int bit = 0; bit < 64; ++bit) {
		std::cout << bit << " " << insensitiveProgCount[bit] << std::endl;
	}

	return 0;
}