#include <iostream>
#include <cstdint>
#include <vector>
#include <unordered_set>
#include "../superscalar.hpp"
#include "../common.hpp"

int main() {
	std::cout << "THIS PROGRAM REQUIRES MORE THAN 16 GB OF RAM TO COMPLETE" << std::endl;
	std::vector<uint64_t> dummy;
	constexpr uint64_t superscalarMul0 = 6364136223846793005ULL;
	constexpr uint64_t superscalarAdd1 = 0x810A978A59F5A1FC; //9298410992540426748ULL; //9298410992540426048ULL
	constexpr uint64_t superscalarAdd2 = 12065312585734608966ULL;
	constexpr uint64_t superscalarAdd3 = 0x8126B91CBF22495C; //9306329213124610396ULL;
	constexpr uint64_t superscalarAdd4 = 5281919268842080866ULL;
	constexpr uint64_t superscalarAdd5 = 10536153434571861004ULL;
	constexpr uint64_t superscalarAdd6 = 3398623926847679864ULL;
	constexpr uint64_t superscalarAdd7 = 9549104520008361294ULL;
	constexpr uint32_t totalItems = randomx::DatasetSize / randomx::CacheLineSize;
	std::unordered_set<uint64_t> registerValues;
	registerValues.reserve(totalItems);
	registerValues.rehash(totalItems);
	int collisionCount[9] = { 0 };
	for (uint32_t itemNumber = 0; itemNumber < totalItems; ++itemNumber) {
		uint64_t rl[8];
		rl[0] = (itemNumber + 1) * superscalarMul0;
		rl[1] = rl[0] ^ superscalarAdd1;
		rl[2] = rl[0] ^ superscalarAdd2;
		rl[3] = rl[0] ^ superscalarAdd3;
		rl[4] = rl[0] ^ superscalarAdd4;
		rl[5] = rl[0] ^ superscalarAdd5;
		rl[6] = rl[0] ^ superscalarAdd6;
		rl[7] = rl[0] ^ superscalarAdd7;
		int blockCollisions = 0;
		for (int i = 0; i < 8; ++i) {
			uint64_t reducedValue = rl[i] & 0x3FFFFFFFFFFFF8; //bits 3-53 only
			if (registerValues.find(reducedValue) != registerValues.end()) {
				blockCollisions++;
				std::cout << "Item " << itemNumber << ": collision of register r" << i << std::endl;
			}
			else {
				registerValues.insert(reducedValue);
			}
		}
		collisionCount[blockCollisions]++;
		if ((itemNumber % (320 * 1024)) == 0)
			std::cout << "Item " << itemNumber << " processed" << std::endl;
	}

	for (int i = 0; i < 9; ++i) {
		std::cout << i << " register(s) collide in " << collisionCount[i] << " items" << std::endl;
	}

	return 0;
}