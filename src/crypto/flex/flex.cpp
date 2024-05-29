#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#include "crypto/cn/CnHash.h"
#include "backend/cpu/Cpu.h"

extern "C" {
#include "../randomx/panthera/sysendian.h"
#include "../ghostrider/sph_blake.h"
#include "../ghostrider/sph_bmw.h"
#include "../ghostrider/sph_groestl.h"
#include "../ghostrider/sph_jh.h"
#include "../ghostrider/sph_keccak.h"
#include "../ghostrider/sph_skein.h"
#include "../ghostrider/sph_luffa.h"
#include "../ghostrider/sph_cubehash.h"
#include "../ghostrider/sph_shavite.h"
#include "../ghostrider/sph_simd.h"
#include "../ghostrider/sph_echo.h"
#include "../ghostrider/sph_hamsi.h"
#include "../ghostrider/sph_fugue.h"
#include "../ghostrider/sph_shabal.h"
#include "../ghostrider/sph_whirlpool.h"
#include "../ghostrider/sph_sha2.h"
}
#include <stdio.h>

using namespace xmrig;

enum Algo {
        BLAKE = 0,
        BMW,
        GROESTL,
        KECCAK,
        SKEIN,
        LUFFA,
        CUBEHASH,
        SHAVITE,
        SIMD,
        ECHO,
        HAMSI,
        FUGUE,
        SHABAL,
        WHIRLPOOL,
        HASH_FUNC_COUNT
};

enum CNAlgo {
	CNDark = 0,
	CNDarklite,
	CNFast,
	CNLite,
	CNTurtle,
	CNTurtlelite,
	CN_HASH_FUNC_COUNT
};


static void selectAlgo(unsigned char nibble, bool* selectedAlgos, uint8_t* selectedIndex, int algoCount, int* currentCount) {
	uint8_t algoDigit = (nibble & 0x0F) % algoCount;
	if(!selectedAlgos[algoDigit]) {
		selectedAlgos[algoDigit] = true;
		selectedIndex[currentCount[0]] = algoDigit;
		currentCount[0] = currentCount[0] + 1;
	}
	algoDigit = (nibble >> 4) % algoCount;
	if(!selectedAlgos[algoDigit]) {
		selectedAlgos[algoDigit] = true;
		selectedIndex[currentCount[0]] = algoDigit;
		currentCount[0] = currentCount[0] + 1;
	}
}

static void getAlgoString(void *mem, unsigned int size, uint8_t* selectedAlgoOutput, int algoCount) {
  int i;
  unsigned char *p = (unsigned char *)mem;
  unsigned int len = size/2;
  unsigned char j = 0;
  bool selectedAlgo[algoCount];
  for(int z=0; z < algoCount; z++) {
	  selectedAlgo[z] = false;
  }
  int selectedCount = 0;
  for (i=0;i<len; i++) {
	  selectAlgo(p[i], selectedAlgo, selectedAlgoOutput, algoCount, &selectedCount);
	  if(selectedCount == algoCount) {
		  break;
	  }
  }
  if(selectedCount < algoCount) {
	for(uint8_t i = 0; i < algoCount; i++) {
		if(!selectedAlgo[i]) {
			selectedAlgoOutput[selectedCount] = i;
			selectedCount++;
		}
	}
  }
}

void print_hex_memory(void *mem, unsigned int size) {
  int i;
  unsigned char *p = (unsigned char *)mem;
  unsigned int len = size/2;
  for (i=0;i<len; i++) {
    printf("%02x", p[(len - i - 1)]);
  }
  printf("\n");
}

void SwapBytes(void *pv, unsigned int n)
{
    char *p = (char*)pv;
    unsigned int lo, hi;
    for(lo=0, hi=n-1; hi>lo; lo++, hi--)
    {
        char tmp=p[lo];
        p[lo] = p[hi];
        p[hi] = tmp;
    }
}

void flex_hash(const char* input, char* output, cryptonight_ctx** ctx) {
	uint32_t hash[64/4];
	sph_blake512_context ctx_blake;
	sph_bmw512_context ctx_bmw;
	sph_groestl512_context ctx_groestl;
	sph_jh512_context ctx_jh;
	sph_keccak512_context ctx_keccak;
	sph_skein512_context ctx_skein;
	sph_luffa512_context ctx_luffa;
	sph_cubehash512_context ctx_cubehash;
	sph_shavite512_context ctx_shavite;
	sph_simd512_context ctx_simd;
	sph_echo512_context ctx_echo;
	sph_hamsi512_context ctx_hamsi;
	sph_fugue512_context ctx_fugue;
	sph_shabal512_context ctx_shabal;
	sph_whirlpool_context ctx_whirlpool;
	sph_sha256_context ctx_sha;
	void *in = (void*) input;
	int size = 80;
	sph_keccak512_init(&ctx_keccak);
	sph_keccak512(&ctx_keccak, in, size);
	sph_keccak512_close(&ctx_keccak, hash);
	uint8_t selectedAlgoOutput[15] = {0};
	uint8_t selectedCNAlgoOutput[14] = {0};
	getAlgoString(&hash, 64, selectedAlgoOutput, 14);
	getAlgoString(&hash, 64, selectedCNAlgoOutput, 6);
	//printf("previous hash=");
	//print_hex_memory(&input[4], 64);
        const CnHash::AlgoVariant av = Cpu::info()->hasAES() ? CnHash::AV_SINGLE : CnHash::AV_SINGLE_SOFT;
	int i;
	for (i = 0; i < 18; i++)
	{
		uint8_t algo;
		uint8_t cnAlgo;
		int coreSelection;
		int cnSelection = -1;
		if(i < 5) {
			coreSelection = i;
		} else if(i < 11) {
			coreSelection = i-1;
		} else {
			coreSelection = i-2;
		}
		if(i==5) {
			coreSelection = -1;
			cnSelection = 0;
		}
		if(i==11) {
			coreSelection = -1;
			cnSelection = 1;
		}
		if(i==17) {
			coreSelection = -1;
			cnSelection = 2;
		}
		if(coreSelection >= 0) {
			algo = selectedAlgoOutput[(uint8_t)coreSelection];
		} else {
			algo = 16; // skip core hashing for this loop iteration
		}
		if(cnSelection >=0) {
			cnAlgo = selectedCNAlgoOutput[(uint8_t)cnSelection];
		} else {
			cnAlgo = 14; // skip cn hashing for this loop iteration
		}
		//selection cnAlgo. if a CN algo is selected then core algo will not be selected
                cn_hash_fun f;
		switch(cnAlgo)
		{
		 case CNDark:
                        //cryptonightdark_hash((const char*)in, (char*)hash, size, 1); 
                        f = CnHash::fn(Algorithm::CN_GR_0, av, Assembly::AUTO); 
			break;
		 case CNDarklite:
                        //cryptonightdarklite_hash((const char*)in, (char*)hash, size, 1);
                        f = CnHash::fn(Algorithm::CN_GR_1, av, Assembly::AUTO);
			break;
		 case CNFast:
                        //cryptonightfast_hash((const char*)in, (char*)hash, size, 1);
                        f = CnHash::fn(Algorithm::CN_GR_2, av, Assembly::AUTO);
			break;
		 case CNLite:
                        //cryptonightlite_hash((const char*)in, (char*)hash, size, 1);
                        f = CnHash::fn(Algorithm::CN_GR_3, av, Assembly::AUTO);
			break;
		 case CNTurtle:
                        //cryptonightturtle_hash((const char*)in, (char*)hash, size, 1); 
                        f = CnHash::fn(Algorithm::CN_GR_4, av, Assembly::AUTO);
			break;
		 case CNTurtlelite:
                        //cryptonightturtlelite_hash((const char*)in, (char*)hash, size, 1);
                        f = CnHash::fn(Algorithm::CN_GR_5, av, Assembly::AUTO);
			break;
                 default:
                        f = nullptr;
		}
                if (f) f((const uint8_t*)in, size, (uint8_t*)hash, ctx, 101);

		//selection core algo
		switch (algo) {
		case BLAKE:
				sph_blake512_init(&ctx_blake);
				sph_blake512(&ctx_blake, in, size);
				sph_blake512_close(&ctx_blake, hash);
				break;
		case BMW:
				sph_bmw512_init(&ctx_bmw);
				sph_bmw512(&ctx_bmw, in, size);
				sph_bmw512_close(&ctx_bmw, hash);
				break;
		case GROESTL:
				sph_groestl512_init(&ctx_groestl);
				sph_groestl512(&ctx_groestl, in, size);
				sph_groestl512_close(&ctx_groestl, hash);
				break;
		case KECCAK:
				sph_keccak512_init(&ctx_keccak);
				sph_keccak512(&ctx_keccak, in, size);
				sph_keccak512_close(&ctx_keccak, hash);
				break;
		case SKEIN:
				sph_skein512_init(&ctx_skein);
				sph_skein512(&ctx_skein, in, size);
				sph_skein512_close(&ctx_skein, hash);
				break;
		case LUFFA:
				sph_luffa512_init(&ctx_luffa);
				sph_luffa512(&ctx_luffa, in, size);
				sph_luffa512_close(&ctx_luffa, hash);
				break;
		case CUBEHASH:
				sph_cubehash512_init(&ctx_cubehash);
				sph_cubehash512(&ctx_cubehash, in, size);
				sph_cubehash512_close(&ctx_cubehash, hash);
				break;
		case SHAVITE:
				sph_shavite512_init(&ctx_shavite);
				sph_shavite512(&ctx_shavite, in, size);
				sph_shavite512_close(&ctx_shavite, hash);
				break;
		case SIMD:
				sph_simd512_init(&ctx_simd);
				sph_simd512(&ctx_simd, in, size);
				sph_simd512_close(&ctx_simd, hash);
				break;
		case ECHO:
				sph_echo512_init(&ctx_echo);
				sph_echo512(&ctx_echo, in, size);
				sph_echo512_close(&ctx_echo, hash);
				break;
		case HAMSI:
				sph_hamsi512_init(&ctx_hamsi);
				sph_hamsi512(&ctx_hamsi, in, size);
				sph_hamsi512_close(&ctx_hamsi, hash);
				break;
		case FUGUE:
				sph_fugue512_init(&ctx_fugue);
				sph_fugue512(&ctx_fugue, in, size);
				sph_fugue512_close(&ctx_fugue, hash);
				break;
		case SHABAL:
				sph_shabal512_init(&ctx_shabal);
				sph_shabal512(&ctx_shabal, in, size);
				sph_shabal512_close(&ctx_shabal, hash);
				break;
		case WHIRLPOOL:
				sph_whirlpool_init(&ctx_whirlpool);
				sph_whirlpool(&ctx_whirlpool, in, size);
				sph_whirlpool_close(&ctx_whirlpool, hash);
				break;
		}
		in = (void*) hash;
		size = 64;
	}
	sph_keccak256_init(&ctx_keccak);
	sph_keccak256(&ctx_keccak, in, size);
	sph_keccak256_close(&ctx_keccak, hash);
	memcpy(output, hash, 32);
}
