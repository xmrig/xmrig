/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2019 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018      Lee Clagett <https://github.com/vtnerd>
 * Copyright 2018-2019 tevador     <tevador@gmail.com>
 * Copyright 2018-2019 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2019 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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


#include "crypto/kawpow/KPHash.h"
#include "crypto/kawpow/KPCache.h"
#include "3rdparty/libethash/ethash.h"
#include "3rdparty/libethash/ethash_internal.h"
#include "3rdparty/libethash/data_sizes.h"

#ifdef _MSC_VER
#include <intrin.h>
#endif

namespace xmrig {


static const uint32_t ravencoin_kawpow[15] = {
        0x00000072, //R
        0x00000041, //A
        0x00000056, //V
        0x00000045, //E
        0x0000004E, //N
        0x00000043, //C
        0x0000004F, //O
        0x00000049, //I
        0x0000004E, //N
        0x0000004B, //K
        0x00000041, //A
        0x00000057, //W
        0x00000050, //P
        0x0000004F, //O
        0x00000057, //W
};


static const uint32_t fnv_prime = 0x01000193;
static const uint32_t fnv_offset_basis = 0x811c9dc5;


static inline uint32_t fnv1a(uint32_t u, uint32_t v)
{
    return (u ^ v) * fnv_prime;
}


static inline uint32_t kiss99(uint32_t& z, uint32_t& w, uint32_t& jsr, uint32_t& jcong)
{
    z = 36969 * (z & 0xffff) + (z >> 16);
    w = 18000 * (w & 0xffff) + (w >> 16);

    jcong = 69069 * jcong + 1234567;

    jsr ^= (jsr << 17);
    jsr ^= (jsr >> 13);
    jsr ^= (jsr << 5);

    return (((z << 16) + w) ^ jcong) + jsr;
}


static inline uint32_t rotl(uint32_t n, uint32_t c)
{
#ifdef _MSC_VER
    return _rotl(n, c);
#else
    c &= 31;
    uint32_t neg_c = (uint32_t)(-(int32_t)c);
    return (n << c) | (n >> (neg_c & 31));
#endif
}


static inline uint32_t rotr(uint32_t n, uint32_t c)
{
#ifdef _MSC_VER
    return _rotr(n, c);
#else
    c &= 31;
    uint32_t neg_c = (uint32_t)(-(int32_t)c);
    return (n >> c) | (n << (neg_c & 31));
#endif
}


static inline void random_merge(uint32_t& a, uint32_t b, uint32_t selector)
{
    const uint32_t x = (selector >> 16) % 31 + 1;
    switch (selector % 4)
    {
    case 0:
        a = (a * 33) + b;
        break;
    case 1:
        a = (a ^ b) * 33;
        break;
    case 2:
        a = rotl(a, x) ^ b;
        break;
    case 3:
        a = rotr(a, x) ^ b;
        break;
    default:
#ifdef _MSC_VER
        __assume(false);
#else
        __builtin_unreachable();
#endif
        break;
    }
}


static inline uint32_t clz(uint32_t a)
{
#ifdef _MSC_VER
    unsigned long index;
    _BitScanReverse(&index, a);
    return a ? (31 - index) : 32;
#else
    return a ? (uint32_t)__builtin_clz(a) : 32;
#endif
}


static inline uint32_t popcount(uint32_t a)
{
#ifdef _MSC_VER
    return __popcnt(a);
#else
    return __builtin_popcount(a);
#endif
}


static inline uint32_t random_math(uint32_t a, uint32_t b, uint32_t selector)
{
    switch (selector % 11)
    {
    case 0:
        return a + b;
    case 1:
        return a * b;
    case 2:
        return (uint64_t(a) * b) >> 32;
    case 3:
        return (a < b) ? a : b;
    case 4:
        return rotl(a, b);
    case 5:
        return rotr(a, b);
    case 6:
        return a & b;
    case 7:
        return a | b;
    case 8:
        return a ^ b;
    case 9:
        return clz(a) + clz(b);
    case 10:
        return popcount(a) + popcount(b);
    default:
#ifdef _MSC_VER
        __assume(false);
#else
        __builtin_unreachable();
#endif
        break;
    }
}


void KPHash::calculate(const KPCache& light_cache, uint32_t block_height, const uint8_t (&header_hash)[32], uint64_t nonce, uint32_t (&output)[8], uint32_t (&mix_hash)[8])
{
    uint32_t keccak_state[25];
    uint32_t mix[LANES][REGS];

    memcpy(keccak_state, header_hash, sizeof(header_hash));
    memcpy(keccak_state + 8, &nonce, sizeof(nonce));
    memcpy(keccak_state + 10, ravencoin_kawpow, sizeof(ravencoin_kawpow));

    ethash_keccakf800(keccak_state);

    uint32_t z = fnv1a(fnv_offset_basis, keccak_state[0]);
    uint32_t w = fnv1a(z, keccak_state[1]);
    uint32_t jsr, jcong;

    for (uint32_t l = 0; l < LANES; ++l) {
        uint32_t z1 = z;
        uint32_t w1 = w;
        jsr = fnv1a(w, l);
        jcong = fnv1a(jsr, l);

        for (uint32_t r = 0; r < REGS; ++r) {
            mix[l][r] = kiss99(z1, w1, jsr, jcong);
        }
    }

    const uint32_t prog_number = block_height / PERIOD_LENGTH;

    uint32_t dst_seq[REGS];
    uint32_t src_seq[REGS];

    z = fnv1a(fnv_offset_basis, prog_number);
    w = fnv1a(z, 0);
    jsr = fnv1a(w, prog_number);
    jcong = fnv1a(jsr, 0);

    for (uint32_t i = 0; i < REGS; ++i)
    {
        dst_seq[i] = i;
        src_seq[i] = i;
    }

    for (uint32_t i = REGS; i > 1; --i)
    {
        std::swap(dst_seq[i - 1], dst_seq[kiss99(z, w, jsr, jcong) % i]);
        std::swap(src_seq[i - 1], src_seq[kiss99(z, w, jsr, jcong) % i]);
    }

    const uint32_t epoch = light_cache.epoch();
    const uint32_t num_items = static_cast<uint32_t>(dag_sizes[epoch] / ETHASH_MIX_BYTES / 2);

    constexpr size_t num_words_per_lane = 256 / (sizeof(uint32_t) * LANES);
    constexpr int max_operations = (CNT_CACHE > CNT_MATH) ? CNT_CACHE : CNT_MATH;

    ethash_light cache;
    cache.cache = light_cache.data();
    cache.cache_size = light_cache.size();
    cache.block_number = block_height;

    cache.num_parent_nodes = cache.cache_size / sizeof(node);
    KPCache::calculate_fast_mod_data(cache.num_parent_nodes, cache.reciprocal, cache.increment, cache.shift);

    uint32_t z0 = z;
    uint32_t w0 = w;
    uint32_t jsr0 = jsr;
    uint32_t jcong0 = jcong;

    for (uint32_t r = 0; r < ETHASH_ACCESSES; ++r) {
        uint32_t item_index = (mix[r % LANES][0] % num_items) * 4;

        node item[4];
        ethash_calculate_dag_item_opt(item + 0, item_index + 0, KPCache::num_dataset_parents, &cache);
        ethash_calculate_dag_item_opt(item + 1, item_index + 1, KPCache::num_dataset_parents, &cache);
        ethash_calculate_dag_item_opt(item + 2, item_index + 2, KPCache::num_dataset_parents, &cache);
        ethash_calculate_dag_item_opt(item + 3, item_index + 3, KPCache::num_dataset_parents, &cache);

        uint32_t dst_counter = 0;
        uint32_t src_counter = 0;

        z = z0;
        w = w0;
        jsr = jsr0;
        jcong = jcong0;

        for (uint32_t i = 0; i < max_operations; ++i) {
            if (i < CNT_CACHE) {
                const uint32_t src = src_seq[(src_counter++) % REGS];
                const uint32_t dst = dst_seq[(dst_counter++) % REGS];
                const uint32_t sel = kiss99(z, w, jsr, jcong);
                for (uint32_t j = 0; j < LANES; ++j) {
                    random_merge(mix[j][dst], light_cache.l1_cache()[mix[j][src] % KPCache::l1_cache_num_items], sel);
                }
            }

            if (i < CNT_MATH)
            {
                const uint32_t src_rnd = kiss99(z, w, jsr, jcong) % (REGS * (REGS - 1));
                const uint32_t src1 = src_rnd % REGS;
                uint32_t src2 = src_rnd / REGS;
                if (src2 >= src1) {
                    ++src2;
                }

                const uint32_t sel1 = kiss99(z, w, jsr, jcong);
                const uint32_t dst = dst_seq[(dst_counter++) % REGS];
                const uint32_t sel2 = kiss99(z, w, jsr, jcong);

                for (size_t l = 0; l < LANES; ++l)
                {
                    const uint32_t data = random_math(mix[l][src1], mix[l][src2], sel1);
                    random_merge(mix[l][dst], data, sel2);
                }
            }
        }

        uint32_t dsts[num_words_per_lane];
        uint32_t sels[num_words_per_lane];
        for (uint32_t i = 0; i < num_words_per_lane; ++i) {
            dsts[i] = (i == 0) ? 0 : dst_seq[(dst_counter++) % REGS];
            sels[i] = kiss99(z, w, jsr, jcong);
        }

        for (uint32_t l = 0; l < LANES; ++l) {
            const uint32_t offset = ((l ^ r) % LANES) * num_words_per_lane;
            for (size_t i = 0; i < num_words_per_lane; ++i) {
                random_merge(mix[l][dsts[i]], ((uint32_t*)item)[offset + i], sels[i]);
            }
        }
    }

    uint32_t lane_hash[LANES];
    for (uint32_t l = 0; l < LANES; ++l)
    {
        lane_hash[l] = fnv_offset_basis;
        for (uint32_t i = 0; i < REGS; ++i) {
            lane_hash[l] = fnv1a(lane_hash[l], mix[l][i]);
        }
    }

    constexpr uint32_t num_words = 8;

    for (uint32_t i = 0; i < num_words; ++i) {
        mix_hash[i] = fnv_offset_basis;
    }

    for (uint32_t l = 0; l < LANES; ++l)
        mix_hash[l % num_words] = fnv1a(mix_hash[l % num_words], lane_hash[l]);

    memcpy(keccak_state + 8, mix_hash, sizeof(mix_hash));
    memcpy(keccak_state + 16, ravencoin_kawpow, sizeof(uint32_t) * 9);

    ethash_keccakf800(keccak_state);

    memcpy(output, keccak_state, sizeof(output));
}


} // namespace xmrig
