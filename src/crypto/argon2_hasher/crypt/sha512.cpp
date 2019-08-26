#include "crypto/argon2_hasher/common/DLLExport.h"

#include <cstring>
#include <fstream>
#include "sha512.h"

const unsigned long long SHA512::sha512_k[80] = //ULL = uint64
        {0x428a2f98d728ae22ULL, 0x7137449123ef65cdULL,
         0xb5c0fbcfec4d3b2fULL, 0xe9b5dba58189dbbcULL,
         0x3956c25bf348b538ULL, 0x59f111f1b605d019ULL,
         0x923f82a4af194f9bULL, 0xab1c5ed5da6d8118ULL,
         0xd807aa98a3030242ULL, 0x12835b0145706fbeULL,
         0x243185be4ee4b28cULL, 0x550c7dc3d5ffb4e2ULL,
         0x72be5d74f27b896fULL, 0x80deb1fe3b1696b1ULL,
         0x9bdc06a725c71235ULL, 0xc19bf174cf692694ULL,
         0xe49b69c19ef14ad2ULL, 0xefbe4786384f25e3ULL,
         0x0fc19dc68b8cd5b5ULL, 0x240ca1cc77ac9c65ULL,
         0x2de92c6f592b0275ULL, 0x4a7484aa6ea6e483ULL,
         0x5cb0a9dcbd41fbd4ULL, 0x76f988da831153b5ULL,
         0x983e5152ee66dfabULL, 0xa831c66d2db43210ULL,
         0xb00327c898fb213fULL, 0xbf597fc7beef0ee4ULL,
         0xc6e00bf33da88fc2ULL, 0xd5a79147930aa725ULL,
         0x06ca6351e003826fULL, 0x142929670a0e6e70ULL,
         0x27b70a8546d22ffcULL, 0x2e1b21385c26c926ULL,
         0x4d2c6dfc5ac42aedULL, 0x53380d139d95b3dfULL,
         0x650a73548baf63deULL, 0x766a0abb3c77b2a8ULL,
         0x81c2c92e47edaee6ULL, 0x92722c851482353bULL,
         0xa2bfe8a14cf10364ULL, 0xa81a664bbc423001ULL,
         0xc24b8b70d0f89791ULL, 0xc76c51a30654be30ULL,
         0xd192e819d6ef5218ULL, 0xd69906245565a910ULL,
         0xf40e35855771202aULL, 0x106aa07032bbd1b8ULL,
         0x19a4c116b8d2d0c8ULL, 0x1e376c085141ab53ULL,
         0x2748774cdf8eeb99ULL, 0x34b0bcb5e19b48a8ULL,
         0x391c0cb3c5c95a63ULL, 0x4ed8aa4ae3418acbULL,
         0x5b9cca4f7763e373ULL, 0x682e6ff3d6b2b8a3ULL,
         0x748f82ee5defb2fcULL, 0x78a5636f43172f60ULL,
         0x84c87814a1f0ab72ULL, 0x8cc702081a6439ecULL,
         0x90befffa23631e28ULL, 0xa4506cebde82bde9ULL,
         0xbef9a3f7b2c67915ULL, 0xc67178f2e372532bULL,
         0xca273eceea26619cULL, 0xd186b8c721c0c207ULL,
         0xeada7dd6cde0eb1eULL, 0xf57d4f7fee6ed178ULL,
         0x06f067aa72176fbaULL, 0x0a637dc5a2c898a6ULL,
         0x113f9804bef90daeULL, 0x1b710b35131c471bULL,
         0x28db77f523047d84ULL, 0x32caab7b40c72493ULL,
         0x3c9ebe0a15c9bebcULL, 0x431d67c49c100d4cULL,
         0x4cc5d4becb3e42b6ULL, 0x597f299cfc657e2aULL,
         0x5fcb6fab3ad6faecULL, 0x6c44198c4a475817ULL};

void SHA512::transform(const unsigned char *message, unsigned int block_nb)
{
    uint64 w[80];
    uint64 wv[8];
    uint64 t1, t2;
    const unsigned char *sub_block;
    int i, j;
    for (i = 0; i < (int) block_nb; i++) {
        sub_block = message + (i << 7);
        for (j = 0; j < 16; j++) {
            SHA2_PACK64(&sub_block[j << 3], &w[j]);
        }
        for (j = 16; j < 80; j++) {
            w[j] =  SHA512_F4(w[j -  2]) + w[j -  7] + SHA512_F3(w[j - 15]) + w[j - 16];
        }
        for (j = 0; j < 8; j++) {
            wv[j] = m_h[j];
        }
        for (j = 0; j < 80; j++) {
            t1 = wv[7] + SHA512_F2(wv[4]) + SHA2_CH(wv[4], wv[5], wv[6])
                 + sha512_k[j] + w[j];
            t2 = SHA512_F1(wv[0]) + SHA2_MAJ(wv[0], wv[1], wv[2]);
            wv[7] = wv[6];
            wv[6] = wv[5];
            wv[5] = wv[4];
            wv[4] = wv[3] + t1;
            wv[3] = wv[2];
            wv[2] = wv[1];
            wv[1] = wv[0];
            wv[0] = t1 + t2;
        }
        for (j = 0; j < 8; j++) {
            m_h[j] += wv[j];
        }

    }
}

void SHA512::init()
{
    m_h[0] = 0x6a09e667f3bcc908ULL;
    m_h[1] = 0xbb67ae8584caa73bULL;
    m_h[2] = 0x3c6ef372fe94f82bULL;
    m_h[3] = 0xa54ff53a5f1d36f1ULL;
    m_h[4] = 0x510e527fade682d1ULL;
    m_h[5] = 0x9b05688c2b3e6c1fULL;
    m_h[6] = 0x1f83d9abfb41bd6bULL;
    m_h[7] = 0x5be0cd19137e2179ULL;
    m_len = 0;
    m_tot_len = 0;
}

void SHA512::update(const unsigned char *message, unsigned int len)
{
    unsigned int block_nb;
    unsigned int new_len, rem_len, tmp_len;
    const unsigned char *shifted_message;
    tmp_len = SHA384_512_BLOCK_SIZE - m_len;
    rem_len = len < tmp_len ? len : tmp_len;
    memcpy(&m_block[m_len], message, rem_len);
    if (m_len + len < SHA384_512_BLOCK_SIZE) {
        m_len += len;
        return;
    }
    new_len = len - rem_len;
    block_nb = new_len / SHA384_512_BLOCK_SIZE;
    shifted_message = message + rem_len;
    transform(m_block, 1);
    transform(shifted_message, block_nb);
    rem_len = new_len % SHA384_512_BLOCK_SIZE;
    memcpy(m_block, &shifted_message[block_nb << 7], rem_len);
    m_len = rem_len;
    m_tot_len += (block_nb + 1) << 7;
}

void SHA512::final(unsigned char *digest)
{
    unsigned int block_nb;
    unsigned int pm_len;
    unsigned int len_b;
    int i;
    block_nb = 1 + ((SHA384_512_BLOCK_SIZE - 17)
                    < (m_len % SHA384_512_BLOCK_SIZE));
    len_b = (m_tot_len + m_len) << 3;
    pm_len = block_nb << 7;
    memset(m_block + m_len, 0, pm_len - m_len);
    m_block[m_len] = 0x80;
    SHA2_UNPACK32(len_b, m_block + pm_len - 4);
    transform(m_block, block_nb);
    for (i = 0 ; i < 8; i++) {
        SHA2_UNPACK64(m_h[i], &digest[i << 3]);
    }
}

unsigned char *SHA512::hash(unsigned char *input, size_t length)
{
    unsigned char *digest = (unsigned char*)malloc(SHA512::DIGEST_SIZE);
    memset(digest,0,SHA512::DIGEST_SIZE);
    SHA512 ctx = SHA512();
    ctx.init();
    ctx.update(input, length);
    ctx.final(digest);
    return digest;
}
