/* $Id: jh.c 255 2011-06-07 19:50:20Z tp $ */
/*
 * JH implementation.
 *
 * ==========================(LICENSE BEGIN)============================
 *
 * Copyright (c) 2007-2010  Projet RNRT SAPHIR
 * 
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 * 
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ===========================(LICENSE END)=============================
 *
 * @author   Thomas Pornin <thomas.pornin@cryptolog.com>
 */

#define SPH_JH_64   1
#define SPH_LITTLE_ENDIAN 1

#define SPH_C32(x)	x
#define SPH_C64(x)	x
typedef uint sph_u32;
typedef ulong sph_u64;

/*
 * The internal bitslice representation may use either big-endian or
 * little-endian (true bitslice operations do not care about the bit
 * ordering, and the bit-swapping linear operations in JH happen to
 * be invariant through endianness-swapping). The constants must be
 * defined according to the chosen endianness; we use some
 * byte-swapping macros for that.
 */

#if SPH_LITTLE_ENDIAN

#define C32e(x)     ((SPH_C32(x) >> 24) \
          | ((SPH_C32(x) >>  8) & SPH_C32(0x0000FF00)) \
          | ((SPH_C32(x) <<  8) & SPH_C32(0x00FF0000)) \
          | ((SPH_C32(x) << 24) & SPH_C32(0xFF000000)))
#define dec32e_aligned   sph_dec32le_aligned
#define enc32e           sph_enc32le

#define C64e(x)     ((SPH_C64(x) >> 56) \
          | ((SPH_C64(x) >> 40) & SPH_C64(0x000000000000FF00)) \
          | ((SPH_C64(x) >> 24) & SPH_C64(0x0000000000FF0000)) \
          | ((SPH_C64(x) >>  8) & SPH_C64(0x00000000FF000000)) \
          | ((SPH_C64(x) <<  8) & SPH_C64(0x000000FF00000000)) \
          | ((SPH_C64(x) << 24) & SPH_C64(0x0000FF0000000000)) \
          | ((SPH_C64(x) << 40) & SPH_C64(0x00FF000000000000)) \
          | ((SPH_C64(x) << 56) & SPH_C64(0xFF00000000000000)))
#define dec64e_aligned   sph_dec64le_aligned
#define enc64e           sph_enc64le

#else

#define C32e(x)     SPH_C32(x)
#define dec32e_aligned   sph_dec32be_aligned
#define enc32e           sph_enc32be
#define C64e(x)     SPH_C64(x)
#define dec64e_aligned   sph_dec64be_aligned
#define enc64e           sph_enc64be

#endif

#define Sb(x0, x1, x2, x3, c)   do { \
    x3 = ~x3; \
    x0 ^= (c) & ~x2; \
    tmp = (c) ^ (x0 & x1); \
    x0 ^= x2 & x3; \
    x3 ^= ~x1 & x2; \
    x1 ^= x0 & x2; \
    x2 ^= x0 & ~x3; \
    x0 ^= x1 | x3; \
    x3 ^= x1 & x2; \
    x1 ^= tmp & x0; \
    x2 ^= tmp; \
  } while (0)

#define Lb(x0, x1, x2, x3, x4, x5, x6, x7)   do { \
    x4 ^= x1; \
    x5 ^= x2; \
    x6 ^= x3 ^ x0; \
    x7 ^= x0; \
    x0 ^= x5; \
    x1 ^= x6; \
    x2 ^= x7 ^ x4; \
    x3 ^= x4; \
  } while (0)

static const __constant ulong C[] =
{
	0x67F815DFA2DED572UL, 0x571523B70A15847BUL, 0xF6875A4D90D6AB81UL, 0x402BD1C3C54F9F4EUL, 
	0x9CFA455CE03A98EAUL, 0x9A99B26699D2C503UL, 0x8A53BBF2B4960266UL, 0x31A2DB881A1456B5UL, 
	0xDB0E199A5C5AA303UL, 0x1044C1870AB23F40UL, 0x1D959E848019051CUL, 0xDCCDE75EADEB336FUL, 
	0x416BBF029213BA10UL, 0xD027BBF7156578DCUL, 0x5078AA3739812C0AUL, 0xD3910041D2BF1A3FUL, 
	0x907ECCF60D5A2D42UL, 0xCE97C0929C9F62DDUL, 0xAC442BC70BA75C18UL, 0x23FCC663D665DFD1UL, 
	0x1AB8E09E036C6E97UL, 0xA8EC6C447E450521UL, 0xFA618E5DBB03F1EEUL, 0x97818394B29796FDUL, 
	0x2F3003DB37858E4AUL, 0x956A9FFB2D8D672AUL, 0x6C69B8F88173FE8AUL, 0x14427FC04672C78AUL, 
	0xC45EC7BD8F15F4C5UL, 0x80BB118FA76F4475UL, 0xBC88E4AEB775DE52UL, 0xF4A3A6981E00B882UL, 
	0x1563A3A9338FF48EUL, 0x89F9B7D524565FAAUL, 0xFDE05A7C20EDF1B6UL, 0x362C42065AE9CA36UL, 
	0x3D98FE4E433529CEUL, 0xA74B9A7374F93A53UL, 0x86814E6F591FF5D0UL, 0x9F5AD8AF81AD9D0EUL, 
	0x6A6234EE670605A7UL, 0x2717B96EBE280B8BUL, 0x3F1080C626077447UL, 0x7B487EC66F7EA0E0UL, 
	0xC0A4F84AA50A550DUL, 0x9EF18E979FE7E391UL, 0xD48D605081727686UL, 0x62B0E5F3415A9E7EUL, 
	0x7A205440EC1F9FFCUL, 0x84C9F4CE001AE4E3UL, 0xD895FA9DF594D74FUL, 0xA554C324117E2E55UL, 
	0x286EFEBD2872DF5BUL, 0xB2C4A50FE27FF578UL, 0x2ED349EEEF7C8905UL, 0x7F5928EB85937E44UL, 
	0x4A3124B337695F70UL, 0x65E4D61DF128865EUL, 0xE720B95104771BC7UL, 0x8A87D423E843FE74UL, 
	0xF2947692A3E8297DUL, 0xC1D9309B097ACBDDUL, 0xE01BDC5BFB301B1DUL, 0xBF829CF24F4924DAUL, 
	0xFFBF70B431BAE7A4UL, 0x48BCF8DE0544320DUL, 0x39D3BB5332FCAE3BUL, 0xA08B29E0C1C39F45UL, 
	0x0F09AEF7FD05C9E5UL, 0x34F1904212347094UL, 0x95ED44E301B771A2UL, 0x4A982F4F368E3BE9UL, 
	0x15F66CA0631D4088UL, 0xFFAF52874B44C147UL, 0x30C60AE2F14ABB7EUL, 0xE68C6ECCC5B67046UL, 
	0x00CA4FBD56A4D5A4UL, 0xAE183EC84B849DDAUL, 0xADD1643045CE5773UL, 0x67255C1468CEA6E8UL, 
	0x16E10ECBF28CDAA3UL, 0x9A99949A5806E933UL, 0x7B846FC220B2601FUL, 0x1885D1A07FACCED1UL, 
	0xD319DD8DA15B5932UL, 0x46B4A5AAC01C9A50UL, 0xBA6B04E467633D9FUL, 0x7EEE560BAB19CAF6UL, 
	0x742128A9EA79B11FUL, 0xEE51363B35F7BDE9UL, 0x76D350755AAC571DUL, 0x01707DA3FEC2463AUL, 
	0x42D8A498AFC135F7UL, 0x79676B9E20ECED78UL, 0xA8DB3AEA15638341UL, 0x832C83324D3BC3FAUL, 
	0xF347271C1F3B40A7UL, 0x9A762DB734F04059UL, 0xFD4F21D26C4E3EE7UL, 0xEF5957DC398DFDB8UL, 
	0xDAEB492B490C9B8DUL, 0x0D70F36849D7A25BUL, 0x84558D7AD0AE3B7DUL, 0x658EF8E4F0E9A5F5UL, 
	0x533B1036F4A2B8A0UL, 0x5AEC3E759E07A80CUL, 0x4F88E85692946891UL, 0x4CBCBAF8555CB05BUL, 
	0x7B9487F3993BBBE3UL, 0x5D1C6B72D6F4DA75UL, 0x6DB334DC28ACAE64UL, 0x71DB28B850A5346CUL, 
	0x2A518D10F2E261F8UL, 0xFC75DD593364DBE3UL, 0xA23FCE43F1BCAC1CUL, 0xB043E8023CD1BB67UL, 
	0x75A12988CA5B0A33UL, 0x5C5316B44D19347FUL, 0x1E4D790EC3943B92UL, 0x3FAFEEB6D7757479UL, 
	0x21391ABEF7D4A8EAUL, 0x5127234C097EF45CUL, 0xD23C32BA5324A326UL, 0xADD5A66D4A17A344UL, 
	0x08C9F2AFA63E1DB5UL, 0x563C6B91983D5983UL, 0x4D608672A17CF84CUL, 0xF6C76E08CC3EE246UL, 
	0x5E76BCB1B333982FUL, 0x2AE6C4EFA566D62BUL, 0x36D4C1BEE8B6F406UL, 0x6321EFBC1582EE74UL, 
	0x69C953F40D4EC1FDUL, 0x26585806C45A7DA7UL, 0x16FAE0061614C17EUL, 0x3F9D63283DAF907EUL, 
	0x0CD29B00E3F2C9D2UL, 0x300CD4B730CEAA5FUL, 0x9832E0F216512A74UL, 0x9AF8CEE3D830EB0DUL, 
	0x9279F1B57B9EC54BUL, 0xD36886046EE651FFUL, 0x316796E6574D239BUL, 0x05750A17F3A6E6CCUL, 
	0xCE6C3213D98176B1UL, 0x62A205F88452173CUL, 0x47154778B3CB2BF4UL, 0x486A9323825446FFUL, 
	0x65655E4E0758DF38UL, 0x8E5086FC897CFCF2UL, 0x86CA0BD0442E7031UL, 0x4E477830A20940F0UL, 
	0x8338F7D139EEA065UL, 0xBD3A2CE437E95EF7UL, 0x6FF8130126B29721UL, 0xE7DE9FEFD1ED44A3UL, 
	0xD992257615DFA08BUL, 0xBE42DC12F6F7853CUL, 0x7EB027AB7CECA7D8UL, 0xDEA83EAADA7D8D53UL, 
	0xD86902BD93CE25AAUL, 0xF908731AFD43F65AUL, 0xA5194A17DAEF5FC0UL, 0x6A21FD4C33664D97UL, 
	0x701541DB3198B435UL, 0x9B54CDEDBB0F1EEAUL, 0x72409751A163D09AUL, 0xE26F4791BF9D75F6UL
};

#define Ceven_hi(r)   (C[((r) << 2) + 0])
#define Ceven_lo(r)   (C[((r) << 2) + 1])
#define Codd_hi(r)    (C[((r) << 2) + 2])
#define Codd_lo(r)    (C[((r) << 2) + 3])

#define S(x0, x1, x2, x3, cb, r)   do { \
    Sb(x0 ## h, x1 ## h, x2 ## h, x3 ## h, cb ## hi(r)); \
    Sb(x0 ## l, x1 ## l, x2 ## l, x3 ## l, cb ## lo(r)); \
  } while (0)

#define L(x0, x1, x2, x3, x4, x5, x6, x7)   do { \
    Lb(x0 ## h, x1 ## h, x2 ## h, x3 ## h, \
      x4 ## h, x5 ## h, x6 ## h, x7 ## h); \
    Lb(x0 ## l, x1 ## l, x2 ## l, x3 ## l, \
      x4 ## l, x5 ## l, x6 ## l, x7 ## l); \
  } while (0)

#define Wz(x, c, n)   do { \
    sph_u64 t = (x ## h & (c)) << (n); \
    x ## h = ((x ## h >> (n)) & (c)) | t; \
    t = (x ## l & (c)) << (n); \
    x ## l = ((x ## l >> (n)) & (c)) | t; \
  } while (0)

#define W0(x)   Wz(x, SPH_C64(0x5555555555555555),  1)
#define W1(x)   Wz(x, SPH_C64(0x3333333333333333),  2)
#define W2(x)   Wz(x, SPH_C64(0x0F0F0F0F0F0F0F0F),  4)
#define W3(x)   Wz(x, SPH_C64(0x00FF00FF00FF00FF),  8)
#define W4(x)   Wz(x, SPH_C64(0x0000FFFF0000FFFF), 16)
#define W5(x)   Wz(x, SPH_C64(0x00000000FFFFFFFF), 32)
#define W6(x)   do { \
    sph_u64 t = x ## h; \
    x ## h = x ## l; \
    x ## l = t; \
  } while (0)

#define SL(ro)   SLu(r + ro, ro)

#define SLu(r, ro)   do { \
    S(h0, h2, h4, h6, Ceven_, r); \
    S(h1, h3, h5, h7, Codd_, r); \
    L(h0, h2, h4, h6, h1, h3, h5, h7); \
    W ## ro(h1); \
    W ## ro(h3); \
    W ## ro(h5); \
    W ## ro(h7); \
  } while (0)

#if SPH_SMALL_FOOTPRINT_JH

/*
 * The "small footprint" 64-bit version just uses a partially unrolled
 * loop.
 */

#define E8   do { \
    unsigned r; \
    for (r = 0; r < 42; r += 7) { \
      SL(0); \
      SL(1); \
      SL(2); \
      SL(3); \
      SL(4); \
      SL(5); \
      SL(6); \
    } \
  } while (0)

#else

/*
 * On a "true 64-bit" architecture, we can unroll at will.
 */

#define E8   do { \
    SLu( 0, 0); \
    SLu( 1, 1); \
    SLu( 2, 2); \
    SLu( 3, 3); \
    SLu( 4, 4); \
    SLu( 5, 5); \
    SLu( 6, 6); \
    SLu( 7, 0); \
    SLu( 8, 1); \
    SLu( 9, 2); \
    SLu(10, 3); \
    SLu(11, 4); \
    SLu(12, 5); \
    SLu(13, 6); \
    SLu(14, 0); \
    SLu(15, 1); \
    SLu(16, 2); \
    SLu(17, 3); \
    SLu(18, 4); \
    SLu(19, 5); \
    SLu(20, 6); \
    SLu(21, 0); \
    SLu(22, 1); \
    SLu(23, 2); \
    SLu(24, 3); \
    SLu(25, 4); \
    SLu(26, 5); \
    SLu(27, 6); \
    SLu(28, 0); \
    SLu(29, 1); \
    SLu(30, 2); \
    SLu(31, 3); \
    SLu(32, 4); \
    SLu(33, 5); \
    SLu(34, 6); \
    SLu(35, 0); \
    SLu(36, 1); \
    SLu(37, 2); \
    SLu(38, 3); \
    SLu(39, 4); \
    SLu(40, 5); \
    SLu(41, 6); \
  } while (0)

#endif
