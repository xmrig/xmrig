#ifndef WOLF_AES_CL
#define WOLF_AES_CL

#ifdef cl_amd_media_ops2
#   pragma OPENCL EXTENSION cl_amd_media_ops2 : enable

#   define xmrig_amd_bfe(src0, src1, src2) amd_bfe(src0, src1, src2)
#else
/* taken from: https://www.khronos.org/registry/OpenCL/extensions/amd/cl_amd_media_ops2.txt
 *     Built-in Function:
 *     uintn amd_bfe (uintn src0, uintn src1, uintn src2)
 *   Description
 *     NOTE: operator >> below represent logical right shift
 *     offset = src1.s0 & 31;
 *     width = src2.s0 & 31;
 *     if width = 0
 *         dst.s0 = 0;
 *     else if (offset + width) < 32
 *         dst.s0 = (src0.s0 << (32 - offset - width)) >> (32 - width);
 *     else
 *         dst.s0 = src0.s0 >> offset;
 *     similar operation applied to other components of the vectors
 */
inline int xmrig_amd_bfe(const uint src0, const uint offset, const uint width)
{
    /* casts are removed because we can implement everything as uint
     * int offset = src1;
     * int width = src2;
     * remove check for edge case, this function is always called with
     * `width==8`
     * @code
     *   if ( width == 0 )
     *      return 0;
     * @endcode
     */
    if ((offset + width) < 32u) {
        return (src0 << (32u - offset - width)) >> (32u - width);
    }

    return src0 >> offset;
}
#endif


// AES table - the other three are generated on the fly

static const __constant uint AES0_C[256] =
{
    0xA56363C6U, 0x847C7CF8U, 0x997777EEU, 0x8D7B7BF6U,
    0x0DF2F2FFU, 0xBD6B6BD6U, 0xB16F6FDEU, 0x54C5C591U,
    0x50303060U, 0x03010102U, 0xA96767CEU, 0x7D2B2B56U,
    0x19FEFEE7U, 0x62D7D7B5U, 0xE6ABAB4DU, 0x9A7676ECU,
    0x45CACA8FU, 0x9D82821FU, 0x40C9C989U, 0x877D7DFAU,
    0x15FAFAEFU, 0xEB5959B2U, 0xC947478EU, 0x0BF0F0FBU,
    0xECADAD41U, 0x67D4D4B3U, 0xFDA2A25FU, 0xEAAFAF45U,
    0xBF9C9C23U, 0xF7A4A453U, 0x967272E4U, 0x5BC0C09BU,
    0xC2B7B775U, 0x1CFDFDE1U, 0xAE93933DU, 0x6A26264CU,
    0x5A36366CU, 0x413F3F7EU, 0x02F7F7F5U, 0x4FCCCC83U,
    0x5C343468U, 0xF4A5A551U, 0x34E5E5D1U, 0x08F1F1F9U,
    0x937171E2U, 0x73D8D8ABU, 0x53313162U, 0x3F15152AU,
    0x0C040408U, 0x52C7C795U, 0x65232346U, 0x5EC3C39DU,
    0x28181830U, 0xA1969637U, 0x0F05050AU, 0xB59A9A2FU,
    0x0907070EU, 0x36121224U, 0x9B80801BU, 0x3DE2E2DFU,
    0x26EBEBCDU, 0x6927274EU, 0xCDB2B27FU, 0x9F7575EAU,
    0x1B090912U, 0x9E83831DU, 0x742C2C58U, 0x2E1A1A34U,
    0x2D1B1B36U, 0xB26E6EDCU, 0xEE5A5AB4U, 0xFBA0A05BU,
    0xF65252A4U, 0x4D3B3B76U, 0x61D6D6B7U, 0xCEB3B37DU,
    0x7B292952U, 0x3EE3E3DDU, 0x712F2F5EU, 0x97848413U,
    0xF55353A6U, 0x68D1D1B9U, 0x00000000U, 0x2CEDEDC1U,
    0x60202040U, 0x1FFCFCE3U, 0xC8B1B179U, 0xED5B5BB6U,
    0xBE6A6AD4U, 0x46CBCB8DU, 0xD9BEBE67U, 0x4B393972U,
    0xDE4A4A94U, 0xD44C4C98U, 0xE85858B0U, 0x4ACFCF85U,
    0x6BD0D0BBU, 0x2AEFEFC5U, 0xE5AAAA4FU, 0x16FBFBEDU,
    0xC5434386U, 0xD74D4D9AU, 0x55333366U, 0x94858511U,
    0xCF45458AU, 0x10F9F9E9U, 0x06020204U, 0x817F7FFEU,
    0xF05050A0U, 0x443C3C78U, 0xBA9F9F25U, 0xE3A8A84BU,
    0xF35151A2U, 0xFEA3A35DU, 0xC0404080U, 0x8A8F8F05U,
    0xAD92923FU, 0xBC9D9D21U, 0x48383870U, 0x04F5F5F1U,
    0xDFBCBC63U, 0xC1B6B677U, 0x75DADAAFU, 0x63212142U,
    0x30101020U, 0x1AFFFFE5U, 0x0EF3F3FDU, 0x6DD2D2BFU,
    0x4CCDCD81U, 0x140C0C18U, 0x35131326U, 0x2FECECC3U,
    0xE15F5FBEU, 0xA2979735U, 0xCC444488U, 0x3917172EU,
    0x57C4C493U, 0xF2A7A755U, 0x827E7EFCU, 0x473D3D7AU,
    0xAC6464C8U, 0xE75D5DBAU, 0x2B191932U, 0x957373E6U,
    0xA06060C0U, 0x98818119U, 0xD14F4F9EU, 0x7FDCDCA3U,
    0x66222244U, 0x7E2A2A54U, 0xAB90903BU, 0x8388880BU,
    0xCA46468CU, 0x29EEEEC7U, 0xD3B8B86BU, 0x3C141428U,
    0x79DEDEA7U, 0xE25E5EBCU, 0x1D0B0B16U, 0x76DBDBADU,
    0x3BE0E0DBU, 0x56323264U, 0x4E3A3A74U, 0x1E0A0A14U,
    0xDB494992U, 0x0A06060CU, 0x6C242448U, 0xE45C5CB8U,
    0x5DC2C29FU, 0x6ED3D3BDU, 0xEFACAC43U, 0xA66262C4U,
    0xA8919139U, 0xA4959531U, 0x37E4E4D3U, 0x8B7979F2U,
    0x32E7E7D5U, 0x43C8C88BU, 0x5937376EU, 0xB76D6DDAU,
    0x8C8D8D01U, 0x64D5D5B1U, 0xD24E4E9CU, 0xE0A9A949U,
    0xB46C6CD8U, 0xFA5656ACU, 0x07F4F4F3U, 0x25EAEACFU,
    0xAF6565CAU, 0x8E7A7AF4U, 0xE9AEAE47U, 0x18080810U,
    0xD5BABA6FU, 0x887878F0U, 0x6F25254AU, 0x722E2E5CU,
    0x241C1C38U, 0xF1A6A657U, 0xC7B4B473U, 0x51C6C697U,
    0x23E8E8CBU, 0x7CDDDDA1U, 0x9C7474E8U, 0x211F1F3EU,
    0xDD4B4B96U, 0xDCBDBD61U, 0x868B8B0DU, 0x858A8A0FU,
    0x907070E0U, 0x423E3E7CU, 0xC4B5B571U, 0xAA6666CCU,
    0xD8484890U, 0x05030306U, 0x01F6F6F7U, 0x120E0E1CU,
    0xA36161C2U, 0x5F35356AU, 0xF95757AEU, 0xD0B9B969U,
    0x91868617U, 0x58C1C199U, 0x271D1D3AU, 0xB99E9E27U,
    0x38E1E1D9U, 0x13F8F8EBU, 0xB398982BU, 0x33111122U,
    0xBB6969D2U, 0x70D9D9A9U, 0x898E8E07U, 0xA7949433U,
    0xB69B9B2DU, 0x221E1E3CU, 0x92878715U, 0x20E9E9C9U,
    0x49CECE87U, 0xFF5555AAU, 0x78282850U, 0x7ADFDFA5U,
    0x8F8C8C03U, 0xF8A1A159U, 0x80898909U, 0x170D0D1AU,
    0xDABFBF65U, 0x31E6E6D7U, 0xC6424284U, 0xB86868D0U,
    0xC3414182U, 0xB0999929U, 0x772D2D5AU, 0x110F0F1EU,
    0xCBB0B07BU, 0xFC5454A8U, 0xD6BBBB6DU, 0x3A16162CU
};

#define BYTE(x, y) (xmrig_amd_bfe((x), (y) << 3U, 8U))

inline uint4 AES_Round_bittube2(const __local uint *AES0, const __local uint *AES1, uint4 x, uint4 k)
{
    x = ~x;
    k.s0 ^= AES0[BYTE(x.s0, 0)] ^ AES1[BYTE(x.s1, 1)] ^ rotate(AES0[BYTE(x.s2, 2)] ^ AES1[BYTE(x.s3, 3)], 16U);
    x.s0 ^= k.s0;
    k.s1 ^= AES0[BYTE(x.s1, 0)] ^ AES1[BYTE(x.s2, 1)] ^ rotate(AES0[BYTE(x.s3, 2)] ^ AES1[BYTE(x.s0, 3)], 16U);
    x.s1 ^= k.s1;
    k.s2 ^= AES0[BYTE(x.s2, 0)] ^ AES1[BYTE(x.s3, 1)] ^ rotate(AES0[BYTE(x.s0, 2)] ^ AES1[BYTE(x.s1, 3)], 16U);
    x.s2 ^= k.s2;
    k.s3 ^= AES0[BYTE(x.s3, 0)] ^ AES1[BYTE(x.s0, 1)] ^ rotate(AES0[BYTE(x.s1, 2)] ^ AES1[BYTE(x.s2, 3)], 16U);
    return k;
}

uint4 AES_Round(const __local uint *AES0, const __local uint *AES1, const __local uint *AES2, const __local uint *AES3, const uint4 X, uint4 key)
{
    key.s0 ^= AES0[BYTE(X.s0, 0)] ^ AES1[BYTE(X.s1, 1)] ^ AES2[BYTE(X.s2, 2)] ^ AES3[BYTE(X.s3, 3)];
    key.s1 ^= AES0[BYTE(X.s1, 0)] ^ AES1[BYTE(X.s2, 1)] ^ AES2[BYTE(X.s3, 2)] ^ AES3[BYTE(X.s0, 3)];
    key.s2 ^= AES0[BYTE(X.s2, 0)] ^ AES1[BYTE(X.s3, 1)] ^ AES2[BYTE(X.s0, 2)] ^ AES3[BYTE(X.s1, 3)];
    key.s3 ^= AES0[BYTE(X.s3, 0)] ^ AES1[BYTE(X.s0, 1)] ^ AES2[BYTE(X.s1, 2)] ^ AES3[BYTE(X.s2, 3)];

    return key;
}

uint4 AES_Round_Two_Tables(const __local uint *AES0, const __local uint *AES1, const uint4 X, uint4 key)
{
    key.s0 ^= AES0[BYTE(X.s0, 0)] ^ AES1[BYTE(X.s1, 1)] ^ rotate(AES0[BYTE(X.s2, 2)] ^ AES1[BYTE(X.s3, 3)], 16U);
    key.s1 ^= AES0[BYTE(X.s1, 0)] ^ AES1[BYTE(X.s2, 1)] ^ rotate(AES0[BYTE(X.s3, 2)] ^ AES1[BYTE(X.s0, 3)], 16U);
    key.s2 ^= AES0[BYTE(X.s2, 0)] ^ AES1[BYTE(X.s3, 1)] ^ rotate(AES0[BYTE(X.s0, 2)] ^ AES1[BYTE(X.s1, 3)], 16U);
    key.s3 ^= AES0[BYTE(X.s3, 0)] ^ AES1[BYTE(X.s0, 1)] ^ rotate(AES0[BYTE(X.s1, 2)] ^ AES1[BYTE(X.s2, 3)], 16U);

    return key;
}

#endif
