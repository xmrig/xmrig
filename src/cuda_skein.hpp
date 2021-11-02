#pragma once

typedef unsigned int    uint_t;             /* native unsigned integer */

#define SKEIN_MODIFIER_WORDS  ( 2)          /* number of modifier (tweak) words */

#define SKEIN_256_STATE_WORDS ( 4)
#define SKEIN_512_STATE_WORDS ( 8)
#define SKEIN1024_STATE_WORDS (16)

#define SKEIN_256_STATE_BYTES ( 8*SKEIN_256_STATE_WORDS)
#define SKEIN_512_STATE_BYTES ( 8*SKEIN_512_STATE_WORDS)
#define SKEIN1024_STATE_BYTES ( 8*SKEIN1024_STATE_WORDS)

#define SKEIN_256_STATE_BITS  (64*SKEIN_256_STATE_WORDS)
#define SKEIN_512_STATE_BITS  (64*SKEIN_512_STATE_WORDS)
#define SKEIN1024_STATE_BITS  (64*SKEIN1024_STATE_WORDS)

#define SKEIN_256_BLOCK_BYTES ( 8*SKEIN_256_STATE_WORDS)
#define SKEIN_512_BLOCK_BYTES ( 8*SKEIN_512_STATE_WORDS)
#define SKEIN1024_BLOCK_BYTES ( 8*SKEIN1024_STATE_WORDS)

#define SKEIN_MK_64(hi32,lo32)  ((lo32) + (((uint64_t) (hi32)) << 32))
#define SKEIN_KS_PARITY         SKEIN_MK_64(0x1BD11BDA,0xA9FC1A22)

#define SKEIN_T1_BIT(BIT)       ((BIT) - 64)            /* offset 64 because it's the second word  */

#define SKEIN_T1_POS_FIRST      SKEIN_T1_BIT(126)       /* bits 126     : first block flag         */
#define SKEIN_T1_POS_BIT_PAD    SKEIN_T1_BIT(119)       /* bit  119     : partial final input byte */
#define SKEIN_T1_POS_FINAL      SKEIN_T1_BIT(127)       /* bit  127     : final block flag         */
#define SKEIN_T1_POS_BLK_TYPE   SKEIN_T1_BIT(120)       /* bits 120..125: type field               */

#define SKEIN_T1_FLAG_FIRST     (((uint64_t)  1 ) << SKEIN_T1_POS_FIRST)
#define SKEIN_T1_FLAG_BIT_PAD   (((uint64_t)  1 ) << SKEIN_T1_POS_BIT_PAD)
#define SKEIN_T1_FLAG_FINAL     (((uint64_t)  1 ) << SKEIN_T1_POS_FINAL)

#define SKEIN_BLK_TYPE_MSG      (48)                    /* message processing */
#define SKEIN_BLK_TYPE_OUT      (63)                    /* output stage */

#define SKEIN_T1_BLK_TYPE(T)   (((uint64_t) (SKEIN_BLK_TYPE_##T)) << SKEIN_T1_POS_BLK_TYPE)

#define SKEIN_T1_BLK_TYPE_MSG   SKEIN_T1_BLK_TYPE(MSG)  /* message processing */
#define SKEIN_T1_BLK_TYPE_OUT   SKEIN_T1_BLK_TYPE(OUT)  /* output stage */

#define SKEIN_T1_BLK_TYPE_OUT_FINAL       (SKEIN_T1_BLK_TYPE_OUT | SKEIN_T1_FLAG_FINAL)

#define Skein_Set_Tweak(ctxPtr,TWK_NUM,tVal)    {(ctxPtr)->h.T[TWK_NUM] = (tVal);}

#define Skein_Set_T0(ctxPtr,T0) Skein_Set_Tweak(ctxPtr,0,T0)
#define Skein_Set_T1(ctxPtr,T1) Skein_Set_Tweak(ctxPtr,1,T1)

#define Skein_Set_T0_T1(ctxPtr,T0,T1) { \
  Skein_Set_T0(ctxPtr,(T0)); \
  Skein_Set_T1(ctxPtr,(T1)); }

#define Skein_Start_New_Type(ctxPtr,BLK_TYPE)   \
{ Skein_Set_T0_T1(ctxPtr,0,SKEIN_T1_FLAG_FIRST | SKEIN_T1_BLK_TYPE_##BLK_TYPE); (ctxPtr)->h.bCnt=0; }

#define Skein_Set_Bit_Pad_Flag(hdr)      { (hdr).T[1] |=  SKEIN_T1_FLAG_BIT_PAD;     }

#define KW_TWK_BASE     (0)
#define KW_KEY_BASE     (3)
#define ks              (kw + KW_KEY_BASE)
#define ts              (kw + KW_TWK_BASE)

#define R512(p0,p1,p2,p3,p4,p5,p6,p7,R512ROT,rNum) \
	X##p0 += X##p1; X##p1 = ROTL64(X##p1,R512ROT##_0); X##p1 ^= X##p0; \
	X##p2 += X##p3; X##p3 = ROTL64(X##p3,R512ROT##_1); X##p3 ^= X##p2; \
	X##p4 += X##p5; X##p5 = ROTL64(X##p5,R512ROT##_2); X##p5 ^= X##p4; \
	X##p6 += X##p7; X##p7 = ROTL64(X##p7,R512ROT##_3); X##p7 ^= X##p6;

#define I512(R) \
	X0   += ks[((R)+1) % 9]; \
	X1   += ks[((R)+2) % 9]; \
	X2   += ks[((R)+3) % 9]; \
	X3   += ks[((R)+4) % 9]; \
	X4   += ks[((R)+5) % 9]; \
	X5   += ks[((R)+6) % 9] + ts[((R)+1) % 3]; \
	X6   += ks[((R)+7) % 9] + ts[((R)+2) % 3]; \
	X7   += ks[((R)+8) % 9] + (R)+1;


#define R512_8_rounds(R) \
	R512(0,1,2,3,4,5,6,7,R_512_0,8*(R)+ 1); \
	R512(2,1,4,7,6,5,0,3,R_512_1,8*(R)+ 2); \
	R512(4,1,6,3,0,5,2,7,R_512_2,8*(R)+ 3); \
	R512(6,1,0,7,2,5,4,3,R_512_3,8*(R)+ 4); \
	I512(2*(R)); \
	R512(0,1,2,3,4,5,6,7,R_512_4,8*(R)+ 5); \
	R512(2,1,4,7,6,5,0,3,R_512_5,8*(R)+ 6); \
	R512(4,1,6,3,0,5,2,7,R_512_6,8*(R)+ 7); \
	R512(6,1,0,7,2,5,4,3,R_512_7,8*(R)+ 8); \
	I512(2*(R)+1);

typedef struct
{
	size_t  hashBitLen;
	size_t  bCnt;
	uint64_t  T[SKEIN_MODIFIER_WORDS];
} Skein_Ctxt_Hdr_t;

typedef struct {
	Skein_Ctxt_Hdr_t h;
	uint64_t  X[SKEIN_256_STATE_WORDS];
	uint8_t  b[SKEIN_256_BLOCK_BYTES];
} Skein_256_Ctxt_t;

typedef struct {
	Skein_Ctxt_Hdr_t h;
	uint64_t  X[SKEIN_512_STATE_WORDS];
	uint8_t  b[SKEIN_512_BLOCK_BYTES];
} Skein_512_Ctxt_t;

typedef struct {
	Skein_Ctxt_Hdr_t h;
	uint64_t  X[SKEIN1024_STATE_WORDS];
	uint8_t  b[SKEIN1024_BLOCK_BYTES];
} Skein1024_Ctxt_t;

typedef struct {
	uint_t  statebits;
	union {
		Skein_Ctxt_Hdr_t h;
		Skein_256_Ctxt_t ctx_256;
		Skein_512_Ctxt_t ctx_512;
		Skein1024_Ctxt_t ctx1024;
	} u;
} skeinHashState;

__device__ void cn_skein_init(skeinHashState *state, size_t hashBitLen)
{
	const uint64_t SKEIN_512_IV_256[] =
	{
		SKEIN_MK_64(0xCCD044A1,0x2FDB3E13),
		SKEIN_MK_64(0xE8359030,0x1A79A9EB),
		SKEIN_MK_64(0x55AEA061,0x4F816E6F),
		SKEIN_MK_64(0x2A2767A4,0xAE9B94DB),
		SKEIN_MK_64(0xEC06025E,0x74DD7683),
		SKEIN_MK_64(0xE7A436CD,0xC4746251),
		SKEIN_MK_64(0xC36FBAF9,0x393AD185),
		SKEIN_MK_64(0x3EEDBA18,0x33EDFC13)
	};

	Skein_512_Ctxt_t *ctx = &state->u.ctx_512;

	ctx->h.hashBitLen = hashBitLen;

	memcpy(ctx->X, SKEIN_512_IV_256, sizeof(ctx->X));

	Skein_Start_New_Type(ctx, MSG);
}

__device__ void cn_skein512_processblock(Skein_512_Ctxt_t * __restrict__ ctx, const uint8_t * __restrict__ blkPtr, size_t blkCnt, size_t byteCntAdd)
{
	enum {
		R_512_0_0=46, R_512_0_1=36, R_512_0_2=19, R_512_0_3=37,
		R_512_1_0=33, R_512_1_1=27, R_512_1_2=14, R_512_1_3=42,
		R_512_2_0=17, R_512_2_1=49, R_512_2_2=36, R_512_2_3=39,
		R_512_3_0=44, R_512_3_1= 9, R_512_3_2=54, R_512_3_3=56,
		R_512_4_0=39, R_512_4_1=30, R_512_4_2=34, R_512_4_3=24,
		R_512_5_0=13, R_512_5_1=50, R_512_5_2=10, R_512_5_3=17,
		R_512_6_0=25, R_512_6_1=29, R_512_6_2=39, R_512_6_3=43,
		R_512_7_0= 8, R_512_7_1=35, R_512_7_2=56, R_512_7_3=22
	};

	uint64_t X0,X1,X2,X3,X4,X5,X6,X7;
	uint64_t w[SKEIN_512_STATE_WORDS];
	uint64_t kw[SKEIN_512_STATE_WORDS+4];

	ts[0] = ctx->h.T[0];
	ts[1] = ctx->h.T[1];

	do
	{

		ts[0] += byteCntAdd;

		ks[0] = ctx->X[0];
		ks[1] = ctx->X[1];
		ks[2] = ctx->X[2];
		ks[3] = ctx->X[3];
		ks[4] = ctx->X[4];
		ks[5] = ctx->X[5];
		ks[6] = ctx->X[6];
		ks[7] = ctx->X[7];
		ks[8] = ks[0] ^ ks[1] ^ ks[2] ^ ks[3] ^
		ks[4] ^ ks[5] ^ ks[6] ^ ks[7] ^ SKEIN_KS_PARITY;

		ts[2] = ts[0] ^ ts[1];

		memcpy(w, blkPtr, SKEIN_512_STATE_WORDS << 3);

		X0 = w[0] + ks[0];
		X1 = w[1] + ks[1];
		X2 = w[2] + ks[2];
		X3 = w[3] + ks[3];
		X4 = w[4] + ks[4];
		X5 = w[5] + ks[5] + ts[0];
		X6 = w[6] + ks[6] + ts[1];
		X7 = w[7] + ks[7];

		blkPtr += SKEIN_512_BLOCK_BYTES;

		R512_8_rounds( 0);
		R512_8_rounds( 1);
		R512_8_rounds( 2);
		R512_8_rounds( 3);
		R512_8_rounds( 4);
		R512_8_rounds( 5);
		R512_8_rounds( 6);
		R512_8_rounds( 7);
		R512_8_rounds( 8);

		ctx->X[0] = X0 ^ w[0];
		ctx->X[1] = X1 ^ w[1];
		ctx->X[2] = X2 ^ w[2];
		ctx->X[3] = X3 ^ w[3];
		ctx->X[4] = X4 ^ w[4];
		ctx->X[5] = X5 ^ w[5];
		ctx->X[6] = X6 ^ w[6];
		ctx->X[7] = X7 ^ w[7];

		ts[1] &= ~SKEIN_T1_FLAG_FIRST;
	} 
	while (--blkCnt);

	ctx->h.T[0] = ts[0];
	ctx->h.T[1] = ts[1];
}

__device__ void cn_skein_final(skeinHashState * __restrict__ state, uint8_t * __restrict__ hashVal)
{
	size_t i,n,byteCnt;
	uint64_t X[SKEIN_512_STATE_WORDS];
	Skein_512_Ctxt_t *ctx = (Skein_512_Ctxt_t *)&state->u.ctx_512;
	//size_t tmp;
	//uint8_t *p8;
	//uint64_t *p64;

	ctx->h.T[1] |= SKEIN_T1_FLAG_FINAL;

	if (ctx->h.bCnt < SKEIN_512_BLOCK_BYTES) 
	{
		memset(&ctx->b[ctx->h.bCnt],0,SKEIN_512_BLOCK_BYTES - ctx->h.bCnt);
		//p8 = &ctx->b[ctx->h.bCnt];
		//tmp = SKEIN_512_BLOCK_BYTES - ctx->h.bCnt;
		//for( i = 0; i < tmp; i++ ) *(p8+i) = 0;
	}

	cn_skein512_processblock(ctx,ctx->b,1,ctx->h.bCnt);

	byteCnt = (ctx->h.hashBitLen + 7) >> 3;

	//uint8_t  b[SKEIN_512_BLOCK_BYTES] == 64
	memset(ctx->b,0,sizeof(ctx->b));
	//p64 = (uint64_t *)ctx->b;
	//for( i = 0; i < 8; i++ ) *(p64+i) = 0;

	memcpy(X,ctx->X,sizeof(X));

	for (i=0;i*SKEIN_512_BLOCK_BYTES < byteCnt;i++) 
	{
		((uint64_t *)ctx->b)[0]= (uint64_t)i;
		Skein_Start_New_Type(ctx,OUT_FINAL);
		cn_skein512_processblock(ctx,ctx->b,1,sizeof(uint64_t));
		n = byteCnt - i*SKEIN_512_BLOCK_BYTES;
		if (n >= SKEIN_512_BLOCK_BYTES)
		n  = SKEIN_512_BLOCK_BYTES;
		memcpy(hashVal+i*SKEIN_512_BLOCK_BYTES,ctx->X,n);
		memcpy(ctx->X,X,sizeof(X));   /* restore the counter mode key for next time */
	}
}

__device__ void cn_skein512_update(Skein_512_Ctxt_t * __restrict__ ctx, const uint8_t * __restrict__ msg, size_t msgByteCnt)
{
	size_t n;

	if (msgByteCnt + ctx->h.bCnt > SKEIN_512_BLOCK_BYTES) 
	{

		if (ctx->h.bCnt) 
		{

			n = SKEIN_512_BLOCK_BYTES - ctx->h.bCnt;

			if (n) 
			{
				memcpy(&ctx->b[ctx->h.bCnt],msg,n);
				msgByteCnt  -= n;
				msg         += n;
				ctx->h.bCnt += n;
			}

			cn_skein512_processblock(ctx,ctx->b,1,SKEIN_512_BLOCK_BYTES);
			ctx->h.bCnt = 0;
		}

		if (msgByteCnt > SKEIN_512_BLOCK_BYTES) 
		{
			n = (msgByteCnt-1) / SKEIN_512_BLOCK_BYTES;
			cn_skein512_processblock(ctx,msg,n,SKEIN_512_BLOCK_BYTES);
			msgByteCnt -= n * SKEIN_512_BLOCK_BYTES;
			msg        += n * SKEIN_512_BLOCK_BYTES;
		}
	}

	if (msgByteCnt) 
	{
		memcpy(&ctx->b[ctx->h.bCnt],msg,msgByteCnt);
		ctx->h.bCnt += msgByteCnt;
	}
}

__device__ void cn_skein_update(skeinHashState * __restrict__ state, const BitSequence * __restrict__ data, DataLength databitlen)
{
	if ((databitlen & 7) == 0) 
	{
		cn_skein512_update(&state->u.ctx_512,data,databitlen >> 3);
	}
	else 
	{

		size_t bCnt = (databitlen >> 3) + 1;
		uint8_t b,mask;

		mask = (uint8_t) (1u << (7 - (databitlen & 7)));
		b    = (uint8_t) ((data[bCnt-1] & (0-mask)) | mask);

		cn_skein512_update(&state->u.ctx_512,data,bCnt-1);
		cn_skein512_update(&state->u.ctx_512,&b  ,  1   );

		Skein_Set_Bit_Pad_Flag(state->u.h);
	}
}

__device__ void cn_skein(const BitSequence * __restrict__ data, DataLength len, BitSequence * __restrict__ hashval)
{
	int hashbitlen = 256;
	DataLength databitlen = len << 3;
	skeinHashState state;

	state.statebits = 64*SKEIN_512_STATE_WORDS;

	cn_skein_init(&state, hashbitlen);
	cn_skein_update(&state, data, databitlen);
	cn_skein_final(&state, hashval);
}
