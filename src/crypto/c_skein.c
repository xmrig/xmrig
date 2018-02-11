#define  SKEIN_PORT_CODE 

#include <stddef.h>                          
#include <string.h>      
#include "c_skein.h"       

#ifndef SKEIN_512_NIST_MAX_HASHBITS
#define SKEIN_512_NIST_MAX_HASHBITS (512)
#endif

#define  SKEIN_MODIFIER_WORDS  ( 2)          

#define  SKEIN_512_STATE_WORDS ( 8)
#define  SKEIN_MAX_STATE_WORDS (16)

#define  SKEIN_512_STATE_BYTES ( 8*SKEIN_512_STATE_WORDS)
#define  SKEIN_512_STATE_BITS  (64*SKEIN_512_STATE_WORDS)
#define  SKEIN_512_BLOCK_BYTES ( 8*SKEIN_512_STATE_WORDS)

#define SKEIN_RND_SPECIAL       (1000u)
#define SKEIN_RND_KEY_INITIAL   (SKEIN_RND_SPECIAL+0u)
#define SKEIN_RND_KEY_INJECT    (SKEIN_RND_SPECIAL+1u)
#define SKEIN_RND_FEED_FWD      (SKEIN_RND_SPECIAL+2u)

typedef struct
{
  size_t  hashBitLen;                      
  size_t  bCnt;                            
  u64b_t  T[SKEIN_MODIFIER_WORDS];         
} Skein_Ctxt_Hdr_t;

typedef struct                               
{
  Skein_Ctxt_Hdr_t h;                      
  u64b_t  X[SKEIN_512_STATE_WORDS];        
  u08b_t  b[SKEIN_512_BLOCK_BYTES];        
} Skein_512_Ctxt_t;


static int  Skein_512_Init  (Skein_512_Ctxt_t *ctx, size_t hashBitLen);
static int  Skein_512_Update(Skein_512_Ctxt_t *ctx, const u08b_t *msg, size_t msgByteCnt);
static int  Skein_512_Final (Skein_512_Ctxt_t *ctx, u08b_t * hashVal);

#ifndef SKEIN_TREE_HASH
#define SKEIN_TREE_HASH (1)
#endif
#define SKEIN_T1_BIT(BIT)       ((BIT) - 64)            

#define SKEIN_T1_POS_TREE_LVL   SKEIN_T1_BIT(112)       
#define SKEIN_T1_POS_BIT_PAD    SKEIN_T1_BIT(119)       
#define SKEIN_T1_POS_BLK_TYPE   SKEIN_T1_BIT(120)       
#define SKEIN_T1_POS_FIRST      SKEIN_T1_BIT(126)       
#define SKEIN_T1_POS_FINAL      SKEIN_T1_BIT(127)       


#define SKEIN_T1_FLAG_FIRST     (((u64b_t)  1 ) << SKEIN_T1_POS_FIRST)
#define SKEIN_T1_FLAG_FINAL     (((u64b_t)  1 ) << SKEIN_T1_POS_FINAL)
#define SKEIN_T1_FLAG_BIT_PAD   (((u64b_t)  1 ) << SKEIN_T1_POS_BIT_PAD)


#define SKEIN_T1_TREE_LVL_MASK  (((u64b_t)0x7F) << SKEIN_T1_POS_TREE_LVL)
#define SKEIN_T1_TREE_LEVEL(n)  (((u64b_t) (n)) << SKEIN_T1_POS_TREE_LVL)


#define SKEIN_BLK_TYPE_KEY      ( 0)                    
#define SKEIN_BLK_TYPE_CFG      ( 4)                    
#define SKEIN_BLK_TYPE_PERS     ( 8)                    
#define SKEIN_BLK_TYPE_PK       (12)                    
#define SKEIN_BLK_TYPE_KDF      (16)                    
#define SKEIN_BLK_TYPE_NONCE    (20)                    
#define SKEIN_BLK_TYPE_MSG      (48)                    
#define SKEIN_BLK_TYPE_OUT      (63)                    
#define SKEIN_BLK_TYPE_MASK     (63)                    

#define SKEIN_T1_BLK_TYPE(T)   (((u64b_t) (SKEIN_BLK_TYPE_##T)) << SKEIN_T1_POS_BLK_TYPE)
#define SKEIN_T1_BLK_TYPE_KEY   SKEIN_T1_BLK_TYPE(KEY)  
#define SKEIN_T1_BLK_TYPE_CFG   SKEIN_T1_BLK_TYPE(CFG)  
#define SKEIN_T1_BLK_TYPE_PERS  SKEIN_T1_BLK_TYPE(PERS) 
#define SKEIN_T1_BLK_TYPE_PK    SKEIN_T1_BLK_TYPE(PK)   
#define SKEIN_T1_BLK_TYPE_KDF   SKEIN_T1_BLK_TYPE(KDF)  
#define SKEIN_T1_BLK_TYPE_NONCE SKEIN_T1_BLK_TYPE(NONCE)
#define SKEIN_T1_BLK_TYPE_MSG   SKEIN_T1_BLK_TYPE(MSG)  
#define SKEIN_T1_BLK_TYPE_OUT   SKEIN_T1_BLK_TYPE(OUT)  
#define SKEIN_T1_BLK_TYPE_MASK  SKEIN_T1_BLK_TYPE(MASK) 

#define SKEIN_T1_BLK_TYPE_CFG_FINAL       (SKEIN_T1_BLK_TYPE_CFG | SKEIN_T1_FLAG_FINAL)
#define SKEIN_T1_BLK_TYPE_OUT_FINAL       (SKEIN_T1_BLK_TYPE_OUT | SKEIN_T1_FLAG_FINAL)

#define SKEIN_VERSION           (1)

#ifndef SKEIN_ID_STRING_LE      
#define SKEIN_ID_STRING_LE      (0x33414853)            /* "SHA3" (little-endian)*/
#endif

#define SKEIN_MK_64(hi32,lo32)  ((lo32) + (((u64b_t) (hi32)) << 32))
#define SKEIN_SCHEMA_VER        SKEIN_MK_64(SKEIN_VERSION,SKEIN_ID_STRING_LE)
#define SKEIN_KS_PARITY         SKEIN_MK_64(0x1BD11BDA,0xA9FC1A22)

#define SKEIN_CFG_STR_LEN       (4*8)


#define SKEIN_CFG_TREE_LEAF_SIZE_POS  ( 0)
#define SKEIN_CFG_TREE_NODE_SIZE_POS  ( 8)
#define SKEIN_CFG_TREE_MAX_LEVEL_POS  (16)

#define SKEIN_CFG_TREE_LEAF_SIZE_MSK  (((u64b_t) 0xFF) << SKEIN_CFG_TREE_LEAF_SIZE_POS)
#define SKEIN_CFG_TREE_NODE_SIZE_MSK  (((u64b_t) 0xFF) << SKEIN_CFG_TREE_NODE_SIZE_POS)
#define SKEIN_CFG_TREE_MAX_LEVEL_MSK  (((u64b_t) 0xFF) << SKEIN_CFG_TREE_MAX_LEVEL_POS)

#define SKEIN_CFG_TREE_INFO(leaf,node,maxLvl)                   \
  ( (((u64b_t)(leaf  )) << SKEIN_CFG_TREE_LEAF_SIZE_POS) |    \
  (((u64b_t)(node  )) << SKEIN_CFG_TREE_NODE_SIZE_POS) |    \
  (((u64b_t)(maxLvl)) << SKEIN_CFG_TREE_MAX_LEVEL_POS) )

#define SKEIN_CFG_TREE_INFO_SEQUENTIAL SKEIN_CFG_TREE_INFO(0,0,0) 
#define Skein_Get_Tweak(ctxPtr,TWK_NUM)         ((ctxPtr)->h.T[TWK_NUM])
#define Skein_Set_Tweak(ctxPtr,TWK_NUM,tVal)    {(ctxPtr)->h.T[TWK_NUM] = (tVal);}

#define Skein_Get_T0(ctxPtr)    Skein_Get_Tweak(ctxPtr,0)
#define Skein_Get_T1(ctxPtr)    Skein_Get_Tweak(ctxPtr,1)
#define Skein_Set_T0(ctxPtr,T0) Skein_Set_Tweak(ctxPtr,0,T0)
#define Skein_Set_T1(ctxPtr,T1) Skein_Set_Tweak(ctxPtr,1,T1)


#define Skein_Set_T0_T1(ctxPtr,T0,T1)           \
{                                           \
  Skein_Set_T0(ctxPtr,(T0));                  \
  Skein_Set_T1(ctxPtr,(T1));                  \
}

#define Skein_Set_Type(ctxPtr,BLK_TYPE)         \
  Skein_Set_T1(ctxPtr,SKEIN_T1_BLK_TYPE_##BLK_TYPE)


#define Skein_Start_New_Type(ctxPtr,BLK_TYPE)   \
{ Skein_Set_T0_T1(ctxPtr,0,SKEIN_T1_FLAG_FIRST | SKEIN_T1_BLK_TYPE_##BLK_TYPE); (ctxPtr)->h.bCnt=0; }

#define Skein_Clear_First_Flag(hdr)      { (hdr).T[1] &= ~SKEIN_T1_FLAG_FIRST;       }
#define Skein_Set_Bit_Pad_Flag(hdr)      { (hdr).T[1] |=  SKEIN_T1_FLAG_BIT_PAD;     }

#define Skein_Set_Tree_Level(hdr,height) { (hdr).T[1] |= SKEIN_T1_TREE_LEVEL(height);}
#define Skein_Show_Block(bits,ctx,X,blkPtr,wPtr,ksEvenPtr,ksOddPtr)
#define Skein_Show_Round(bits,ctx,r,X)
#define Skein_Show_R_Ptr(bits,ctx,r,X_ptr)
#define Skein_Show_Final(bits,ctx,cnt,outPtr)
#define Skein_Show_Key(bits,ctx,key,keyBytes)


#ifndef SKEIN_ERR_CHECK        
#define Skein_Assert(x,retCode)
#define Skein_assert(x)
#elif   defined(SKEIN_ASSERT)
#include <assert.h>     
#define Skein_Assert(x,retCode) assert(x) 
#define Skein_assert(x)         assert(x) 
#else
#include <assert.h>     
#define Skein_Assert(x,retCode) { if (!(x)) return retCode; } 
#define Skein_assert(x)         assert(x)                     
#endif
enum    
{   
  
  R_512_0_0=46, R_512_0_1=36, R_512_0_2=19, R_512_0_3=37,
  R_512_1_0=33, R_512_1_1=27, R_512_1_2=14, R_512_1_3=42,
  R_512_2_0=17, R_512_2_1=49, R_512_2_2=36, R_512_2_3=39,
  R_512_3_0=44, R_512_3_1= 9, R_512_3_2=54, R_512_3_3=56,
  R_512_4_0=39, R_512_4_1=30, R_512_4_2=34, R_512_4_3=24,
  R_512_5_0=13, R_512_5_1=50, R_512_5_2=10, R_512_5_3=17,
  R_512_6_0=25, R_512_6_1=29, R_512_6_2=39, R_512_6_3=43,
  R_512_7_0= 8, R_512_7_1=35, R_512_7_2=56, R_512_7_3=22,
};

#ifndef SKEIN_ROUNDS
#define SKEIN_512_ROUNDS_TOTAL (72)
#else                                        
#define SKEIN_512_ROUNDS_TOTAL (8*((((SKEIN_ROUNDS/ 10) + 5) % 10) + 5))
#endif

#define MK_64 SKEIN_MK_64


const u64b_t SKEIN_512_IV_256[] =
    {
    MK_64(0xCCD044A1,0x2FDB3E13),
    MK_64(0xE8359030,0x1A79A9EB),
    MK_64(0x55AEA061,0x4F816E6F),
    MK_64(0x2A2767A4,0xAE9B94DB),
    MK_64(0xEC06025E,0x74DD7683),
    MK_64(0xE7A436CD,0xC4746251),
    MK_64(0xC36FBAF9,0x393AD185),
    MK_64(0x3EEDBA18,0x33EDFC13)
    };

#ifndef SKEIN_USE_ASM
#define SKEIN_USE_ASM   (0)                     
#endif

#ifndef SKEIN_LOOP
#define SKEIN_LOOP 001                          
#endif

#define BLK_BITS        (WCNT*64)               
#define KW_TWK_BASE     (0)
#define KW_KEY_BASE     (3)
#define ks              (kw + KW_KEY_BASE)                
#define ts              (kw + KW_TWK_BASE)

#ifdef SKEIN_DEBUG
#define DebugSaveTweak(ctx) { ctx->h.T[0] = ts[0]; ctx->h.T[1] = ts[1]; }
#else
#define DebugSaveTweak(ctx)
#endif

#if !(SKEIN_USE_ASM & 512)
static void Skein_512_Process_Block(Skein_512_Ctxt_t *ctx,const u08b_t *blkPtr,size_t blkCnt,size_t byteCntAdd)
    { 
    enum
        {
        WCNT = SKEIN_512_STATE_WORDS
        };
#undef  RCNT
#define RCNT  (SKEIN_512_ROUNDS_TOTAL/8)

#ifdef  SKEIN_LOOP                              
#define SKEIN_UNROLL_512 (((SKEIN_LOOP)/10)%10)
#else
#define SKEIN_UNROLL_512 (0)
#endif

#if SKEIN_UNROLL_512
#if (RCNT % SKEIN_UNROLL_512)
#error "Invalid SKEIN_UNROLL_512"               
#endif
    size_t  r;
    u64b_t  kw[WCNT+4+RCNT*2];                  /* key schedule words : chaining vars + tweak + "rotation"*/
#else
    u64b_t  kw[WCNT+4];                         
#endif
    u64b_t  X0,X1,X2,X3,X4,X5,X6,X7;            
    u64b_t  w [WCNT];                           
#ifdef SKEIN_DEBUG
    const u64b_t *Xptr[8];                      
    Xptr[0] = &X0;  Xptr[1] = &X1;  Xptr[2] = &X2;  Xptr[3] = &X3;
    Xptr[4] = &X4;  Xptr[5] = &X5;  Xptr[6] = &X6;  Xptr[7] = &X7;
#endif

    Skein_assert(blkCnt != 0);                  
    ts[0] = ctx->h.T[0];
    ts[1] = ctx->h.T[1];
    do  {
        
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

        Skein_Get64_LSB_First(w,blkPtr,WCNT); 
        DebugSaveTweak(ctx);
        Skein_Show_Block(BLK_BITS,&ctx->h,ctx->X,blkPtr,w,ks,ts);

        X0   = w[0] + ks[0];                    
        X1   = w[1] + ks[1];
        X2   = w[2] + ks[2];
        X3   = w[3] + ks[3];
        X4   = w[4] + ks[4];
        X5   = w[5] + ks[5] + ts[0];
        X6   = w[6] + ks[6] + ts[1];
        X7   = w[7] + ks[7];

        blkPtr += SKEIN_512_BLOCK_BYTES;

        Skein_Show_R_Ptr(BLK_BITS,&ctx->h,SKEIN_RND_KEY_INITIAL,Xptr);
        
#define Round512(p0,p1,p2,p3,p4,p5,p6,p7,ROT,rNum)                  \
    X##p0 += X##p1; X##p1 = RotL_64(X##p1,ROT##_0); X##p1 ^= X##p0; \
    X##p2 += X##p3; X##p3 = RotL_64(X##p3,ROT##_1); X##p3 ^= X##p2; \
    X##p4 += X##p5; X##p5 = RotL_64(X##p5,ROT##_2); X##p5 ^= X##p4; \
    X##p6 += X##p7; X##p7 = RotL_64(X##p7,ROT##_3); X##p7 ^= X##p6; \

#if SKEIN_UNROLL_512 == 0                       
#define R512(p0,p1,p2,p3,p4,p5,p6,p7,ROT,rNum)        \
    Round512(p0,p1,p2,p3,p4,p5,p6,p7,ROT,rNum)                      \
    Skein_Show_R_Ptr(BLK_BITS,&ctx->h,rNum,Xptr);

#define I512(R)                                                     \
    X0   += ks[((R)+1) % 9];     \
    X1   += ks[((R)+2) % 9];                                        \
    X2   += ks[((R)+3) % 9];                                        \
    X3   += ks[((R)+4) % 9];                                        \
    X4   += ks[((R)+5) % 9];                                        \
    X5   += ks[((R)+6) % 9] + ts[((R)+1) % 3];                      \
    X6   += ks[((R)+7) % 9] + ts[((R)+2) % 3];                      \
    X7   += ks[((R)+8) % 9] +     (R)+1;                            \
    Skein_Show_R_Ptr(BLK_BITS,&ctx->h,SKEIN_RND_KEY_INJECT,Xptr);
#else                                       
#define R512(p0,p1,p2,p3,p4,p5,p6,p7,ROT,rNum)                      \
    Round512(p0,p1,p2,p3,p4,p5,p6,p7,ROT,rNum)                      \
    Skein_Show_R_Ptr(BLK_BITS,&ctx->h,4*(r-1)+rNum,Xptr);

#define I512(R)                                                     \
    X0   += ks[r+(R)+0];         \
    X1   += ks[r+(R)+1];                                            \
    X2   += ks[r+(R)+2];                                            \
    X3   += ks[r+(R)+3];                                            \
    X4   += ks[r+(R)+4];                                            \
    X5   += ks[r+(R)+5] + ts[r+(R)+0];                              \
    X6   += ks[r+(R)+6] + ts[r+(R)+1];                              \
    X7   += ks[r+(R)+7] +    r+(R)   ;                              \
    ks[r +       (R)+8] = ks[r+(R)-1];     \
    ts[r +       (R)+2] = ts[r+(R)-1];                              \
    Skein_Show_R_Ptr(BLK_BITS,&ctx->h,SKEIN_RND_KEY_INJECT,Xptr);

    for (r=1;r < 2*RCNT;r+=2*SKEIN_UNROLL_512)   
#endif                         
        {
#define R512_8_rounds(R)    \
        R512(0,1,2,3,4,5,6,7,R_512_0,8*(R)+ 1);   \
        R512(2,1,4,7,6,5,0,3,R_512_1,8*(R)+ 2);   \
        R512(4,1,6,3,0,5,2,7,R_512_2,8*(R)+ 3);   \
        R512(6,1,0,7,2,5,4,3,R_512_3,8*(R)+ 4);   \
        I512(2*(R));                              \
        R512(0,1,2,3,4,5,6,7,R_512_4,8*(R)+ 5);   \
        R512(2,1,4,7,6,5,0,3,R_512_5,8*(R)+ 6);   \
        R512(4,1,6,3,0,5,2,7,R_512_6,8*(R)+ 7);   \
        R512(6,1,0,7,2,5,4,3,R_512_7,8*(R)+ 8);   \
        I512(2*(R)+1);        

        R512_8_rounds( 0);

#define R512_Unroll_R(NN) ((SKEIN_UNROLL_512 == 0 && SKEIN_512_ROUNDS_TOTAL/8 > (NN)) || (SKEIN_UNROLL_512 > (NN)))

  #if   R512_Unroll_R( 1)
        R512_8_rounds( 1);
  #endif
  #if   R512_Unroll_R( 2)
        R512_8_rounds( 2);
  #endif
  #if   R512_Unroll_R( 3)
        R512_8_rounds( 3);
  #endif
  #if   R512_Unroll_R( 4)
        R512_8_rounds( 4);
  #endif
  #if   R512_Unroll_R( 5)
        R512_8_rounds( 5);
  #endif
  #if   R512_Unroll_R( 6)
        R512_8_rounds( 6);
  #endif
  #if   R512_Unroll_R( 7)
        R512_8_rounds( 7);
  #endif
  #if   R512_Unroll_R( 8)
        R512_8_rounds( 8);
  #endif
  #if   R512_Unroll_R( 9)
        R512_8_rounds( 9);
  #endif
  #if   R512_Unroll_R(10)
        R512_8_rounds(10);
  #endif
  #if   R512_Unroll_R(11)
        R512_8_rounds(11);
  #endif
  #if   R512_Unroll_R(12)
        R512_8_rounds(12);
  #endif
  #if   R512_Unroll_R(13)
        R512_8_rounds(13);
  #endif
  #if   R512_Unroll_R(14)
        R512_8_rounds(14);
  #endif
  #if  (SKEIN_UNROLL_512 > 14)
#error  "need more unrolling in Skein_512_Process_Block"
  #endif
        }

        
        ctx->X[0] = X0 ^ w[0];
        ctx->X[1] = X1 ^ w[1];
        ctx->X[2] = X2 ^ w[2];
        ctx->X[3] = X3 ^ w[3];
        ctx->X[4] = X4 ^ w[4];
        ctx->X[5] = X5 ^ w[5];
        ctx->X[6] = X6 ^ w[6];
        ctx->X[7] = X7 ^ w[7];
        Skein_Show_Round(BLK_BITS,&ctx->h,SKEIN_RND_FEED_FWD,ctx->X);

        ts[1] &= ~SKEIN_T1_FLAG_FIRST;
        }
    while (--blkCnt);
    ctx->h.T[0] = ts[0];
    ctx->h.T[1] = ts[1];
    }
#endif
static int Skein_512_Init(Skein_512_Ctxt_t *ctx, size_t hashBitLen)
    {
    union
        {
        u08b_t  b[SKEIN_512_STATE_BYTES];
        u64b_t  w[SKEIN_512_STATE_WORDS];
        } cfg;                              
        
    Skein_Assert(hashBitLen > 0,SKEIN_BAD_HASHLEN);
    ctx->h.hashBitLen = hashBitLen;         

    switch (hashBitLen)
        {             
#ifndef SKEIN_NO_PRECOMP
        case  256: memcpy(ctx->X,SKEIN_512_IV_256,sizeof(ctx->X));  break;
#endif
        default:
            
            
            Skein_Start_New_Type(ctx,CFG_FINAL);        

            cfg.w[0] = Skein_Swap64(SKEIN_SCHEMA_VER);  
            cfg.w[1] = Skein_Swap64(hashBitLen);        
            cfg.w[2] = Skein_Swap64(SKEIN_CFG_TREE_INFO_SEQUENTIAL);
            memset(&cfg.w[3],0,sizeof(cfg) - 3*sizeof(cfg.w[0])); 

            
            memset(ctx->X,0,sizeof(ctx->X));            
            Skein_512_Process_Block(ctx,cfg.b,1,SKEIN_CFG_STR_LEN);
            break;
        }

    
    
    Skein_Start_New_Type(ctx,MSG);              

    return SKEIN_SUCCESS;
    }
static int Skein_512_Update(Skein_512_Ctxt_t *ctx, const u08b_t *msg, size_t msgByteCnt)
    {
    size_t n;

    Skein_Assert(ctx->h.bCnt <= SKEIN_512_BLOCK_BYTES,SKEIN_FAIL);    

    
    if (msgByteCnt + ctx->h.bCnt > SKEIN_512_BLOCK_BYTES)
        {
        if (ctx->h.bCnt)                              
            {
            n = SKEIN_512_BLOCK_BYTES - ctx->h.bCnt;  
            if (n)
                {
                Skein_assert(n < msgByteCnt);         
                memcpy(&ctx->b[ctx->h.bCnt],msg,n);
                msgByteCnt  -= n;
                msg         += n;
                ctx->h.bCnt += n;
                }
            Skein_assert(ctx->h.bCnt == SKEIN_512_BLOCK_BYTES);
            Skein_512_Process_Block(ctx,ctx->b,1,SKEIN_512_BLOCK_BYTES);
            ctx->h.bCnt = 0;
            }
        
        if (msgByteCnt > SKEIN_512_BLOCK_BYTES)
            {
            n = (msgByteCnt-1) / SKEIN_512_BLOCK_BYTES;   
            Skein_512_Process_Block(ctx,msg,n,SKEIN_512_BLOCK_BYTES);
            msgByteCnt -= n * SKEIN_512_BLOCK_BYTES;
            msg        += n * SKEIN_512_BLOCK_BYTES;
            }
        Skein_assert(ctx->h.bCnt == 0);
        }

    
    if (msgByteCnt)
        {
        Skein_assert(msgByteCnt + ctx->h.bCnt <= SKEIN_512_BLOCK_BYTES);
        memcpy(&ctx->b[ctx->h.bCnt],msg,msgByteCnt);
        ctx->h.bCnt += msgByteCnt;
        }

    return SKEIN_SUCCESS;
    }
static int Skein_512_Final(Skein_512_Ctxt_t *ctx, u08b_t *hashVal)
    {
    size_t i,n,byteCnt;
    u64b_t X[SKEIN_512_STATE_WORDS];
    Skein_Assert(ctx->h.bCnt <= SKEIN_512_BLOCK_BYTES,SKEIN_FAIL);    

    ctx->h.T[1] |= SKEIN_T1_FLAG_FINAL;                 
    if (ctx->h.bCnt < SKEIN_512_BLOCK_BYTES)            
        memset(&ctx->b[ctx->h.bCnt],0,SKEIN_512_BLOCK_BYTES - ctx->h.bCnt);

    Skein_512_Process_Block(ctx,ctx->b,1,ctx->h.bCnt);  
    
    
    byteCnt = (ctx->h.hashBitLen + 7) >> 3;             

    
    memset(ctx->b,0,sizeof(ctx->b));  
    memcpy(X,ctx->X,sizeof(X));       
    for (i=0;i*SKEIN_512_BLOCK_BYTES < byteCnt;i++)
        {
        ((u64b_t *)ctx->b)[0]= Skein_Swap64((u64b_t) i); 
        Skein_Start_New_Type(ctx,OUT_FINAL);
        Skein_512_Process_Block(ctx,ctx->b,1,sizeof(u64b_t)); 
        n = byteCnt - i*SKEIN_512_BLOCK_BYTES;   
        if (n >= SKEIN_512_BLOCK_BYTES)
            n  = SKEIN_512_BLOCK_BYTES;
        Skein_Put64_LSB_First(hashVal+i*SKEIN_512_BLOCK_BYTES,ctx->X,n);   
        Skein_Show_Final(512,&ctx->h,n,hashVal+i*SKEIN_512_BLOCK_BYTES);
        memcpy(ctx->X,X,sizeof(X));   
        }
    return SKEIN_SUCCESS;
    }

#if defined(SKEIN_CODE_SIZE) || defined(SKEIN_PERF)
static size_t Skein_512_API_CodeSize(void)
    {
    return ((u08b_t *) Skein_512_API_CodeSize) -
           ((u08b_t *) Skein_512_Init);
    }
#endif

typedef struct
{
  uint_t  statebits;                      
  union
  {
    Skein_Ctxt_Hdr_t h;                 
    Skein_512_Ctxt_t ctx_512;
  } u;
}
hashState;


static SkeinHashReturn Init  (hashState *state, int hashbitlen);
static SkeinHashReturn Update(hashState *state, const SkeinBitSequence *data, SkeinDataLength databitlen);
static SkeinHashReturn Final (hashState *state,       SkeinBitSequence *hashval);

/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/

static SkeinHashReturn Init(hashState *state, int hashbitlen)
{
    state->statebits = 64*SKEIN_512_STATE_WORDS;
    return Skein_512_Init(&state->u.ctx_512,(size_t) hashbitlen);
}

/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/

static SkeinHashReturn Update(hashState *state, const SkeinBitSequence *data, SkeinDataLength databitlen)
{
  
  Skein_Assert((state->u.h.T[1] & SKEIN_T1_FLAG_BIT_PAD) == 0 || databitlen == 0, SKEIN_FAIL);

  Skein_Assert(state->statebits % 256 == 0 && (state->statebits-256) < 1024,SKEIN_FAIL);
  if ((databitlen & 7) == 0)  
  {
    return Skein_512_Update(&state->u.ctx_512,data,databitlen >> 3);
  }
  else
  {   
    size_t bCnt = (databitlen >> 3) + 1;                  
    u08b_t b,mask;

    mask = (u08b_t) (1u << (7 - (databitlen & 7)));       
    b    = (u08b_t) ((data[bCnt-1] & (0-mask)) | mask);   

    Skein_512_Update(&state->u.ctx_512,data,bCnt-1); 
    Skein_512_Update(&state->u.ctx_512,&b  ,  1   ); 
    Skein_Set_Bit_Pad_Flag(state->u.h);                    

    return SKEIN_SUCCESS;
  }
}

/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/

static SkeinHashReturn Final(hashState *state, SkeinBitSequence *hashval)
{
  Skein_Assert(state->statebits % 256 == 0 && (state->statebits-256) < 1024,FAIL);
  return Skein_512_Final(&state->u.ctx_512,hashval);
}

/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/

SkeinHashReturn skein_hash(int hashbitlen, const SkeinBitSequence *data, 
                SkeinDataLength databitlen,SkeinBitSequence *hashval)
{
  hashState  state;
  SkeinHashReturn r = Init(&state,hashbitlen);
  if (r == SKEIN_SUCCESS)
  { 
    r = Update(&state,data,databitlen);
    Final(&state,hashval);
  }
  return r;
}

void xmr_skein(const SkeinBitSequence *data, SkeinBitSequence *hashval){
  #define XMR_HASHBITLEN 256
  #define XMR_DATABITLEN 1600

  // Init
  hashState  state;
  state.statebits = 64*SKEIN_512_STATE_WORDS;

  // Skein_512_Init(&state.u.ctx_512, (size_t)XMR_HASHBITLEN);
  state.u.ctx_512.h.hashBitLen = XMR_HASHBITLEN;
  memcpy(state.u.ctx_512.X,SKEIN_512_IV_256,sizeof(state.u.ctx_512.X));
  Skein_512_Ctxt_t* ctx = &(state.u.ctx_512);
  Skein_Start_New_Type(ctx,MSG);

  // Update
  if ((XMR_DATABITLEN & 7) == 0){  
    Skein_512_Update(&state.u.ctx_512,data,XMR_DATABITLEN >> 3);
  }else{   
    size_t bCnt = (XMR_DATABITLEN >> 3) + 1;                  
    u08b_t b,mask;

    mask = (u08b_t) (1u << (7 - (XMR_DATABITLEN & 7)));       
    b    = (u08b_t) ((data[bCnt-1] & (0-mask)) | mask);   

    Skein_512_Update(&state.u.ctx_512,data,bCnt-1); 
    Skein_512_Update(&state.u.ctx_512,&b  ,  1   ); 
    Skein_Set_Bit_Pad_Flag(state.u.h);                    
  }

  // Finalize
  Skein_512_Final(&state.u.ctx_512, hashval);
}
