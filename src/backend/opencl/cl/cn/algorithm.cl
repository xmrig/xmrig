enum Algorithm {
    ALGO_INVALID = -1,
    ALGO_CN_0,          // "cn/0"             CryptoNight (original).
    ALGO_CN_1,          // "cn/1"             CryptoNight variant 1 also known as Monero7 and CryptoNightV7.
    ALGO_CN_2,          // "cn/2"             CryptoNight variant 2.
    ALGO_CN_R,          // "cn/r"             CryptoNightR (Monero's variant 4).
    ALGO_CN_FAST,       // "cn/fast"          CryptoNight variant 1 with half iterations.
    ALGO_CN_HALF,       // "cn/half"          CryptoNight variant 2 with half iterations (Masari/Torque).
    ALGO_CN_XAO,        // "cn/xao"           CryptoNight variant 0 (modified, Alloy only).
    ALGO_CN_RTO,        // "cn/rto"           CryptoNight variant 1 (modified, Arto only).
    ALGO_CN_RWZ,        // "cn/rwz"           CryptoNight variant 2 with 3/4 iterations and reversed shuffle operation (Graft).
    ALGO_CN_ZLS,        // "cn/zls"           CryptoNight variant 2 with 3/4 iterations (Zelerius).
    ALGO_CN_DOUBLE,     // "cn/double"        CryptoNight variant 2 with double iterations (X-CASH).
    ALGO_CN_GPU,        // "cn/gpu"           CryptoNight-GPU (Ryo).
    ALGO_CN_LITE_0,     // "cn-lite/0"        CryptoNight-Lite variant 0.
    ALGO_CN_LITE_1,     // "cn-lite/1"        CryptoNight-Lite variant 1.
    ALGO_CN_HEAVY_0,    // "cn-heavy/0"       CryptoNight-Heavy (4 MB).
    ALGO_CN_HEAVY_TUBE, // "cn-heavy/tube"    CryptoNight-Heavy (modified, TUBE only).
    ALGO_CN_HEAVY_XHV,  // "cn-heavy/xhv"     CryptoNight-Heavy (modified, Haven Protocol only).
    ALGO_CN_PICO_0,     // "cn-pico"          CryptoNight Turtle (TRTL)
    ALGO_RX_0,          // "rx/0"             RandomX (reference configuration).
    ALGO_RX_WOW,        // "rx/wow"           RandomWOW (Wownero).
    ALGO_RX_LOKI,       // "rx/loki"          RandomXL (Loki).
    ALGO_AR2_CHUKWA,    // "argon2/chukwa"    Argon2id (Chukwa).
    ALGO_AR2_WRKZ,      // "argon2/wrkz"      Argon2id (WRKZ)
    ALGO_MAX
};
