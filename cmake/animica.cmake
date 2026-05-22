# Animica hashshare PoW (SHA3-256) algorithm family.
#
# The hashing is plain SHA3-256 over `prefix || nonce_le8` where `prefix` is
# the per-job header derived from the Stratum job, and `nonce` is the 64-bit
# little-endian counter the miner sweeps. No large memory dataset (cf. RandomX)
# and no per-block program (cf. KawPow), so this gate compiles a tiny amount
# of additional code and the binary only grows by a few KB.

if (WITH_ANIMICA)
    add_definitions(/DXMRIG_ALGO_ANIMICA)

    list(APPEND HEADERS_CRYPTO
        src/crypto/animica/AnimicaHash.h
        src/crypto/animica/AnimicaWorker.h
        src/crypto/animica/AnimicaAicfDelegate.h
    )

    list(APPEND SOURCES_CRYPTO
        src/crypto/animica/AnimicaHash.cpp
        src/crypto/animica/AnimicaWorker.cpp
        src/crypto/animica/AnimicaAicfDelegate.cpp
    )
else()
    remove_definitions(/DXMRIG_ALGO_ANIMICA)
endif()
