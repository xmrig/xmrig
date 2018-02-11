#pragma once

#include "hash.h"

HashReturn jh_hash(int hashbitlen, const BitSequence *data, DataLength databitlen, BitSequence *hashval);
