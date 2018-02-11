#include <uv.h>
#include "net/SubmitResult.h"
SubmitResult::SubmitResult(int64_t seq, uint32_t diff, uint64_t actualDiff) :
    seq(seq),
    diff(diff),
    actualDiff(actualDiff),
    elapsed(0)
{
    start = uv_hrtime();
}
void SubmitResult::done() { elapsed = (uv_hrtime() - start) / 1000000; }