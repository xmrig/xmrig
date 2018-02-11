#ifndef __SUBMITRESULT_H__
#define __SUBMITRESULT_H__
#include <uv.h>

class SubmitResult
{
public:
    inline SubmitResult() : seq(0), diff(0), actualDiff(0), elapsed(0), start(0) {}
    SubmitResult(int64_t seq, uint32_t diff, uint64_t actualDiff);

    void done();

    int64_t seq;
    uint32_t diff;
    uint64_t actualDiff;
    uint64_t elapsed;

private:
    uint64_t start;
};

#endif