#ifndef __HASHRATE_H__
#define __HASHRATE_H__
#include <stdint.h>
#include <uv.h>
class Hashrate
{
public:
    enum Intervals {
        ShortInterval  = 2500,
        MediumInterval = 60000,
        LargeInterval  = 900000
    };

    Hashrate(int threads);
    double calc(size_t ms) const;
    double calc(size_t threadId, size_t ms) const;
    void add(size_t threadId, uint64_t count, uint64_t timestamp);
    void print();
    void stop();
    void updateHighest();

    inline double highest() const { return m_highest; }
    inline int threads() const    { return m_threads; }

private:
    static void onReport(uv_timer_t *handle);

    constexpr static size_t kBucketSize = 2 << 11;
    constexpr static size_t kBucketMask = kBucketSize - 1;

    double m_highest;
    int m_threads;
    uint32_t* m_top;
    uint64_t** m_counts;
    uint64_t** m_timestamps;
    uv_timer_t m_timer;
};


#endif /* __HASHRATE_H__ */
