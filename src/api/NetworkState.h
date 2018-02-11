#ifndef __NETWORKSTATE_H__
#define __NETWORKSTATE_H__


#include <array>
#include <vector>


class SubmitResult;


class NetworkState
{
public:
    NetworkState();

    int connectionTime() const;
    uint32_t avgTime() const;
    uint32_t latency() const;
    void add(const SubmitResult &result, const char *error);
    void setPool(const char *host, int port, const char *ip);
    void stop();

    char pool[256];
    std::array<uint64_t, 10> topDiff { { } };
    uint32_t diff;
    uint64_t accepted;
    uint64_t failures;
    uint64_t rejected;
    uint64_t total;

private:
    bool m_active;
    std::vector<uint16_t> m_latency;
    uint64_t m_connectionTime;
};

#endif /* __NETWORKSTATE_H__ */
