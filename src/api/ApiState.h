#ifndef __APISTATE_H__
#define __APISTATE_H__
#include "api/NetworkState.h"
#include "rapidjson/fwd.h"
class Hashrate;
class ApiState
{
public:
    ApiState();
    ~ApiState();

    char *get(const char *url, int *status) const;
    void tick(const Hashrate *hashrate);
    void tick(const NetworkState &results);

private:
    char *finalize(rapidjson::Document &doc) const;
    void genId();
    void getConnection(rapidjson::Document &doc) const;
    void getHashrate(rapidjson::Document &doc) const;
    void getIdentify(rapidjson::Document &doc) const;
    void getMiner(rapidjson::Document &doc) const;
    void getResults(rapidjson::Document &doc) const;

    char m_id[17];
    char m_workerId[128];
    double *m_hashrate;
    double m_highestHashrate;
    double m_totalHashrate[3];
    int m_threads;
    NetworkState m_network;
};

#endif /* __APISTATE_H__ */
