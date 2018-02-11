#ifndef __API_H__
#define __API_H__
#include <uv.h>
class ApiState;
class Hashrate;
class NetworkState;

class Api
{
public:
    static bool start();
    static void release();

    static char *get(const char *url, int *status);
    static void tick(const Hashrate *hashrate);
    static void tick(const NetworkState &results);

private:
    static ApiState *m_state;
    static uv_mutex_t m_mutex;
};

#endif /* __API_H__ */
