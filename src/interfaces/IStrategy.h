#ifndef __ISTRATEGY_H__
#define __ISTRATEGY_H__
#include <stdint.h>
class JobResult;
class IStrategy
{
public:
    virtual ~IStrategy() {}

    virtual bool isActive() const                   = 0;
    virtual int64_t submit(const JobResult &result) = 0;
    virtual void connect()                          = 0;
    virtual void resume()                           = 0;
    virtual void stop()                             = 0;
    virtual void tick(uint64_t now)                 = 0;
};


#endif