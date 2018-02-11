#ifndef __IWORKER_H__
#define __IWORKER_H__
#include <stdint.h>

class IWorker
{
public:
    virtual ~IWorker() {}

    virtual uint64_t hashCount() const = 0;
    virtual uint64_t timestamp() const = 0;
    virtual void start()               = 0;
};


#endif