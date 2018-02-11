#ifndef __ICLIENTLISTENER_H__
#define __ICLIENTLISTENER_H__
#include <stdint.h>

class Client;
class Job;
class SubmitResult;

class IClientListener
{
public:
    virtual ~IClientListener() {}

    virtual void onClose(Client *client, int failures)                                           = 0;
    virtual void onJobReceived(Client *client, const Job &job)                                   = 0;
    virtual void onLoginSuccess(Client *client)                                                  = 0;
    virtual void onResultAccepted(Client *client, const SubmitResult &result, const char *error) = 0;
};


#endif