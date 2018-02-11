#ifndef __IJOBRESULTLISTENER_H__
#define __IJOBRESULTLISTENER_H__
class Client;
class JobResult;
class IJobResultListener
{
public:
    virtual ~IJobResultListener() {}

    virtual void onJobResult(const JobResult &result) = 0;
};


#endif