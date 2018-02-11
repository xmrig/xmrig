#ifndef __SINGLEPOOLSTRATEGY_H__
#define __SINGLEPOOLSTRATEGY_H__
#include "interfaces/IClientListener.h"
#include "interfaces/IStrategy.h"

class Client;
class IStrategyListener;
class Url;

class SinglePoolStrategy : public IStrategy, public IClientListener
{
public:
    SinglePoolStrategy(const Url *url, const char *agent, IStrategyListener *listener);

public:
    inline bool isActive() const override  { return m_active; }

    int64_t submit(const JobResult &result) override;
    void connect() override;
    void resume() override;
    void stop() override;
    void tick(uint64_t now) override;

protected:
    void onClose(Client *client, int failures) override;
    void onJobReceived(Client *client, const Job &job) override;
    void onLoginSuccess(Client *client) override;
    void onResultAccepted(Client *client, const SubmitResult &result, const char *error) override;

private:
    bool m_active;
    Client *m_client;
    IStrategyListener *m_listener;
};

#endif /* __SINGLEPOOLSTRATEGY_H__ */
