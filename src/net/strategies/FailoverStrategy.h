#ifndef __FAILOVERSTRATEGY_H__
#define __FAILOVERSTRATEGY_H__
#include <vector>
#include "interfaces/IClientListener.h"
#include "interfaces/IStrategy.h"
class Client;
class IStrategyListener;
class Url;


class FailoverStrategy : public IStrategy, public IClientListener
{
public:
    FailoverStrategy(const std::vector<Url*> &urls, const char *agent, IStrategyListener *listener);

public:
    inline bool isActive() const override  { return m_active >= 0; }

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
    void add(const Url *url, const char *agent);

    int m_active;
    int m_index;
    IStrategyListener *m_listener;
    std::vector<Client*> m_pools;
};

#endif /* __FAILOVERSTRATEGY_H__ */
