#ifndef __DONATESTRATEGY_H__
#define __DONATESTRATEGY_H__
#include <uv.h>
#include "interfaces/IClientListener.h"
#include "interfaces/IStrategy.h"
class Client;
class IStrategyListener;
class Url;
class DonateStrategy : public IStrategy, public IClientListener
{
public:
    DonateStrategy(const char *agent, IStrategyListener *listener);

public:
    inline bool isActive() const override  { return m_active; }
    inline void resume() override          {}

    int64_t submit(const JobResult &result) override;
    void connect() override;
    void stop() override;
    void tick(uint64_t now) override;

protected:
    void onClose(Client *client, int failures) override;
    void onJobReceived(Client *client, const Job &job) override;
    void onLoginSuccess(Client *client) override;
    void onResultAccepted(Client *client, const SubmitResult &result, const char *error) override;

private:
    void idle();
    void suspend();

    static void onTimer(uv_timer_t *handle);

    bool m_active;
    Client *m_client;
    const int m_donateTime;
    const int m_idleTime;
    IStrategyListener *m_listener;
    uv_timer_t m_timer;
};

#endif /* __DONATESTRATEGY_H__ */
