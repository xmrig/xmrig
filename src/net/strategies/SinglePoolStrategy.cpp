#include "interfaces/IStrategyListener.h"
#include "net/Client.h"
#include "net/strategies/SinglePoolStrategy.h"
#include "Options.h"
SinglePoolStrategy::SinglePoolStrategy(const Url *url, const char *agent, IStrategyListener *listener) :
    m_active(false),
    m_listener(listener)
{
    m_client = new Client(0, agent, this);
    m_client->setUrl(url);
    m_client->setRetryPause(Options::i()->retryPause() * 1000);
}
int64_t SinglePoolStrategy::submit(const JobResult &result)
{
    return m_client->submit(result);
}
void SinglePoolStrategy::connect()
{
    m_client->connect();
}
void SinglePoolStrategy::resume()
{
    if (!isActive()) {
        return;
    }

    m_listener->onJob(m_client, m_client->job());
}
void SinglePoolStrategy::stop()
{
    m_client->disconnect();
}
void SinglePoolStrategy::tick(uint64_t now)
{
    m_client->tick(now);
}
void SinglePoolStrategy::onClose(Client *client, int failures)
{
    if (!isActive()) {
        return;
    }

    m_active = false;
    m_listener->onPause(this);
}
void SinglePoolStrategy::onJobReceived(Client *client, const Job &job)
{
    m_listener->onJob(client, job);
}
void SinglePoolStrategy::onLoginSuccess(Client *client)
{
    m_active = true;
    m_listener->onActive(client);
}
void SinglePoolStrategy::onResultAccepted(Client *client, const SubmitResult &result, const char *error)
{
    m_listener->onResultAccepted(client, result, error);
}
