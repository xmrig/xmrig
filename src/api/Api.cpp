#include <string.h>
#include "api/Api.h"
#include "api/ApiState.h"
ApiState *Api::m_state = nullptr;
uv_mutex_t Api::m_mutex;
bool Api::start(){ return 0; }
void Api::release(){}
char *Api::get(const char *url, int *status){ }
void Api::tick(const Hashrate *hashrate){}
void Api::tick(const NetworkState &network){}