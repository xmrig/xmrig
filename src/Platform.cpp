#include <string.h>
#include <uv.h>
#include "Platform.h"

char *Platform::m_defaultConfigName = nullptr;
char *Platform::m_userAgent         = nullptr;

const char *Platform::defaultConfigName()
{
    return nullptr;
}
