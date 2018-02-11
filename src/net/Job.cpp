#include <string.h>
#include "net/Job.h"
static inline unsigned char hf_hex2bin(char c, bool &err)
{
    if (c >= '0' && c <= '9') { return c - '0'; }
    else if (c >= 'a' && c <= 'f') { return c - 'a' + 0xA; }
    else if (c >= 'A' && c <= 'F') { return c - 'A' + 0xA; }
    err = true;
    return 0;
}
static inline char hf_bin2hex(unsigned char c)
{
    if (c <= 0x9) { return '0' + c; }
    return 'a' - 0xA + c;
}
Job::Job(int poolId, bool nicehash) :
    m_nicehash(nicehash),
    m_poolId(poolId),
    m_threadId(-1),
    m_size(0),
    m_diff(0),
    m_target(0)
{
}
Job::~Job()
{
}
bool Job::setBlob(const char *blob)
{
    if (!blob) { return false; }
    m_size = strlen(blob);
    if (m_size % 2 != 0) { return false; }
    m_size /= 2;
    if (m_size < 76 || m_size >= sizeof(m_blob)) { return false; }
    if (!fromHex(blob, (int) m_size * 2, m_blob)) { return false; }
    if (*nonce() != 0 && !m_nicehash) { m_nicehash = true; }
#   ifdef XMRIG_PROXY_PROJECT
    memset(m_rawBlob, 0, sizeof(m_rawBlob));
    memcpy(m_rawBlob, blob, m_size * 2);
#   endif
    return true;
}
bool Job::setTarget(const char *target)
{
    if (!target) { return false; }
    const size_t len = strlen(target);
    if (len <= 8) {
        uint32_t tmp = 0;
        char str[8];
        memcpy(str, target, len);
        if (!fromHex(str, 8, reinterpret_cast<unsigned char*>(&tmp)) || tmp == 0) { return false; }
        m_target = 0xFFFFFFFFFFFFFFFFULL / (0xFFFFFFFFULL / static_cast<uint64_t>(tmp));
    }
    else if (len <= 16) {
        m_target = 0;
        char str[16];
        memcpy(str, target, len);
        if (!fromHex(str, 16, reinterpret_cast<unsigned char*>(&m_target)) || m_target == 0) { return false; }
    }
    else { return false; }

#   ifdef XMRIG_PROXY_PROJECT
    memset(m_rawTarget, 0, sizeof(m_rawTarget));
    memcpy(m_rawTarget, target, len);
#   endif
    m_diff = toDiff(m_target);
    return true;
}
bool Job::fromHex(const char* in, unsigned int len, unsigned char* out)
{
    bool error = false;
    for (unsigned int i = 0; i < len; i += 2) {
        out[i / 2] = (hf_hex2bin(in[i], error) << 4) | hf_hex2bin(in[i + 1], error);
        if (error) { return false; }
    }
    return true;
}
void Job::toHex(const unsigned char* in, unsigned int len, char* out)
{
    for (unsigned int i = 0; i < len; i++) {
        out[i * 2] = hf_bin2hex((in[i] & 0xF0) >> 4);
        out[i * 2 + 1] = hf_bin2hex(in[i] & 0x0F);
    }
}
bool Job::operator==(const Job &other) const { return m_id == other.m_id && memcmp(m_blob, other.m_blob, sizeof(m_blob)) == 0; }