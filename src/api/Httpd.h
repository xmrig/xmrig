#ifndef __HTTPD_H__
#define __HTTPD_H__


#include <uv.h>


struct MHD_Connection;
struct MHD_Daemon;
struct MHD_Response;


class Httpd
{
public:
    Httpd(int port, const char *accessToken);
    bool start();

private:
    int auth(const char *header);

    static int done(MHD_Connection *connection, int status, MHD_Response *rsp);
    static int handler(void *cls, MHD_Connection *connection, const char *url, const char *method, const char *version, const char *upload_data, size_t *upload_data_size, void **con_cls);

    const char *m_accessToken;
    const int m_port;
    MHD_Daemon *m_daemon;
};

#endif /* __HTTPD_H__ */
