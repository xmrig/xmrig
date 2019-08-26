//
// Created by Haifa Bogdan Adnan on 04/08/2018.
//

#ifndef DONATE_HTTP_H
#define DONATE_HTTP_H

using namespace std;

class http {
public:
    http();
    virtual ~http();

    virtual string _http_get(const string &url) { return ""; };
    virtual string _http_post(const string &url, const string &post_data, const string &content_type) { return ""; };
    string _encode(const string &src);
    vector<string> _resolve_host(const string &hostname);

private:
    static int __socketlib_reference;
};

class http_internal_impl : public http {
public:
    virtual string _http_get(const string &url);
    virtual string _http_post(const string &url, const string &post_data, const string &content_type);

private:
    string __get_response(const string &url, const string &post_data, const string &content_type);
};

#endif //DONATE_HTTP_H
