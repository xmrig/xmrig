//
// Created by Haifa Bogdan Adnan on 04/08/2018.
//

#ifndef DONATE_HTTP_H
#define DONATE_HTTP_H

using namespace std;

class Http {
public:
    Http();
    virtual ~Http();

    virtual string httpGet(const string &url) { return ""; };
    virtual string httpPost(const string &url, const string &post_data, const string &content_type) { return ""; };
    string encode(const string &src);
    vector<string> resolveHost(const string &hostname);

private:
    static int m_socketlibReference;
};

class HttpInternalImpl : public Http {
public:
    virtual string httpGet(const string &url);
    virtual string httpPost(const string &url, const string &post_data, const string &content_type);

private:
    string getResponse(const string &url, const string &post_data, const string &content_type);
};

#endif //DONATE_HTTP_H
