#ifndef __ICONSOLELISTENER_H__
#define __ICONSOLELISTENER_H__


class IConsoleListener
{
public:
    virtual ~IConsoleListener() {}

    virtual void onConsoleCommand(char command) = 0;
};


#endif