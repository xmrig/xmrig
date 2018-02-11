#ifndef __APP_H__
#define __APP_H__
#include <uv.h>
#include "interfaces/IConsoleListener.h"
class Console;
class Httpd;
class Network;
class Options;
class App : public IConsoleListener
{
public:
  App(int argc, char **argv);
  ~App();

  int exec();

protected:
  void onConsoleCommand(char command) override;

private:
  void background();
  void close();

  static void onSignal(uv_signal_t *handle, int signum);

  static App *m_self;

  Console *m_console;
  Httpd *m_httpd;
  Network *m_network;
  Options *m_options;
  uv_signal_t m_signal;
};

#endif