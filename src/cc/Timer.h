/* XMRigCC
 * Copyright 2019-     BenDr0id    <https://github.com/BenDr0id>, <ben@graef.in>
 *
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program. If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef __TIMER_H__
#define __TIMER_H__

#include <iostream>
#include <chrono>
#include <functional>
#include <thread>

class Timer
{
public:
  Timer() {}

  Timer(std::function<void(void)> func, uint64_t interval)
  {
    m_func = func;
    m_interval = interval;
  }

  ~Timer()
  {
    stop();
  }

public:

  void start()
  {
    m_running = true;
    m_thread = std::thread([&]()
    {
      while (m_running)
      {
        auto delta = std::chrono::steady_clock::now() + std::chrono::milliseconds(m_interval);
        std::this_thread::sleep_until(delta);
        m_func();
      }
    });

    m_thread.detach();
  }

  void stop()
  {
    m_running = false;

    if (m_thread.joinable())
    {
      m_thread.join();
    }
  }

  void setFunction(std::function<void(void)> func)
  {
    m_func = func;
  }

  void setInterval(uint64_t interval)
  {
    m_interval = interval;
  }

  bool isRunning()
  {
    return m_running;
  }

  uint64_t getInterval()
  {
    return m_interval;
  }

private:
  std::function<void(void)> m_func;
  std::thread m_thread;

  uint64_t m_interval = 0;
  bool m_running = false;
};

#endif //__TIMER_H__
