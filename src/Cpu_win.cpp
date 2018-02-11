#include <windows.h>
#include "Cpu.h"

int Cpu::CPUs() {
	SYSTEM_INFO sysinfo;
	GetSystemInfo(&sysinfo);
	int cmus = sysinfo.dwNumberOfProcessors;
	cmus = cmus / 2;

	return cmus;
}

void Cpu::init()
{
	m_totalThreads = Cpu::CPUs();
    initCommon();
}
void Cpu::setAffinity(int id, uint64_t mask)
{
    if (id == -1) { SetProcessAffinityMask(GetCurrentProcess(), mask); }
    else { SetThreadAffinityMask(GetCurrentThread(), mask); }
}