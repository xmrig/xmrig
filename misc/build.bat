@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
rmdir /S /Q build
del %~dp0\xmrig-%1-win64.zip
mkdir build &&^
cd build &&^
git clone https://github.com/MoneroOcean/xmrig.git &&^
git clone https://github.com/xmrig/xmrig-deps.git &&^
mkdir xmrig\build &&^
cd xmrig\build &&^
git checkout %1 &&^
cmake .. -G "Visual Studio 16 2019" -DXMRIG_DEPS=%~dp0\build\xmrig-deps\msvc2019\x64 &&^
msbuild /p:Configuration=Release xmrig.sln &&^
cd Release &&^
copy ..\..\src\config.json . &&^
copy ..\..\bin\WinRing0\WinRing0x64.sys . &&^
7z a -tzip -mx %~dp0\xmrig-%1-win64.zip xmrig.exe config.json WinRing0x64.sys &&^
cd %~dp0 &&^
rmdir /S /Q build