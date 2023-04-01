@echo off
cd /d "%~dp0"
xmrig.exe --bench=10M --submit
pause
