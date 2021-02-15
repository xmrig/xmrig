@echo off
cd %~dp0
xmlcore.exe --bench=10M --submit
pause
