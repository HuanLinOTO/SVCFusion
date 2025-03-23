chcp 65001
@echo off

echo 正在启动WebUI……


echo %~dp0

set PATH=%~dp0\ffmpeg\bin;%PATH%

set PYTHONPATH="%PYTHONPATH%;%~dp0"

echo %~dp0 > workdir

"%~dp0.conda\python.exe" "%~dp0\launcher.py"

pause