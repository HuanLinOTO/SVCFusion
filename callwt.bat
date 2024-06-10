chcp 65001
@echo off

echo 正在启动WebUI……

echo %~dp0

set PATH=%~dp0\ffmpeg\bin;%PATH%

call "%~dp0.conda\Scripts\activate.bat"

set PYTHONPATH=%PYTHONPATH%;%~dp0

echo %~dp0 > workdir

%~dp0.conda\ddsp.webui.exe %~dp0\webui.bin

pause