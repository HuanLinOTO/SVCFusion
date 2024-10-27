del New-SVCFusion-Preview-0.7z
python -m scripts.buildtools
python -m nuitka .\dist.py --module
@REM del dist.py
del dist.pyi
del dist.build /F /Q
encryption.exe launcher.py webui.bin
enigma64 webui_main.enigma64
@REM 7za a New-SVCFusion-Preview-0 configs configs_template ddspsvc fap lib Music_Source_Separation_Training SVCFusion ReFlowVaeSVC SoVITS "启动 WebUI.bat" webui.bin dist.cp310-win_amd64.pyd callwt.bat vr.py style.css pretrained other_weights
@REM 7za a New-SVCFusion-Preview-0 configs configs_template ddspsvc fap lib Music_Source_Separation_Training SVCFusion ReFlowVaeSVC SoVITS "启动 WebUI.bat" webui.bin dist.cp310-win_amd64.pyd callwt.bat vr.py style.css launcher.py
@REM 7za a SVCFusion-Pack configs configs_template ddspsvc fap lib Music_Source_Separation_Training SVCFusion ReFlowVaeSVC SoVITS "启动 WebUI.bat" webui.bin dist.cp310-win_amd64.pyd callwt.bat vr.py style.css pretrained other_weights .conda ddspsvc_6_1 launcher.py