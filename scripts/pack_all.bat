python scripts/buildtools.py
python -m nuitka .\dist.py --module
del dist.py
del dist.pyi
del dist.build /F /Q
@REM encryption.exe launcher.py webui.bin
enigma64 webui_main.enigma64
7za a New-SVCFusion-Preview-0 configs configs_template ddspsvc fap lib Music_Source_Separation_Training SVCFusion ReFlowVaeSVC SoVITS "启动 WebUI.bat" webui.bin dist.cp310-win_amd64.pyd callwt.bat vr.py style.css .conda pretrain pretrained wt