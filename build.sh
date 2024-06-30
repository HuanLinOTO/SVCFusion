python scripts/buildtools.py
python -m nuitka .\dist.py --module
rm dist.py
rm dist.pyi
rm dist.build
# enigma64 webui_main.enigma64
# 7za a New-SVCFusion-Preview-0 configs configs_template ddspsvc fap ffmpeg lib Music_Source_Separation_Training package_utils ReFlowVaeSVC SoVITS wt "启动 WebUI.bat" webui.bin dist.cp310-win_amd64.pyd callwt.bat vr.py style.css