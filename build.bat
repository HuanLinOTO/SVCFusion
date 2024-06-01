python buildtools.py
python -m nuitka .\dist.py --module
@REM del dist.py
del dist.pyi
del dist.build /F /Q
encryption.exe launcher.py webui.bin