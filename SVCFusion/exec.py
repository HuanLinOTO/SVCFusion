import os
import subprocess
import sys

from loguru import logger


executable = (
    os.path.abspath(".conda\\python.exe")
    if sys.platform == "win32"
    else (sys.executable if "python" in sys.executable else "python")
)


def exec_it(command):
    accumulated_output = ""
    try:
        # command = 'python -c "import time; [print(i) or time.sleep(1) for i in range(1, 6)]"'
        result = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            shell=True,
            text=True,
        )
        accumulated_output += f"Command: {command}\n"
        yield accumulated_output
        progress_line = None
        for line in result.stdout:
            if r"it/s" in line or r"s/it" in line:  # 防止进度条刷屏
                progress_line = line
            else:
                accumulated_output += line
            if progress_line is None:
                yield accumulated_output
            else:
                yield accumulated_output + progress_line
        result.communicate()
    except subprocess.CalledProcessError as e:
        result = e.output
        accumulated_output += f"Error: {result}\n"
        yield accumulated_output


def exec(command: str):
    logger.info(f"Run command: {command}")
    code = os.system(command)
    logger.info(f"Command finished with code: {code}")
    return code


def start_with_cmd(cmd: str):
    # cmd = "call .conda\\Scripts\\activate.bat" + " && " + cmd
    logger.info(f"Run command with cmd: {cmd}")
    if sys.platform == "win32":
        subprocess.Popen(["wt\\wt", *cmd.split(" ")], shell=True)
    else:
        os.system(cmd)
