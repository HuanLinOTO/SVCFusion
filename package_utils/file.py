import hashlib
import os
import shutil
from pathlib import Path

from loguru import logger


# 抄自 fap
def make_dirs(path, clean: bool = False):
    """Make directories.

    Args:
        path (Union[Path, str]): Path to the directory.
        clean (bool, optional): Whether to clean the directory. Defaults to False.
    """
    if isinstance(path, str):
        path = Path(path)

    if path.exists():
        if clean:
            logger.info(f"Cleaning output directory: {path}")
            shutil.rmtree(path)
        else:
            logger.info(f"Output directory already exists: {path}")

    path.mkdir(parents=True, exist_ok=True)


def getResultFileName(audio_path: str):
    # 计算文件的 md5
    md5_hash = ""
    with open(audio_path, "rb") as f:
        data = f.read()
        md5_hash = hashlib.md5(data).hexdigest()
    filename = (
        os.path.basename(audio_path)[::-1].replace(".wav"[::-1], "")[::-1]
        + f"_{md5_hash[:5]}.wav"
    )
    return f"results/{filename}", os.path.exists(f"results/{filename}")
