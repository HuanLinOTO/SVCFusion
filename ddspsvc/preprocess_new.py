import argparse
import asyncio
import os
import shutil
import subprocess
import time

import numpy as np
import rich
import rich.progress
import torch

from . import logger

# PYPATH = ".conda\\python.exe"
PYPATH = "python"


def timer(func):
    def func_wrapper(*args, **kwargs):
        from time import time

        time_start = time()
        result = func(*args, **kwargs)
        time_end = time()
        time_spend = time_end - time_start
        logger.info("\n{0} cost time {1} s\n".format(func.__name__, time_spend))
        return result

    return func_wrapper


async def exec_it(command, callback):
    try:
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            callback(str(line))

        await process.wait()
    except subprocess.CalledProcessError as e:
        logger.error(f"Error: {e}")


def get_callback(
    name, debug, progress: rich.progress.Progress, taskid: rich.progress.TaskID
):
    def real_cb(output):
        output = output
        # if debug:
        #     print(name + ": " + output)
        if "[!!]" in output:
            # logger.info(f"{name}: processed 1 file")
            progress.advance(taskid)
        else:
            print(name + ": " + output)

    return real_cb


def scan_dir(dir_path: str):
    filelists = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith(".wav"):
                # print(os.path.join(dir_path, file))
                filelists.append(os.path.join(root, file))
    return filelists


async def main(
    device: str,
    config: str,
    num_processes: int,
    debug: bool,
):
    with logger.Progress() as progress:
        # 这个脚本纯粹为了快，并没有做异常处理
        logger.warning(
            "NOTICE: This script is only for fast preprocessing, no error handling."
        )

        # 读取 train 和 val 的 filelist 分成 num_processes 块，输出到filelists/{timestramp}/chunk-n.txt 文件夹不存在时创建
        # Create the directory for filelists
        timestamp = str(int(time.time()))
        tmp_path = f"filelists/{timestamp}"
        os.makedirs(tmp_path, exist_ok=True)

        filelist_lines = []
        filelist_lines.extend(scan_dir("data/train/audio"))
        filelist_lines.extend(scan_dir("data/val/audio"))
        print(filelist_lines, len(filelist_lines))
        # exit()
        # 计算每个进程需要处理的行数
        num_train_lines = len(filelist_lines)

        # 确保切割后的行数正确，并处理可能的剩余行
        lines_per_chunk_train = len(filelist_lines) // num_processes

        # 分割 train filelist 到 chunks
        for i in range(num_processes):
            start_idx = i * lines_per_chunk_train
            end_idx = (
                start_idx + lines_per_chunk_train
                if i < num_processes - 1
                else len(filelist_lines)
            )

            train_chunk = filelist_lines[start_idx:end_idx]

            with open(
                f"filelists/{timestamp}/chunk-{i}.txt", "w", encoding="utf-8"
            ) as chunk_file:
                chunk_file.write("\n".join(train_chunk))

        filelists = []
        for root, dirs, files in os.walk(tmp_path):
            for file in files:
                if file.endswith(".txt"):
                    filelists.append(os.path.join(root, file))
        tasks = []
        print(filelists)
        # exit(0)
        num_files = num_train_lines
        taskid = progress.add_task("Preprocessing", total=num_files)
        for i in range(len(filelists)):
            logger.info(f"Processing {filelists[i]}")
            filelist = filelists[i]
            command = f"{PYPATH} -m ddspsvc.preprocess_chunk -c {config} -f {filelist} -d {device} --flag {timestamp}_{i}"
            tasks.append(
                exec_it(
                    command,
                    callback=get_callback(filelist, debug, progress, taskid),
                )
            )
            # task = exec_it(
            #     command,
            #     callback=get_callback(filelist, debug, progress, taskid),
            # )
            if len(tasks) == 2:
                print("start")
                await asyncio.gather(*tasks)
                tasks = []
        # 合并所有的pitch_aug_dict
        for tp in ["train", "val"]:
            pitch_aug_dict = {}
            for i in range(len(filelists)):
                path_pitchaugdict = os.path.join(
                    f"data/{tp}/" + f"pitch_aug_dict_{timestamp}_{i}.npy"
                )
                if os.path.exists(path_pitchaugdict):
                    logger.info(
                        f"Load pitch augmentation dictionary from: {path_pitchaugdict}"
                    )
                    pitch_aug_dict.update(
                        np.load(path_pitchaugdict, allow_pickle=True).item()
                    )
                    os.remove(path_pitchaugdict)
            np.save(
                f"data/{tp}/pitch_aug_dict.npy",
                pitch_aug_dict,
            )
    # Delete the temporary filelists directory
    shutil.rmtree(tmp_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="path to the config file"
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default=None,
        required=False,
        help="cpu or cuda, auto if not set",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=1,
        help="You are advised to set the number of processes to the same as the number of CPU cores",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Whether print subprocess output"
    )

    args = parser.parse_args()
    device = args.device
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_processes = args.num_processes
    debug = args.debug
    # debug = True
    asyncio.run(
        main(
            num_processes=num_processes,
            device=device,
            config=args.config,
            debug=debug,
        )
    )
