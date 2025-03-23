import argparse
import asyncio
import math
import os
import shutil
import subprocess
import time

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


def iter_array(array):
    for i in array:
        yield i


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
        if debug:
            print(name + ": " + output)
        if "[!!]" in output:
            # logger.info(f"{name}: processed 1 file")
            progress.advance(taskid)

    return real_cb


async def main(args, device, f0p, use_diff, sub_num_workers, debug):
    with logger.Progress() as progress:
        # 这个脚本纯粹为了快，并没有做异常处理
        logger.warning(
            "NOTICE: This script is only for fast preprocessing, no error handling."
        )
        # print(speech_encoder)
        logger.info("Using device: " + str(device))

        logger.info("Using f0 extractor: " + f0p)

        # 读取 train 和 val 的 filelist 分成 num_processes 块，输出到filelists/{timestramp}/chunk-n.txt 文件夹不存在时创建
        # Create the directory for filelists
        timestamp = str(int(time.time()))
        tmp_path = f"filelists/{timestamp}"
        os.makedirs(tmp_path, exist_ok=True)

        file_filelists = []
        filelist = [
            *open(args.train_filelist, "r", encoding="utf-8").read().split("\n"),
            *open(args.val_filelist, "r", encoding="utf-8").read().split("\n"),
        ]
        file_per_chunk = 5000.0

        for i in range(math.ceil(float(len(filelist)) / file_per_chunk)):
            with open(f"{tmp_path}/train_{i}.txt", "w", encoding="utf-8") as f:
                f.write(
                    "\n".join(
                        filelist[
                            i * int(file_per_chunk) : (i + 1) * int(file_per_chunk)
                        ]
                    )
                )
            file_filelists.append(f"{tmp_path}/train_{i}.txt")

        tasks = []
        # exit(0)

        semaphore = asyncio.Semaphore(args.num_processes)  # Limit to 4 concurrent tasks

        async def limited_exec_it(command, callback):
            async with semaphore:
                print(command, "running!")
                await exec_it(command, callback)

        num_files = len(file_filelists)
        taskid = progress.add_task("Preprocessing", total=num_files)
        for filelist in file_filelists:
            command = f"{PYPATH} -m SoVITS.preprocess_chunk --f0_predictor {f0p} --filelist {filelist}"
            if use_diff:
                command += " --use_diff"
            tasks.append(
                limited_exec_it(
                    command,
                    callback=get_callback(filelist, debug, progress, taskid),
                )
            )
        await asyncio.gather(*tasks)

    # Delete the temporary filelists directory
    shutil.rmtree(tmp_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", type=str, default=None)
    parser.add_argument(
        "--train-filelist",
        type=str,
        default="filelists/train.txt",
        help="path to val filelist.txt",
    )
    parser.add_argument(
        "--val-filelist",
        type=str,
        default="filelists/val.txt",
        help="path to val filelist.txt",
    )
    parser.add_argument(
        "--use_diff", action="store_true", help="Whether to use the diffusion model"
    )
    parser.add_argument(
        "--f0_predictor",
        type=str,
        default="rmvpe",
        help="Select F0 predictor, can select crepe,pm,dio,harvest,rmvpe,fcpe|default: pm(note: crepe is original F0 using mean filter)",
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
    f0p = args.f0_predictor
    device = args.device
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_processes = args.num_processes
    use_diff = args.use_diff
    debug = args.debug
    # debug = True
    asyncio.run(
        main(
            args=args,
            device=device,
            f0p=f0p,
            use_diff=use_diff,
            debug=debug,
            sub_num_workers=None,
        )
    )
