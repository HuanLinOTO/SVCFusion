import argparse
import asyncio
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

        # Read train and val filelists
        with open(args.train_filelist, "r", encoding="utf-8") as train_file:
            train_lines = train_file.readlines()

        with open(args.val_filelist, "r", encoding="utf-8") as val_file:
            val_lines = val_file.readlines()

        # 计算每个进程需要处理的行数
        num_train_lines = len(train_lines)
        num_val_lines = len(val_lines)

        # 确保切割后的行数正确，并处理可能的剩余行
        lines_per_chunk_train = len(train_lines) // args.num_processes
        lines_per_chunk_val = len(val_lines) // args.num_processes

        # 分割 train filelist 到 chunks
        for i in range(args.num_processes):
            start_idx = i * lines_per_chunk_train
            end_idx = (
                start_idx + lines_per_chunk_train
                if i < args.num_processes - 1
                else len(train_lines)
            )

            train_chunk = train_lines[start_idx:end_idx]

            with open(
                f"filelists/{timestamp}/chunk-{i}.txt", "w", encoding="utf-8"
            ) as chunk_file:
                chunk_file.writelines(train_chunk)

        # 分割 val filelist 到 chunks
        for i in range(args.num_processes):
            start_idx = i * lines_per_chunk_val
            end_idx = (
                start_idx + lines_per_chunk_val
                if i < args.num_processes - 1
                else len(val_lines)
            )

            val_chunk = val_lines[start_idx:end_idx]

            with open(
                f"filelists/{timestamp}/chunk-{i}.txt", "a", encoding="utf-8"
            ) as chunk_file:
                chunk_file.writelines(val_chunk)
        filelists = []
        for root, dirs, files in os.walk(tmp_path):
            for file in files:
                if file.endswith(".txt"):
                    filelists.append(os.path.join(root, file))
        tasks = []
        print(filelists)
        # exit(0)
        num_files = num_train_lines + num_val_lines
        taskid = progress.add_task("Preprocessing", total=num_files)
        for filelist in filelists:
            command = f"{PYPATH} -m SoVITS.preprocess_chunk --f0_predictor {f0p} --filelist {filelist}"
            if use_diff:
                command += " --use_diff"
            tasks.append(
                exec_it(
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
        "--subprocess_num_workers",
        type=int,
        default=1,
        help="Number of workers to use for ThreadPoolExecutor",
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
            sub_num_workers=args.subprocess_num_workers,
        )
    )
