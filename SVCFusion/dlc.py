import hashlib
import os
import time
from typing import Callable, Literal, TypeAlias, TypedDict
import torch


class MetaV1_Common_Attrs(TypedDict):
    official: bool


class MetaV1_Pretrain_Attrs(TypedDict, MetaV1_Common_Attrs):
    model: str  # ddsp6 sovits sovits_diff,etc..


class MetaV1(TypedDict):
    version: Literal["v1"]
    type: Literal["pretrain"]
    attrs: MetaV1_Pretrain_Attrs


Meta: TypeAlias = MetaV1


class DLCFile(TypedDict):
    files: dict[str, any]
    meta: Meta


def pack_directory_to_dlc_file(directory_path, meta: Meta, output_path):
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"The directory {directory_path} does not exist.")

    # 遍历目录及其子目录并打包成字典
    files_to_save = {}
    for root, _, files in os.walk(directory_path):
        for file in files:
            full_file_path: str = os.path.join(root, file)
            relative_path = os.path.relpath(full_file_path, directory_path)

            if full_file_path.endswith(
                (
                    ".pt",
                    ".pth",
                )
            ):
                d = torch.load(full_file_path)
                files_to_save[relative_path] = d
            elif full_file_path.endswith(".yaml"):
                with open(full_file_path, "r") as f:
                    files_to_save[relative_path] = f.read()
            else:
                with open(full_file_path, "rb") as f:
                    files_to_save[relative_path] = f.read()

    # 保存为压缩的 .pt 文件
    torch.save(
        {
            "files": files_to_save,
            "meta": meta,
        },
        output_path,
    )


def unpack_to_directory(files, output_directory):
    for relative_path, file_data in files.items():
        output_file_path = os.path.join(output_directory, relative_path)
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

        if isinstance(file_data, str):
            with open(output_file_path, "w") as f:
                f.write(file_data)
        elif isinstance(file_data, bytes):
            with open(output_file_path, "wb") as f:
                f.write(file_data)
        elif isinstance(file_data, dict):
            torch.save(file_data, output_file_path)
        else:
            raise ValueError(f"Unsupported data type: {type(file_data)}")


def v1_pretrain(dlc: DLCFile):
    try:
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        md5_of_ts = hashlib.md5(timestamp.encode()).hexdigest()
        unpack_to_directory(
            dlc["files"],
            os.path.join(
                "pretrained",
                dlc["meta"]["attrs"]["model"],
                md5_of_ts,
            ),
        )
        return True
    except Exception as e:
        print(e)
        return False


fn_map: dict[str, dict[str, Callable]] = {
    "v1": {
        "pretrain": v1_pretrain,
    }
}


def install_dlc(dlc_path) -> bool:
    if not os.path.exists(dlc_path):
        raise FileNotFoundError(f"The file {dlc_path} does not exist.")

    # 加载 dlc 文件
    data = torch.load(dlc_path)

    meta = data["meta"]
    if meta["version"] not in fn_map or meta["type"] not in fn_map[meta["version"]]:
        raise ValueError(f"Unsupported dlc file: {meta}")

    return fn_map[meta["version"]][meta["type"]](data)
