import base64
import json
import os


def dir_to_dict(root_dir):
    result = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            # 转为 posix 路径
            try:
                with open(file_path, "rb") as f:
                    content = base64.b64encode(f.read()).decode()
            except Exception as e:
                content = f"[Error reading file: {e}]"
            result[
                (
                    os.path.relpath(file_path, root_dir)
                    .replace(".\\", "")
                    .replace("\\", "/")
                )
            ] = content
    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("用法: python dir_to_dict.py <目录路径>")
    else:
        d = dir_to_dict(sys.argv[1])
        # 保存 d 到gradio/files.py
        with open("gradio/files.py", "w", encoding="utf-8") as f:
            f.write("files = ")
            json.dump(d, f, ensure_ascii=False, indent=4)
