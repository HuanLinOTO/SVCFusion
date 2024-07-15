import os
import shutil


def move_and_rename_files(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        for filename in filenames:
            # 获取文件的完整路径
            old_path = os.path.join(dirpath, filename)

            # 获取文件所在目录的相对路径并用下划线替换目录分隔符
            relative_dir = os.path.relpath(dirpath, root_dir).replace(os.sep, "_")

            # 新的文件名和路径
            new_filename = f"{relative_dir}_{filename}"
            new_path = os.path.join(root_dir, new_filename)

            # 移动并重命名文件
            shutil.move(old_path, new_path)
            print(f"Moved and renamed: {old_path} -> {new_path}")


if __name__ == "__main__":
    root_directory = r"E:\Datasets\female_​雨星サイファ"
    move_and_rename_files(root_directory)
