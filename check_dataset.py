import os


def check_files(directory):
    # 获取目标目录下所有的wav文件
    wav_files = [f for f in os.listdir(directory) if f.endswith(".wav")]

    missing_files = []

    for wav_file in wav_files:
        base_name = os.path.splitext(wav_file)[0]

        spec_file = os.path.join(directory, f"{base_name}.spec.pt")
        f0_file = os.path.join(directory, f"{base_name}.wav.f0.npy")
        soft_file = os.path.join(directory, f"{base_name}.wav.soft.pt")

        # 以上三个变量缺一个把这个变量就扔进 missingfiles
        for file in [spec_file, f0_file, soft_file]:
            if not os.path.exists(file):
                missing_files.append(file)

    if missing_files:
        print(missing_files)
    else:
        print("所有文件都存在对应的spec.pt、f0.npy和soft.pt文件。")


# 调用函数，扫描目标目录
target_directory = "data/44k/test"  # 替换成你的目标目录
check_files(target_directory)
