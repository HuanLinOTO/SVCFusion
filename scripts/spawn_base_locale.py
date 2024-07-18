import os

# 原始文件路径和目标文件路径
source_file = "package_utils/locale/zh_CN.py"
target_file = "package_utils/locale/base.py"

# 读取源文件内容
with open(source_file, "r", encoding="utf-8") as file:
    lines = file.readlines()

# 处理文件内容
new_lines = []
inside_multiline_string = False

for line in lines:
    stripped_line = line.strip()
    stripped_line = stripped_line.replace("class _Locale(Locale):", "class Locale():")
    # 如果是类定义行，去掉继承的父类
    if (
        stripped_line.startswith("class")
        and "(" in stripped_line
        and "):" in stripped_line
    ):
        indent = line[: line.index("class")]
        class_name = stripped_line.split("(")[0]
        new_lines.append(f"{indent}{class_name}:\n")

    # 如果是变量定义行，去掉值并加上注释
    elif "=" in line and " = " in line and not inside_multiline_string:
        indent = line[: line.index(line.lstrip())]
        var_name, var_value = line.split("=", 1)
        var_name = var_name.strip()
        var_value = var_value.strip().strip('"')

        if var_value.endswith(('"""', "'''")) and var_value.count(var_value[0]) == 1:
            inside_multiline_string = var_value[0]

        new_line = f'{indent}{var_name} = ""  # {var_value}\n'
        new_lines.append(new_line)

    # 如果在多行字符串内，继续注释
    elif inside_multiline_string:
        ...
        # if stripped_line.endswith(inside_multiline_string):
        #     inside_multiline_string = False
        # new_lines.append(line)

    else:
        # new_lines.append(line)
        ...

# 将处理后的内容写入目标文件
with open(target_file, "w", encoding="utf-8") as file:
    file.writelines(new_lines)

print(f"文件已成功处理并保存到 {target_file}")
