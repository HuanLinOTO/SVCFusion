import os
import ast

# 文件路径
locale_path = "package_utils/locale"
zh_CN_file = os.path.join(locale_path, "zh_CN.py")


# 读取Python文件内容并解析为AST
def parse_py_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        file_content = file.read()
    return ast.parse(file_content)


# 提取Locale类中的所有属性
def extract_locale_attrs(class_node):
    attrs = {}
    for node in class_node.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    attrs[target.id] = ast.literal_eval(node.value)
        elif isinstance(node, ast.ClassDef):
            attrs[node.name] = extract_locale_attrs(node)
    return attrs


# 对比两个Locale类的属性
def compare_locales(locale1, locale2, prefix=""):
    differences = {}
    for key in locale1.keys():
        if key not in locale2:
            differences[prefix + key] = locale1[key]
        elif isinstance(locale1[key], dict) and isinstance(locale2[key], dict):
            sub_diff = compare_locales(locale1[key], locale2[key], prefix + key + ".")
            differences.update(sub_diff)
    return differences


# 解析zh_CN.py文件
zh_CN_tree = parse_py_file(zh_CN_file)
zh_CN_class = next(node for node in zh_CN_tree.body if isinstance(node, ast.ClassDef))
zh_CN_attrs = extract_locale_attrs(zh_CN_class)

# 遍历locale文件夹中的其他语言文件
for file_name in os.listdir(locale_path):
    if file_name.endswith(".py") and file_name not in [
        "zh_CN.py",
        "__init__.py",
        "base.py",
    ]:
        file_path = os.path.join(locale_path, file_name)
        locale_tree = parse_py_file(file_path)
        locale_class = next(
            (node for node in locale_tree.body if isinstance(node, ast.ClassDef)), None
        )

        if locale_class:
            locale_attrs = extract_locale_attrs(locale_class)
            differences = compare_locales(zh_CN_attrs, locale_attrs)

            if differences:
                print(f"与zh_CN.py的差异 ({file_name}):")
                for key, value in differences.items():
                    print(f"  {key}: {value}")
            else:
                print(f"{file_name}与zh_CN.py没有差异")
