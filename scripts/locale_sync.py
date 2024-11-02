import ast
import hashlib
import os
import ollama


def hash_string(value):
    return hashlib.md5(value.encode()).hexdigest()


infos = {
    # "emojilang": {
    #     "name": "emojilang",
    #     "display_name": "😎",
    #     "output_file": "SVCFusion/locale/emoji.py",
    #     "desc": "Emojilang 是一种仅使用表情符号（emoji）作为唯一符号的语言，都通过特定的表情符号组合来表达。这种语言的结构完全依赖于表情符号的形状和意义，无法使用任何文字或传统符号。但是你仍然需要保留代码结构。比如‘你好！请和我交往吧！’ -> '👋😊！🙏❤️➡️👫'",
    # },
    "english": {
        "name": "en-us",
        "display_name": "English (US)",
        "output_file": "SVCFusion/locale/en_US.py",
    },
}


# 提取出 _Locale 类
def extract_locale_class(content):
    tree = ast.parse(content)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "_Locale":
            return node
    return None


# 将类节点转换为字典
def class_to_dict(class_node):
    result = {}
    for item in class_node.body:
        if isinstance(item, ast.ClassDef):
            result[item.name] = class_to_dict(item)
        elif isinstance(item, ast.Assign):
            for target in item.targets:
                if isinstance(target, ast.Name):
                    if isinstance(item.value, ast.Dict):
                        result[target.id] = ast.literal_eval(ast.dump(item.value))
                    else:
                        result[target.id] = item.value.s
    return result


# 合并两个字典
def merge_dicts(dict1, dict2):
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result:
            if isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


# 将字典转换为类定义代码
def dict_to_class(name, d, indent=0):
    ind = " " * indent
    lines = [f"{ind}class {name}(Locale):"]
    for key, value in d.items():
        if isinstance(value, dict):
            lines.append(dict_to_class(key, value, indent + 4))
        else:
            lines.append(f'{ind}    {key} = "{value}"')
    return "\n".join(lines)


def get_translations(lang, info):
    prompt = f"""
你需要将我提供的代码中的字符串翻译为 {lang}

代码的基本结构如下：
```
from SVCFusion.locale.base import Locale

locale_name = "zh-cn"
locale_display_name = "简体中文"


class _Locale(Locale):
```

其中你只需要翻译 _Locale 类中的字符串即可

这是一个 AI 整合包项目，请尽量地道，不要使用机器翻译的口吻

你不需要输出完整的代码，只需要 _Locale 类

你需要将翻译完成的 _Locale 类写在 <OUTPUT></OUTPUT> 标签中

{info.get('desc', '')}
"""

    stream: ast.Iterator[os.Mapping[str, ast.Any]] = ollama.chat(
        "qwen2.5:14b-instruct",
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            # 读取代码
            {
                "role": "user",
                "content": open("SVCFusion/locale/zh_CN.py", encoding="utf-8").read(),
            },
        ],
        options={
            "num_ctx": 25565,
        },
        stream=True,
    )
    result = ""
    for chunk in stream:
        msg = chunk["message"]["content"]
        print(msg, end="", flush=True)
        result += msg
    result = result.strip()
    result = result.split("<OUTPUT>")[1].split("</OUTPUT>")[0].strip()

    # 提取出 _Locale 类
    result = result.split("class _Locale(Locale):")[1].strip()
    return "class _Locale(Locale): \n    " + result


def main():
    for lang in infos:
        info = infos[lang]
        translated = get_translations(lang, info)
        output_file = info["output_file"]
        with open(output_file, "r", encoding="utf-8") as f:
            original_content = f.read()

        locale_name = original_content.split('locale_name = "')[1].split('"')[0]
        locale_display_name = original_content.split('locale_display_name = "')[
            1
        ].split('"')[0]

        new_content = f"""
from SVCFusion.locale.base import Locale

locale_name = "{locale_name}"
locale_display_name = "{locale_display_name}"

{translated}
    """

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(new_content)


if __name__ == "__main__":
    main()
