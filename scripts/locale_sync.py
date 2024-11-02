import ast
import hashlib
import os
import ollama

# class ModifyStringVisitor(ast.NodeTransformer):
#     def __init__(self, class_name, modify_function):
#         self.class_name = class_name
#         self.modify_function = modify_function
#         self.in_class = False
#         self.class_node = None
#     def visit_ClassDef(self, node):
#         if node.name == self.class_name:
#             self.in_class = True
#             self.class_node = node
#             # Transform the node (i.e., apply the modifications)
#             self.generic_visit(node)
#             self.in_class = False
#         else:
#             self.generic_visit(node)
#         return node

#     def visit_Str(self, node):
#         if self.in_class:
#             new_value = self.modify_function(node.s)
#             return ast.copy_location(ast.Str(s=new_value), node)
#         return node

#     def visit_Constant(self, node):
#         # Handle string literals in Python 3.8 and later
#         if self.in_class and isinstance(node.value, str):
#             new_value = self.modify_function(node.value)
#             return ast.copy_location(ast.Constant(value=new_value), node)
#         return node


def hash_string(value):
    return hashlib.md5(value.encode()).hexdigest()


infos = {
    "english": {
        "name": "en-us",
        "display_name": "English (US)",
        "output_file": "SVCFusion/locale/en_US.py",
    },
    "emojilang": {
        "name": "emojilang",
        "display_name": "😎",
        "output_file": "SVCFusion/locale/emoji.py",
        "desc": "emojilang 是一种特殊的语言，完全使用表情符号来表达你的意思，而不是文字",
    },
}


def main():
    for lang in infos:
        info = infos[lang]
        prompt = f"""
        {info.get('desc', '')}

        你需要将我提供的代码中的字符串翻译为{lang}

        代码的基本结构如下：
        ```
        from SVCFusion.locale.base import Locale

        locale_name = "zh-cn"
        locale_display_name = "简体中文"


        class _Locale(Locale):
        ```

        其中你只需要翻译 _Locale 类中的字符串即可

        这是一个 AI 整合包项目，请尽量地道，不要使用机器翻译的口吻

        你需要将翻译完成的 _Locale 类写在 <OUTPUT></OUTPUT> 标签中
        """
        stream = ollama.chat(
            "qwen2.5:14b-instruct",
            messages=[
                {
                    "role": "system",
                    "content": prompt,
                },
                # 读取代码
                {
                    "role": "user",
                    "content": open(
                        "SVCFusion/locale/zh_CN.py", encoding="utf-8"
                    ).read(),
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
        # 解析output中的内容
        result = result.split("<OUTPUT>")[1].split("</OUTPUT>")[0].strip()

        with open(info["output_file"], "w", encoding="utf-8") as f:
            f.write(result)


if __name__ == "__main__":
    main()
