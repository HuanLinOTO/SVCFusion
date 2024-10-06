import ast
import hashlib
import os
import ollama


class ModifyStringVisitor(ast.NodeTransformer):
    def __init__(self, class_name, modify_function):
        self.class_name = class_name
        self.modify_function = modify_function
        self.in_class = False
        self.class_node = None

    def visit_ClassDef(self, node):
        if node.name == self.class_name:
            self.in_class = True
            self.class_node = node
            # Transform the node (i.e., apply the modifications)
            self.generic_visit(node)
            self.in_class = False
        else:
            self.generic_visit(node)
        return node

    def visit_Str(self, node):
        if self.in_class:
            new_value = self.modify_function(node.s)
            return ast.copy_location(ast.Str(s=new_value), node)
        return node

    def visit_Constant(self, node):
        # Handle string literals in Python 3.8 and later
        if self.in_class and isinstance(node.value, str):
            new_value = self.modify_function(node.value)
            return ast.copy_location(ast.Constant(value=new_value), node)
        return node


def hash_string(value):
    return hashlib.md5(value.encode()).hexdigest()


examples = {
    "english": ["你好。", "Hello."],
    "emojilang": ["你好。", "👋🏻"],
}

opt_file = {
    "english": "package_utils/locale/en_US.py",
    "emojilang": "package_utils/locale/emoji.py",
}

info = {
    "english": {
        "name": "en-us",
        "display_name": "English (US)",
    },
    "emojilang": {
        "name": "emojilang",
        "display_name": "😎",
    },
}

model = {
    "english": "qwen2.5:14b-instruct",
    "emojilang": "qwen2.5:14b-instruct",
}

target = {
    "english": "english, text only",
    "emojilang": "emojilang, containing emojis only",
}

for lang in ["english", "emojilang"]:
    prompt = (
        """
    Task: translate input to {target}, keep the markdown format

    Example:
    Input: {example_input}
    Result: {example_output}

    Input: {1}
    Output: 
    """.replace("{example_input}", examples[lang][0])
        .replace("{example_output}", examples[lang][1])
        .replace("{target}", target[lang])
    )

    # 修改字符串的函数，例如将所有字符串转换为大写
    def modify_string(value):
        file = "translate_tmp/" + lang + "/" + hash_string(value) + ".txt"
        os.makedirs(os.path.dirname(file), exist_ok=True)

        if os.path.exists(file):
            with open(file, "r", encoding="utf-8") as f:
                return f.read()
        print(value)
        stream = ollama.generate(
            model=model[lang],
            prompt=prompt.replace("{1}", value),
            stream=True,
        )
        result = ""

        for chunk in stream:
            msg = chunk["response"]
            print(msg, end="", flush=True)
            result += msg

        # 从 [] 中提取翻译结果

        with open(
            file,
            "w",
            encoding="utf-8",
        ) as f:
            f.write(result)
        print("")
        return result

    # 解析zh_CN.py文件
    with open("package_utils/locale/zh_CN.py", "r", encoding="utf-8") as file:
        tree = ast.parse(file.read(), filename="zh_CN.py")

    # 创建访问者并修改AST中的字符串
    modifier = ModifyStringVisitor("_Locale", modify_string)
    modified_tree = modifier.visit(tree)

    # 将修改后的类还原为Python代码并输出到aaa.py文件中
    with open(opt_file[lang], "w", encoding="utf-8") as output_file:
        output_file.write(
            ast.unparse(modified_tree)
            .replace("locale_name = 'zh-cn'", f"locale_name = '{info[lang]['name']}'")
            .replace(
                "locale_display_name = '简体中文'",
                f"locale_display_name = '{info[lang]['display_name']}'",
            )
        )
