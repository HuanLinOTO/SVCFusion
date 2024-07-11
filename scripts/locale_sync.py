import inspect
import os
import importlib.util


def load_module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


locale_dict: dict[str, dict] = {}
locale_to_display_name = {}
locale_to_path = {}


def merge_dicts(ref, source, prefix=""):
    result = source.copy()  # 复制 source 到结果字典
    for key in ref:
        if key not in source:
            print(f"Key {prefix}.{key} not found in source")
            result[key] = ref[key]
        if key in source and isinstance(source[key], dict):
            result[key] = merge_dicts(
                ref[key],
                source[key],
                prefix=f"{prefix}.{key}",
            )
    return result


for filename in os.listdir("package_utils/locale"):
    if filename.endswith(".py") and filename not in ["__init__.py", "base.py"]:
        file_path = os.path.join("package_utils/locale", filename)
        module_name = os.path.splitext(filename)[0]
        module = load_module_from_file(module_name, file_path)

        if (
            hasattr(module, "_Locale")
            and hasattr(module, "locale_name")
            and hasattr(module, "locale_display_name")
        ):
            _Locale = getattr(module, "_Locale")
            locale_name = getattr(module, "locale_name")
            locale_display_name = getattr(module, "locale_display_name")

            locale_dict[locale_name] = _Locale
            locale_to_display_name[locale_name] = locale_display_name

            locale_to_path[locale_name] = os.path.join("package_utils/locale", filename)

            print(f"Loaded {locale_name} from {filename}")


def remove_magic(dict):
    return {key: value for key, value in dict.items() if not key.startswith("__")}


# 递归
def class_to_dict(cls):
    return remove_magic(
        {
            key: class_to_dict(value) if inspect.isclass(value) else value
            for key, value in cls.__dict__.items()
        }
    )


LocaleDict = dict[str, "LocaleDict"]


# 要递归的
def dict_to_class_code(source, class_name, extent_node="", is_root=False):
    if extent_node:
        extent_name = f"{extent_node}{'' if is_root else '.'+class_name}"
        code = f"class {class_name}({extent_name}):\n"
    else:
        code = f"class {class_name}:\n"
    for key, value in source.items():
        if isinstance(value, dict):
            tmp = dict_to_class_code(value, key, extent_name if extent_node else "")
            for i in tmp.split("\n"):
                code += f"    {i}\n"
        else:
            if isinstance(value, str):
                if "\n" in value:
                    value = f'"""\n{value}\n"""'
                else:
                    value = f'"{value}"'

            code += f"    {key} = {value}\n"
    return code


def save_locale_file(class_code, path):
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(class_code)
    except UnicodeEncodeError as _e:
        with open(path, "w") as f:
            f.write(class_code)


base_locale = class_to_dict(locale_dict["zh-cn"])
del locale_dict["zh-cn"]
for lang in locale_dict:
    locale = class_to_dict(locale_dict[lang])
    result = merge_dicts(
        base_locale,
        locale,
        lang,
    )
    class_code = dict_to_class_code(
        result,
        "_Locale",
        "Locale",
        True,
    )
    save_locale_file(
        f"""
from package_utils.locale.base import Locale

locale_name = "{lang}"
locale_display_name = "{locale_to_display_name[lang]}"

"""
        + class_code,
        locale_to_path[lang],
    )
    print(f"Saved {lang}.py")
