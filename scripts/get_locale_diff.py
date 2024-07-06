import os
import importlib.util


def load_module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


locale_dict: dict[str, dict] = {}
text_to_locale = {}


def merge_dicts(a, b):
    result = a.copy()  # 复制a到结果字典
    for key, value in b.items():
        if key not in result:
            result[key] = b[key]
        elif isinstance(value, dict):
            result[key] = merge_dicts(result[key], b[key])
    return result


for filename in os.listdir(os.path.dirname(__file__)):
    if filename.endswith(".py") and filename not in ["__init__.py", "base.py"]:
        file_path = os.path.join(os.path.dirname(__file__), filename)
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
            text_to_locale[locale_display_name] = locale_name

base_locale = locale_dict["zh-cn"].__dict__
del locale_dict["zh-cn"]
for lang in locale_dict:
    for key in base_locale:
        if key not in locale_dict[lang]:
            locale_dict[lang][key] = base_locale[key]
