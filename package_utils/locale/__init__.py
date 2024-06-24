from package_utils.locale.en_US import enUSLocale
from package_utils.locale.zh_CN import zhCNLocale


locale_dict = {
    "zh-cn": zhCNLocale,
    "en-us": enUSLocale,
}

text_to_locale = {
    "简体中文": "zh-cn",
    "English": "en-us",
}

__all__ = ["locale_dict", "text_to_locale"]
