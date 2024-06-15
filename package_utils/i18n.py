from .locale.base import Locale
from .locale import locale_dict

lang = "zh-cn"

"""
国际化
"""
I: Locale = locale_dict[lang]  # noqa: E741
