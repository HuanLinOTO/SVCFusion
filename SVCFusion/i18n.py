from SVCFusion.config import system_config
from .locale.base import Locale
from .locale import locale_dict, text_to_locale

lang = text_to_locale[system_config.pkg.lang]

"""
国际化
"""
I: Locale = locale_dict[lang]  # noqa: E741
