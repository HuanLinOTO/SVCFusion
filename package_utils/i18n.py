from package_utils.config import get_settings
from .locale.base import Locale
from .locale import locale_dict, text_to_locale

lang = text_to_locale[get_settings()["lang"]]

"""
国际化
"""
I: Locale = locale_dict[lang]  # noqa: E741
