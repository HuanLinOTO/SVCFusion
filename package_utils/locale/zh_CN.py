from .base import Locale


class zhCNLocale(Locale):
    class model_chooser(Locale.model_chooser):
        submit_btn_value = "选择模型"
