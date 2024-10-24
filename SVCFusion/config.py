import json
import os
import yaml


class YAMLReader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.file = None
        self.data = None

    def __enter__(self):
        try:
            self.file = open(self.file_path, "r", encoding="utf-8")
            self.data = yaml.safe_load(self.file)
        except Exception:
            self.file = open(self.file_path, "r", encoding="gbk")
            self.data = yaml.safe_load(self.file)
        return self.data

    def __exit__(self, exc_type, exc_value, traceback):
        if self.file:
            self.file.close()


class JSONReader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.file = None
        self.data = None

    def __enter__(self):
        try:
            self.file = open(self.file_path, "r", encoding="utf-8")
            self.data = json.load(self.file)
        except Exception:
            self.file = open(self.file_path, "r", encoding="gbk")
            self.data = json.load(self.file)
        return self.data

    def __exit__(self, exc_type, exc_value, traceback):
        if self.file:
            self.file.close()


def writeConfig(config_path: str, config: dict, format: str = "json") -> None:
    if format == "json":
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
        except UnicodeEncodeError:
            with open(config_path, "w", encoding="gbk") as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
    elif format == "yaml":
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
        except UnicodeEncodeError:
            with open(config_path, "w", encoding="gbk") as f:
                yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
    else:
        raise ValueError("Invalid format.")


def applyChanges(
    config_path: str,
    changes: dict[str, any],
    no_skip: bool = False,
) -> dict:
    if config_path.endswith(".json"):
        with JSONReader(config_path) as c:
            config = c
    else:
        with YAMLReader(config_path) as c:
            config = c

    for key, value in changes.items():
        if key.startswith("#"):
            continue
        if value is None or "." not in key and not no_skip:
            continue
        keys = key.split(".")
        d = config
        for k in keys[:-1]:
            if k not in d:
                d[k] = {}
            d = d[k]
        d[keys[-1]] = value
    writeConfig(config_path, config, format=config_path.split(".")[-1])
    return config


def get_settings():
    if not os.path.exists("configs/svcfusion.json"):
        # 写个 {} 进去
        with open("configs/svcfusion.json", "w") as f:
            f.write("{}")
    with JSONReader("configs/svcfusion.json") as config:
        return config


class DefaultSystemConfig:
    class pkg:
        lang = "简体中文"

    class infer:
        msst_device = "cuda:0"

    class sovits:
        resolve_port_clash = False

    class ddsp6:
        pretrained_model_preference = 0

    class ddsp6_1:
        pretrained_model_preference = 0


class SystemConfig(dict):
    def __init__(self, *args, **kwargs):
        self.default_class = kwargs.pop("default_class", None)
        super(SystemConfig, self).__init__(*args, **kwargs)

    def __getattr__(self, item):
        settings = get_settings()
        if item in self:
            value = self[item]
        elif item in settings:
            value = settings[item]
            self[item] = value  # 缓存设置值
        else:
            value = None

        if value is None and self.default_class:
            value = getattr(self.default_class, item, None)
            if value is not None:
                if isinstance(value, type):
                    return SystemConfig(default_class=value)
                return value

        if value is None:
            raise AttributeError(f"'SystemConfig' object has no attribute '{item}'")

        if isinstance(value, dict) and not isinstance(value, SystemConfig):
            value = SystemConfig(value, default_class=self.default_class)
            self[item] = value

        return value

    def __setattr__(self, item, value):
        self[item] = value

    def __delattr__(self, item):
        del self[item]

    def __setitem__(self, key, value):
        if isinstance(value, dict) and not isinstance(value, SystemConfig):
            value = SystemConfig(value, default_class=self.default_class)
        super().__setitem__(key, value)

    def __getitem__(self, item):
        if item in self:
            value = super().__getitem__(item)
        else:
            settings = get_settings()
            value = settings.get(item)
            if value is not None:
                self[item] = value
            else:
                raise KeyError(f"'SystemConfig' object has no key '{item}'")

        if isinstance(value, dict) and not isinstance(value, SystemConfig):
            value = SystemConfig(value, default_class=self.default_class)
            self[item] = value

        return value


system_config: DefaultSystemConfig = SystemConfig(
    get_settings(),
    default_class=DefaultSystemConfig,
)

__all__ = [
    "system_config",
]
