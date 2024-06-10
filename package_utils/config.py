import json
import yaml


class YAMLReader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.file = None
        self.data = None

    def __enter__(self):
        try:
            self.file = open(self.file_path, "r", encoding="utf-8")
        except UnicodeDecodeError:
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
        self.file = open(self.file_path, "r", encoding="utf-8")
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


def applyChanges(config_path: str, changes: dict[str, any]) -> dict:
    if config_path.endswith(".json"):
        with JSONReader(config_path) as c:
            config = c
    else:
        with YAMLReader(config_path) as c:
            config = c

    for key, value in changes.items():
        if value is None or "." not in key:
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
