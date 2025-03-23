import yaml
import json


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


input_file = r"D:\Projects\SVCFusion\pretrained\sovits\4951da60773534914e7e59ca7b279443\config.yaml"
save_file = input_file


def read_file(file_path):
    if file_path.endswith(".json"):
        with JSONReader(file_path) as config:
            return config
    elif file_path.endswith(".yaml"):
        with YAMLReader(file_path) as config:
            return config


def conv(d: dict) -> dict:
    for key, value in d.copy().items():
        if isinstance(value, dict):
            tmp = conv(value)
            for k, v in tmp.items():
                d[key + "." + k] = v
            del d[key]
    return d


config = read_file(input_file)
result = conv(config)
print(result)

yaml.dump(
    result,
    open(save_file, "w", encoding="utf-8"),
    allow_unicode=True,
    default_flow_style=False,
)
