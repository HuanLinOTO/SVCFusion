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
