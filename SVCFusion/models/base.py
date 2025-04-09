import gradio as gr

from SVCFusion.inference.vocoders import NsfHifiGAN, get_shared_vocoder


class BaseSVCModel:
    model_name = ...

    def infer(self, params, progress=gr.Progress()): ...

    def train(self, params, progress=gr.Progress()): ...

    def preprocess(self, params, progress=gr.Progress()): ...

    def pack_model(self, model_dict): ...
    def install_model(self, package: dict, model_name: str): ...

    def model_filter(self, filepath: str): ...

    def load_model(self, args): ...
    def unload_model(self): ...

    infer_form = {}
    train_form = {}
    preprocess_form = {}

    model_types: dict[str, str] = {}

    @property
    def vocoder(self) -> NsfHifiGAN:
        return get_shared_vocoder()

    def __init__(self) -> None:
        pass
