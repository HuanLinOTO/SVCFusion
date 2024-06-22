import gradio as gr

from package_utils.ui.PathChooser import PathChooser


class AudioSlicer:
    def __init__(self) -> None:
        path_chooser = PathChooser()
        max_duration = gr.Slider(minimum=1, maximum=60, value=10, label="最大时长")
        submit = gr.Button("开始")


class Resample:
    def __init__(self) -> None:
        path_chooser = PathChooser()
        submit = gr.Button("开始")


class ToWAV:
    def __init__(self) -> None:
        path_chooser = PathChooser()
        submit = gr.Button("开始")


class FishAudioPreprocess:
    def __init__(self) -> None:
        with gr.Tabs():
            with gr.TabItem("切音机"):
                AudioSlicer()
            with gr.TabItem("重采样到 44.1 khz"):
                Resample()
            with gr.TabItem("批量转 WAV"):
                ToWAV()
