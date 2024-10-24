import os
import gradio as gr

from SVCFusion.dataset_utils import resample, slice_audio, to_wav
from SVCFusion.i18n import I
from SVCFusion.ui.PathChooser import PathChooser


def vef_path(path):
    if not os.path.exists(path):
        gr.Info(I.fish_audio_preprocess.input_path_not_exist_tip)
        return False
    return True


class AudioSlicer:
    def callback(self, input, output, max_duration):
        if not vef_path(input):
            return
        if input == output:
            gr.Info(I.fish_audio_preprocess.input_output_same_tip)
            return
        slice_audio(input, output, max_duration)

    def __init__(self) -> None:
        path_chooser = PathChooser()
        max_duration = gr.Slider(
            minimum=1,
            maximum=60,
            value=10,
            label="最大时长(/s)",
            interactive=True,
        )
        submit = gr.Button(I.fish_audio_preprocess.submit_btn_value)

        submit.click(
            self.callback,
            inputs=[path_chooser.input, path_chooser.output, max_duration],
        )


class Resample:
    def callback(self, input, output):
        if not vef_path(input):
            return
        resample(input, output)

    def __init__(self) -> None:
        path_chooser = PathChooser()

        submit = gr.Button(I.fish_audio_preprocess.submit_btn_value)
        submit.click(
            self.callback,
            inputs=[path_chooser.input, path_chooser.output],
        )


class ToWAV:
    def callback(self, input, output):
        if not vef_path(input):
            return
        to_wav(input, output)

    def __init__(self) -> None:
        path_chooser = PathChooser()
        submit = gr.Button(I.fish_audio_preprocess.submit_btn_value)

        submit.click(
            self.callback,
            inputs=[path_chooser.input, path_chooser.output],
        )


class FishAudioPreprocess:
    def __init__(self) -> None:
        gr.Markdown("过程请去查看控制台")
        with gr.Tabs():
            with gr.TabItem("切音机"):
                AudioSlicer()
            with gr.TabItem("重采样到 44.1 khz"):
                Resample()
            with gr.TabItem("批量转 WAV"):
                ToWAV()
