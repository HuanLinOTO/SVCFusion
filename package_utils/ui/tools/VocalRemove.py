import gradio as gr

from package_utils.i18n import I
from package_utils.uvr import getVocalAndInstrument


class VocalRemove:
    def callback(self, input_audio):
        vocal, inst = getVocalAndInstrument(input_audio)
        return (
            gr.update(value=vocal),
            gr.update(value=inst),
        )

    def __init__(self):
        input_audio = gr.Audio(
            label=I.vocal_remove.input_audio_label,
            interactive=True,
            editable=True,
            type="filepath",
        )
        submit_btn = gr.Button(I.vocal_remove.submit_btn_value)

        vocal_audio = gr.Audio(
            label=I.vocal_remove.vocal_label,
            type="filepath",
        )
        inst_audio = gr.Audio(
            label=I.vocal_remove.inst_label,
            type="filepath",
        )

        submit_btn.click(
            self.callback,
            inputs=[input_audio],
            outputs=[vocal_audio, inst_audio],
        )
