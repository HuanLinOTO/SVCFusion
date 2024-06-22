import gradio as gr

from package_utils.i18n import I


class PathChooser:
    def __init__(self) -> None:
        self.input = gr.TextBox(label=I.path_chooser.input_path_label)
        self.output = gr.TextBox(label=I.path_chooser.output_path_label)
