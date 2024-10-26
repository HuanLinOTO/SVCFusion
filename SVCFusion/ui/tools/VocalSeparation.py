import os
from pathlib import Path
import shutil
from traceback import print_exception
import gradio as gr

from fap.utils.file import AUDIO_EXTENSIONS
from SVCFusion.i18n import I
from SVCFusion.ui.Form import Form
from SVCFusion.uvr import getVocalAndInstrument


class VocalSeparation:
    def callback(
        self,
        input_audio,
        use_de_reverb,
        use_harmonic_remove,
    ):
        if not input_audio:
            gr.Info(I.vocal_separation.no_file_tip)
            return (
                gr.update(value="tmp/empty.wav"),
                gr.update(value="tmp/empty.wav"),
            )

        vocal, inst = getVocalAndInstrument(
            input_audio,
            True,
            use_de_reverb,
            use_harmonic_remove,
        )
        return (
            gr.update(value=vocal),
            gr.update(value=inst),
        )

    def batch_callback(
        self,
        input_path,
        output_path,
        use_de_reverb,
        use_harmonic_remove,
    ):
        progress = gr.Progress()
        if not input_path:
            gr.Info(I.vocal_separation.no_input_tip)
            return gr.update(value=I.vocal_separation.no_input_tip)
        if not output_path:
            gr.Info(I.vocal_separation.no_output_tip)
            return gr.update(value=I.vocal_separation.no_output_tip)
        if not os.path.exists(input_path):
            gr.Info(I.vocal_separation.input_not_exist_tip)
            return gr.update(value=I.vocal_separation.input_not_exist_tip)
        if not os.path.exists(output_path):
            gr.Info(I.vocal_separation.output_not_exist_tip)
            return gr.update(value=I.vocal_separation.output_not_exist_tip)
        if input_path == output_path:
            gr.Info(I.vocal_separation.input_output_same_tip)
            return gr.update(value=I.vocal_separation.input_output_same_tip)

        vocal_output_dir_path = os.path.join(output_path, "vocal")
        inst_output_dir_path = os.path.join(output_path, "inst")
        os.makedirs(vocal_output_dir_path, exist_ok=True)
        os.makedirs(inst_output_dir_path, exist_ok=True)

        result = ""
        error_files = []
        no_support_files = []
        success_files = []
        for item in progress.tqdm(
            os.listdir(input_path), desc=I.vocal_separation.batch_progress_desc
        ):
            if not isinstance(item, str):
                print("no str path, skip")
                continue
            try:
                item_path = os.path.join(input_path, item)
                if not os.path.isfile(item_path):
                    continue
                if Path(item_path.lower()).suffix in AUDIO_EXTENSIONS:
                    vocal, inst = getVocalAndInstrument(
                        os.path.join(input_path, item),
                        use_de_reverb,
                        use_harmonic_remove,
                        progress,
                    )

                    shutil.copy(vocal, os.path.join(vocal_output_dir_path, item))
                    shutil.copy(inst, os.path.join(inst_output_dir_path, item))

                    success_files.append(item)
                else:
                    no_support_files.append(item)
                    gr.Info(I.vocal_separation.unusable_file_tip.repalce("{1}", item))
            except Exception as e:
                print_exception(e)
                error_files.append(item)

        result += I.vocal_separation.finished + ":\n"
        for item in success_files:
            result += " - " + item + "\n"

        if len(no_support_files) > 0:
            result += I.vocal_separation.unusable_file_tip.replace("{1}", "") + ":\n"
            for item in no_support_files:
                result += " - " + item + "\n"
        if len(error_files) > 0:
            result += I.vocal_separation.error_when_processing + ":\n"
            for item in error_files:
                result += " - " + item + "\n"

        return result

    def __init__(self):
        input_audio = gr.Audio(
            label=I.vocal_separation.input_audio_label,
            interactive=True,
            editable=True,
            type="filepath",
        )

        batch_input_path = gr.Textbox(
            label=I.vocal_separation.input_path_label,
            visible=False,
        )
        batch_output_path = gr.Textbox(
            label=I.vocal_separation.output_path_label,
            visible=False,
        )

        use_batch = gr.Checkbox(
            label=I.vocal_separation.use_batch_label,
            value=False,
        )

        use_de_reverb = gr.Checkbox(
            label=I.vocal_separation.use_de_reverb_label,
            value=True,
        )
        use_harmonic_remove = gr.Checkbox(
            label=I.vocal_separation.use_harmonic_remove_label,
            value=True,
        )

        submit_btn = gr.Button(I.vocal_separation.submit_btn_value)
        batch_submit_btn = gr.Button(
            I.vocal_separation.submit_btn_value,
            visible=False,
        )

        vocal_audio = gr.Audio(
            label=I.vocal_separation.vocal_label,
            type="filepath",
        )
        inst_audio = gr.Audio(
            label=I.vocal_separation.inst_label,
            type="filepath",
        )

        batch_output_message = gr.Textbox(
            label=I.vocal_separation.batch_output_message_label,
            value="",
            visible=False,
        )

        submit_btn.click(
            self.callback,
            inputs=[
                input_audio,
                use_de_reverb,
                use_harmonic_remove,
            ],
            outputs=[vocal_audio, inst_audio],
        )

        batch_submit_btn.click(
            self.batch_callback,
            inputs=[
                batch_input_path,
                batch_output_path,
                use_de_reverb,
                use_harmonic_remove,
            ],
            outputs=[batch_output_message],
        )

        # 默认不展示的
        for comp in [
            batch_input_path,
            batch_output_path,
            batch_output_message,
            batch_submit_btn,
        ]:
            use_batch.change(
                Form.get_change_display_fn(None, True),
                inputs=[use_batch],
                outputs=[
                    comp,
                ],
            )

        # 默认展示的
        for comp in [
            vocal_audio,
            inst_audio,
            input_audio,
            submit_btn,
        ]:
            use_batch.change(
                Form.get_change_display_fn(None, False),
                inputs=[use_batch],
                outputs=[
                    comp,
                ],
            )
