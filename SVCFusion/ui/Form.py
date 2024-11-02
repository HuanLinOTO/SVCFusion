from typing import Callable, Dict
import gradio as gr
from SVCFusion.i18n import I
from SVCFusion.ui.DeviceChooser import DeviceChooser

from .FormTypes import FormComponent, FormDict, ParamInfo


class Form:
    def get_change_display_fn(self, reserve):
        def fn(value):
            return gr.update(visible=value if reserve else not value)

        return fn

    def change_model(self, model):
        # 如果 self.model_name_list 不存在 model，index 给个 -1
        if model not in self.model_name_list:
            index = -1
        else:
            index = self.model_name_list.index(model)
        result = []
        for i in range(len(self.model_name_list)):
            result.append(gr.update(visible=i == index))
            # result.append(gr.update(visible=True))
        if len(result) == 1:
            return result[0]
        return result

    def get_fn(self, model_name):
        cb = self.models[model_name]["callback"]
        form = self.models[model_name]["form"]

        def fn(*args):
            result = {}
            i = 0
            for key in form:
                # dropdown_liked_checkbox
                if form[key]["type"] == "device_chooser":
                    result[key] = DeviceChooser.get_device_str_from_index(args[i])
                elif form[key]["type"] == "dropdown_liked_checkbox":
                    result[key] = args[i] == I.form.dorpdown_liked_checkbox_yes
                else:
                    result[key] = args[i]
                i += 1

            for key in self.extra_inputs_keys:
                result[key] = args[i]
            result["_model_name"] = model_name
            return cb(result, progress=gr.Progress())

            # print(cb, result)

        return fn

    def parse_item(self, item: FormComponent):
        if item["type"] == "slider":
            comp = gr.Slider(
                label=item["label"],
                info=item["info"],
                minimum=item["min"],
                maximum=item["max"],
                value=item["default"],
                step=item["step"],
                interactive=True,
                visible=item.get("visible", True),
            )
        elif item["type"] == "dropdown":
            comp = gr.Dropdown(
                label=item["label"],
                info=item.get("info", None),
                choices=item["choices"],
                value=item["default"],
                type=item.get("value_type", "value"),
                interactive=True,
                visible=item.get("visible", True),
            )
        elif item["type"] == "dropdown_liked_checkbox":
            if isinstance(item["default"], Callable):

                def value_proxy():
                    return (
                        I.form.dorpdown_liked_checkbox_yes
                        if item["default"]()
                        else I.form.dorpdown_liked_checkbox_no
                    )
            else:

                def value_proxy():
                    return (
                        I.form.dorpdown_liked_checkbox_yes
                        if item["default"]
                        else I.form.dorpdown_liked_checkbox_no
                    )

            comp = gr.Dropdown(
                label=item["label"],
                info=item["info"],
                choices=[
                    I.form.dorpdown_liked_checkbox_yes,
                    I.form.dorpdown_liked_checkbox_no,
                ],
                value=value_proxy,
                interactive=True,
                visible=item.get("visible", True),
            )
        elif item["type"] == "checkbox" or item["type"] == "show_switch":
            comp = gr.Checkbox(
                label=item["label"],
                info=item.get("info", None),
                value=item["default"],
                interactive=True,
                visible=item.get("visible", True),
            )
        elif item["type"] == "audio":
            comp = gr.Audio(
                label=item["label"],
                type="filepath",
                interactive=True,
                visible=item.get("visible", True),
                editable=True,
            )
        elif item["type"] == "device_chooser":
            comp = DeviceChooser(
                show=True,
                info=item.get("info", None),
            ).device_dropdown
        elif item["type"] == "file":
            comp = gr.Files(
                label=item["label"],
                type="filepath",
                interactive=True,
                visible=item.get("visible", True),
            )
        else:
            raise Exception("未知类型", item)

        return comp

    def __init__(
        self,
        triger_comp: gr.component,
        models: FormDict | Callable[..., FormDict],
        extra_inputs: Dict[str, gr.component] = {},
        vertical=False,
        use_audio_opt=False,
        use_textbox_opt=False,
        show_submit=True,
        submit_btn_text=I.form.submit_btn_value,
        default_display_key="",
    ) -> None:
        self.models = models
        self.vertical = vertical
        self.model_name_list = []

        if not default_display_key:
            default_display_key = list(models.keys())[0]

        self.index2param_info: Dict[int, ParamInfo] = {}
        self.key2param_info: Dict[str, ParamInfo] = {}

        self.param_comp_list = []

        groups = []
        is_fisrt_group = True
        total_i = 0
        for model_name in models:
            self.model_name_list.append(model_name)
            form = models[model_name]["form"]
            group = gr.Group(visible=model_name == default_display_key)
            groups.append(group)
            with group:
                items = []
                num_grad = 0
                if len(items) % 4 == 0:
                    num_grad = 4
                elif len(items) % 3 == 0:
                    num_grad = 3
                elif len(items) % 2 == 0:
                    num_grad = 2
                i = 0
                if not vertical:
                    row = gr.Row()
                    row.__enter__()
                for key in form:
                    i += 1
                    item = form[key]
                    comp = self.parse_item(item)
                    # if comp is None:
                    #     print("comp is None", item)
                    items.append(comp)

                    self.param_comp_list.append(comp)
                    self.index2param_info[total_i] = {
                        "model_name": model_name,
                        "key": key,
                        "comp": comp,
                        "info": item,
                    }
                    self.key2param_info[key] = {
                        "model_name": model_name,
                        "key": key,
                        "comp": comp,
                        "info": item,
                    }

                    total_i += 1

                    if item["type"] == "audio" or item.get("individual", False) is True:
                        i = 0
                        if not vertical:
                            row.__exit__()
                            row = gr.Row()
                            row.__enter__()
                    if i % num_grad == 0 and not vertical:
                        row.__exit__()
                        row = gr.Row()
                        row.__enter__()

                    if item.get("addition_tip_when_update", False):
                        comp.change(
                            item["addition_tip_when_update"],
                            inputs=[comp],
                            outputs=[gr.HTML()],
                        )
                        row.__exit__()
                        row = gr.Row()
                        row.__enter__()

                if not vertical:
                    row.__exit__()
                submit = gr.Button(
                    submit_btn_text, visible=show_submit, variant="primary"
                )
                inputs = [*items]
                self.extra_inputs_keys = []
                for key in extra_inputs:
                    self.extra_inputs_keys.append(key)
                    inputs.append(extra_inputs[key])
                # print("group", model_name, group._id)
                if use_audio_opt:
                    audio_output_1 = gr.Audio(
                        type="filepath",
                        label=I.form.audio_output_1,
                    )
                    audio_output_2 = gr.Audio(
                        type="filepath",
                        label=I.form.audio_output_2,
                        visible=False,
                    )
                    audio_output_batch_1 = gr.Files(
                        type="filepath",
                        label=I.form.audio_output_1,
                        visible=False,
                    )
                    audio_output_batch_2 = gr.Files(
                        type="filepath",
                        label=I.form.audio_output_2,
                        visible=False,
                    )
                outputs = []
                if use_audio_opt:
                    outputs.append(audio_output_1)
                    outputs.append(audio_output_2)
                    outputs.append(audio_output_batch_1)
                    outputs.append(audio_output_batch_2)
                if use_textbox_opt:
                    outputs.append(gr.Textbox(label=I.form.textbox_output))
                submit.click(
                    self.get_fn(model_name),
                    inputs=inputs,
                    outputs=outputs,
                )

                # 后处理

                for key in self.key2param_info:
                    item = self.key2param_info[key]["info"]
                    if item["type"] == "show_switch":
                        comp: gr.Checkbox = self.key2param_info[key]["comp"]
                        for sub_comp in item["default_show"]:
                            comp.change(
                                self.get_change_display_fn(False),
                                inputs=[comp],
                                outputs=[
                                    self.key2param_info[sub_comp]["comp"],
                                ],
                            )
                        for sub_comp in item["other_show"]:
                            comp.change(
                                self.get_change_display_fn(True),
                                inputs=[comp],
                                outputs=[
                                    self.key2param_info[sub_comp]["comp"],
                                ],
                            )

        triger_comp.change(self.change_model, inputs=[triger_comp], outputs=groups)
