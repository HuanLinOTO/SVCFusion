import os
from typing import Callable, Dict, TypedDict
import gradio as gr
from SVCFusion.const_vars import WORK_DIR_PATH
from SVCFusion.i18n import I
from SVCFusion.model_utils import detect_current_model_by_path
from SVCFusion.models.inited import model_name_list, model_list
from SVCFusion.ui.DeviceChooser import DeviceChooser
from SVCFusion.ui.Form import Form
from SVCFusion.ui.FormTypes import FormDict


class ModelDropdownInfo(TypedDict):
    model_type_index: str
    model_type_name: str


class ModelChooser:
    def result_normalize(self, result):
        for key in result:
            if isinstance(result[key], str) and (
                result[key].endswith(I.model_chooser.unuse_value)
                or result[key].endswith(I.model_chooser.no_model_value)
            ):
                result[key] = None
        return result

    # def get_on_topic_result(): ...

    def refresh_search_paths(self):
        self.search_paths = [
            WORK_DIR_PATH,
            *[
                "archive/" + p
                for p in os.listdir("archive")
                if os.path.isdir(os.path.join("archive", p))
            ],
            *[
                "models/" + p
                for p in os.listdir("models")
                if os.path.isdir(os.path.join("models", p))
            ],
        ]
        self.choices = self.search_path_to_text()

    def search_path_to_text(self):
        return [
            I.model_chooser.workdir_name,
            *[
                p.replace("models/", f"{I.model_chooser.models_dir_name} - ").replace(
                    "archive/", f"{I.model_chooser.archive_dir_name} - "
                )
                for p in self.search_paths
                if not p.startswith("exp")
            ],
        ]

    def get_search_path_val(self):
        self.refresh_search_paths()
        return gr.update(
            choices=self.choices,
            value=self.choices[0],
        )

    def get_models_from_search_path(self, search_path):
        # os 扫描目录 获取模型
        model_type_index = detect_current_model_by_path(search_path, alert=True)
        result = {}
        for p in os.listdir(search_path):
            if os.path.isfile(os.path.join(search_path, p)):
                model_type = model_list[model_type_index].model_filter(p)
                if result.get(model_type) is None:
                    result[model_type] = []
                result[model_type].append(p)
        if os.path.exists(search_path + "/diffusion") and os.path.isdir(
            search_path + "/diffusion"
        ):
            for p in os.listdir(search_path + "/diffusion"):
                if os.path.isfile(os.path.join(search_path + "/diffusion", p)):
                    model_type = model_list[model_type_index].model_filter(
                        search_path + "/diffusion/" + p
                    )
                    if result.get(model_type) is None:
                        result[model_type] = []
                    result[model_type].append("diffusion/" + p)

        return result

    selected_search_path = ""

    search_path_handlers = []

    def on_search_path_change(self, handler):
        self.search_path_handlers.append(handler)

    def update_search_path(self, search_path):
        self.selected_search_path = search_path
        for handler in self.search_path_handlers:
            handler(search_path)

    def update_selected(self, search_path, device, *params_values):
        search_path = self.search_paths[search_path]
        model_type_index = detect_current_model_by_path(search_path)

        model_dropdown_values = params_values[: len(self.dropdown_index2model_info)]
        extra_form_values = params_values[len(self.dropdown_index2model_info) :]

        result = {}
        on_topic_extra_form_values = {}

        for i in range(len(extra_form_values)):
            info = self.extra_form.index2param_info[i]
            if info["model_name"] == model_name_list[model_type_index]:
                on_topic_extra_form_values[info["key"]] = extra_form_values[i]

        device = DeviceChooser.get_device_str_from_index(device)

        for i in range(len(model_dropdown_values)):
            model_dropdown = model_dropdown_values[i]
            model_info = self.dropdown_index2model_info[i]
            if model_info["model_type_index"] == model_type_index:
                # print(model_info["model_type_name"], model_dropdown)
                result[model_info["model_type_name"]] = os.path.join(
                    search_path, model_dropdown
                )
            i += 1
        result["device"] = device
        result.update(on_topic_extra_form_values)

        result = self.result_normalize(result)
        self.selected_parameters = result
        self.seleted_model_type_index = model_type_index

    def on_submit(self):
        spks = self.submit_func(self.seleted_model_type_index, self.selected_parameters)
        if spks is None:
            spks = []
        return gr.update(
            choices=spks if len(spks) > 0 else [I.model_chooser.no_spk_value],
            value=spks[0] if len(spks) > 0 else I.model_chooser.no_spk_value,
        )

    def on_refresh(self, search_path):
        search_path = self.search_paths[search_path]
        self.update_search_path(search_path)

        models = self.get_models_from_search_path(search_path)
        model_type_index = detect_current_model_by_path(search_path)

        result = []
        i = 0
        for model in model_list:
            for type in model.model_types:
                m: list = models.get(type, [I.model_chooser.no_model_value])
                if I.model_chooser.unuse_value not in m:
                    m.append(I.model_chooser.unuse_value)
                result.append(
                    gr.update(
                        choices=m,
                        value=m[0] if len(m) > 0 else I.model_chooser.no_model_value,
                        visible=self.dropdown_index2model_info[i]["model_type_index"]
                        == model_type_index,
                    )
                )
                i += 1
        # print(len(result))
        return (
            *result,
            gr.update(
                value=model_name_list[model_type_index],
                interactive=model_type_index == -1,
            ),
            gr.update(visible=model_type_index != -1 and self.show_submit_button),
        )

    def on_refresh_with_search_path(self, search_path):
        if not search_path:
            result = [gr.update() for i in range((len(self.model_dropdowns) + 3))]
            return result
        result = self.on_refresh(search_path)
        return (
            gr.update(
                choices=self.choices,
                value=search_path,
            ),
            *result,
        )

    def __init__(
        self,
        on_submit: Callable = lambda *x: None,
        show_options=True,
        show_submit_button=True,
        submit_btn_text=I.model_chooser.submit_btn_value,
    ) -> None:
        self.submit_func = on_submit
        self.show_submit_button = show_submit_button

        self.search_paths = self.refresh_search_paths()
        self.seach_path_dropdown = gr.Dropdown(
            label=I.model_chooser.search_path_label,
            value=self.get_search_path_val,
            type="index",
            interactive=True,
            allow_custom_value=True,
        )
        models = self.get_models_from_search_path(self.search_paths[0])
        self.model_dropdowns = []
        self.dropdown_index2model_info: Dict[int, ModelDropdownInfo] = {}
        is_first_model = True
        i = 0
        extra_form = {}

        for model in model_list:
            for type in model.model_types:
                m = models.get(type, [I.model_chooser.no_model_value])
                self.model_dropdowns.append(
                    gr.Dropdown(
                        label=f"{I.model_chooser.choose_model_dropdown_prefix} - "
                        + model.model_types[type],
                        choices=m,
                        value=m[0] if len(m) > 0 else I.model_chooser.no_model_value,
                        interactive=True,
                        visible=is_first_model,
                    )
                )
                self.dropdown_index2model_info[i] = {
                    "model_type_index": model_name_list.index(model.model_name),
                    "model_type_name": type,
                }
                is_first_model = False
                i += 1

            if hasattr(model, "model_chooser_extra_form"):
                model_chooser_extra_form: FormDict = model.model_chooser_extra_form
                extra_form[model.model_name] = {
                    "form": model_chooser_extra_form,
                    "callback": lambda: None,
                }

        with gr.Group():
            self.refresh_btn = gr.Button(
                I.model_chooser.refresh_btn_value,
                interactive=True,
            )
        with gr.Group():
            with gr.Row():
                self.model_type_dropdown = gr.Dropdown(
                    label=I.model_chooser.model_type_dropdown_label,
                    choices=model_name_list,
                    interactive=False,
                )
                self.device_chooser = DeviceChooser(show=show_options)
            self.spk_dropdown = gr.Dropdown(
                label=I.model_chooser.spk_dropdown_label,
                choices=[I.model_chooser.no_spk_option],
                value=I.model_chooser.no_spk_option,
                interactive=True,
                visible=show_options,
            )
        if len(extra_form) > 0 and show_options:
            self.extra_form = Form(
                triger_comp=self.model_type_dropdown,
                models=extra_form,
                show_submit=False,
                vertical=True,
            )

        self.load_model_btn = gr.Button(
            submit_btn_text,
            variant="primary",
            interactive=True,
            visible=show_submit_button,
        )

        self.seach_path_dropdown.change(
            self.on_refresh,
            [self.seach_path_dropdown],
            [
                *self.model_dropdowns,
                self.model_type_dropdown,
                self.load_model_btn,
            ],
        )

        self.refresh_btn.click(
            self.on_refresh,
            [self.seach_path_dropdown],
            [
                *self.model_dropdowns,
                self.model_type_dropdown,
                self.load_model_btn,
            ],
        )

        self.load_model_btn.click(
            self.on_submit,
            outputs=[self.spk_dropdown],
        )

        for item in [
            self.seach_path_dropdown,
            self.model_type_dropdown,
            *self.model_dropdowns,
            *(self.extra_form.param_comp_list if hasattr(self, "extra_form") else []),
        ]:
            item.change(
                self.update_selected,
                inputs=[
                    self.seach_path_dropdown,
                    self.device_chooser.device_dropdown,
                    *self.model_dropdowns,
                    *(
                        self.extra_form.param_comp_list
                        if hasattr(self, "extra_form")
                        else []
                    ),
                ],
            )
