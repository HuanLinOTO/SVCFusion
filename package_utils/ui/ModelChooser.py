import os
from typing import Callable, Dict, TypedDict
import gradio as gr
from package_utils.const_vars import WORK_DIR_PATH
from package_utils.i18n import I
from package_utils.model_utils import detect_current_model_by_path
from package_utils.models.inited import model_name_list, model_list
from package_utils.ui.DeviceChooser import DeviceChooser
from package_utils.ui.Form import Form
from package_utils.ui.FormTypes import FormDict


class ModelDropdownInfo(TypedDict):
    model_type_index: str
    model_type_name: str


class ModelChooser:
    def result_normalize(self, result):
        for key in result:
            if isinstance(result[key], str) and (
                result[key].endswith("不使用") or result[key].endswith("无模型")
            ):
                result[key] = None
        return result

    def get_on_topic_result(): ...

    def refresh_search_paths(self):
        self.search_paths = [
            WORK_DIR_PATH,
            *[
                "archieve/" + p
                for p in os.listdir("archieve")
                if os.path.isdir(os.path.join("archieve", p))
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
            "工作目录",
            *[
                p.replace("models/", "models 文件夹 - ").replace(
                    "archieve/", "已归档训练 - "
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
        model_type_index = detect_current_model_by_path(search_path)
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

    def update_search_path(self, search_path):
        self.selected_search_path = search_path

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
        # Q:我这个变量用于储存已选的参数，应该叫啥名字
        self.selected_parameters = result
        self.seleted_model_type_index = model_type_index

    def on_submit(self):
        spks = self.submit_func(self.seleted_model_type_index, self.selected_parameters)
        if spks is None:
            spks = []
        return gr.update(
            choices=spks if len(spks) > 0 else ["无说话人"],
            value=spks[0] if len(spks) > 0 else "无说话人",
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
                m = models.get(type, ["无模型"])
                m.append("不使用")
                result.append(
                    gr.update(
                        choices=m,
                        value=m[0] if len(m) > 0 else "无模型",
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

    def __init__(
        self,
        on_submit: Callable = lambda *x: None,
        show_options=True,
        show_submit_button=True,
    ) -> None:
        self.submit_func = on_submit
        self.show_submit_button = show_submit_button

        self.search_paths = self.refresh_search_paths()
        self.seach_path_dropdown = gr.Dropdown(
            label="搜索路径",
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
                m = models.get(type, ["无模型"])
                self.model_dropdowns.append(
                    gr.Dropdown(
                        label="选择模型-" + model.model_types[type],
                        choices=m,
                        value=m[0] if len(m) > 0 else "无模型",
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
                "刷新选项",
                interactive=True,
            )
        with gr.Group():
            with gr.Row():
                self.model_type_dropdown = gr.Dropdown(
                    label="模型类型",
                    choices=model_name_list,
                    interactive=False,
                )
                self.device_chooser = DeviceChooser(show=show_options)
            self.spk_dropdown = gr.Dropdown(
                label="选择说话人",
                choices=["未加载模型"],
                value="未加载模型",
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
            I.model_chooser.submit_btn_value,
            variant="primary",
            interactive=True,
            visible=show_submit_button,
        )

        self.seach_path_dropdown.change(
            self.on_refresh,
            [self.seach_path_dropdown],
            [*self.model_dropdowns, self.model_type_dropdown, self.load_model_btn],
        )

        self.refresh_btn.click(
            self.on_refresh,
            [self.seach_path_dropdown],
            [*self.model_dropdowns, self.model_type_dropdown, self.load_model_btn],
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
