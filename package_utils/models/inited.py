from package_utils.models.common import (
    infer_fn_proxy,
    preprocess_fn_proxy,
    train_fn_proxy,
)
from package_utils.models.ddsp import DDSPModel
from package_utils.models.sovits import SoVITSModel
from package_utils.models.reflow import ReflowVAESVCModel


ddsp_model = DDSPModel()
sovits_model = SoVITSModel()
reflow_vae_svc_model = ReflowVAESVCModel()

model_list = [ddsp_model, reflow_vae_svc_model, sovits_model]
model_name_list = [model.model_name for model in model_list]


infer_form = {}
for model in model_list:
    infer_form[model.model_name] = {
        "form": model.infer_form,
        "callback": infer_fn_proxy(model.infer),
    }

preprocess_form = {}
for model in model_list:
    preprocess_form[model.model_name] = {
        "form": model.preprocess_form,
        "callback": preprocess_fn_proxy(model.preprocess),
    }


train_form = {}

train_models_dict = {}

for model in model_list:
    for sub_model in model.model_types:
        if model.train_form.get(sub_model) is None:
            continue
        display_name = model.model_name + " - " + model.model_types[sub_model]

        if train_models_dict.get(model.model_name) is None:
            train_models_dict[model.model_name] = []
        train_models_dict[model.model_name].append(display_name)

        train_form[display_name] = {
            "form": model.train_form[sub_model],
            "callback": train_fn_proxy(model.train),
        }
print(train_models_dict)

__all__ = [
    "infer_form",
    "preprocess_form",
    "train_form",
    "model_name_list",
    "train_models_dict",
    "model_list",
]

model_name_list.append("未知模型")
