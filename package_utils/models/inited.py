from package_utils.models.ddsp import DDSPModel
from package_utils.models.sovits import SoVITSModel
from package_utils.models.reflow import ReflowVAESVCModel


ddsp_model = DDSPModel()
sovits_model = SoVITSModel()
reflow_vae_svc_model = ReflowVAESVCModel()

model_list = [ddsp_model, reflow_vae_svc_model, sovits_model]
model_name_list = [model.model_name for model in model_list]
model_name_list.append("未知模型")


infer_form = {}
for model in model_list:
    infer_form[model.model_name] = {
        "form": model.infer_form,
        "callback": model.infer,
    }

preprocess_form = {}
for model in model_list:
    preprocess_form[model.model_name] = {
        "form": model.preprocess_form,
        "callback": model.preprocess,
    }

train_form = {}
for model in model_list:
    train_form[model.model_name] = {
        "form": model.train_form,
        "callback": model.train,
    }
