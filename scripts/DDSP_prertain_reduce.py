import torch

input_file = r"D:\Projects\SVCFusion\models\3080x20_ddsp_底模\model.pt"

output_file = "./model_0.pt"

model = torch.load(input_file)

del model["model"]["ddsp_model.unit2ctrl.spk_embed.weight"]
model["global_step"] = 0

torch.save(model, output_file)
