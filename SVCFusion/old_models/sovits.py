import json


class SoVITSModel:
    def __init__(self) -> None:
        pass

    def load_model(self, device, path):
        pass

    def infer_core(
        self,
        audio,
        sample_rate,
        cache_md5,
        keychange,
        method,
        f0_extractor,
        threhold,
        num_formant_shift_key,
        spk,
        progress,
    ):
        pass

    def save_config(
        device,
        batchsize,
        num_workers,
        amp_dtype,
        lr,
        cache_device,
        cache_all,
        interval_val,
        interval_force_save,
        interval_log,
        train_epoch,
        gamma,
        # 0 主模型 1 浅扩散
        model_type=0,
    ):
        # 读取 configs/sovits_template.json
        with open("configs/sovits_template.json", "r") as f:
            config = json.load(f)
        config["train"]["batch_size"] = batchsize
        if not amp_dtype == "fp32":
            config["train"]["fp16_run"] = True
        config["train"]["half_type"] = amp_dtype
        config["train"]["learning_rate"] = lr
