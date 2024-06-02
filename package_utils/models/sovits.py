import gc
import os
from sys import executable
import time

import torch
import torchaudio
from SoVITS.inference import infer_tool
from SoVITS.inference.infer_tool import Svc
from package_utils.config import JSONReader
from package_utils.dataset_utils import auto_normalize_dataset
from package_utils.exec import exec
from package_utils.ui.FormTypes import FormDictInModelClass
from .common import common_infer_form, common_preprocess_form
import gradio as gr


import soundfile as sf


class SoVITSModel:
    model_name = "So-VITS-SVC"

    _infer_form: FormDictInModelClass = {
        "cluster_infer_ratio": {
            "type": "slider",
            "max": 1,
            "min": 0,
            "default": 0.5,
            "step": 0.1,
            "label": "聚类/特征比例",
            "info": "聚类/特征占比，范围0-1，若没有训练聚类模型或特征检索则默认0即可",
        },
        "linear_gradient": {
            "type": "slider",
            "info": "两段音频切片的交叉淡入长度",
            "label": "渐变长度",
            "default": 0,
            "min": 0,
            "max": 1,
            "step": 0.1,
        },
        "k_step": {
            "type": "slider",
            "max": 1000,
            "min": 1,
            "default": 100,
            "step": 1,
            "label": "扩散步数",
            "info": "越大越接近扩散模型的结果，默认100",
        },
        "enhancer_adaptive_key": {
            "type": "slider",
            "max": 12,
            "min": -12,
            "step": 1,
            "default": 0,
            "label": "增强器适应",
            "info": "使增强器适应更高的音域(单位为半音数)|默认为0",
        },
        "f0_filter_threshold": {
            "type": "slider",
            "max": 1,
            "min": 0,
            "default": 0.05,
            "step": 0.01,
            "label": "f0 过滤阈值",
            "info": "只有使用crepe时有效. 数值范围从0-1. 降低该值可减少跑调概率，但会增加哑音",
        },
        "audio_predict_f0": {
            "type": "checkbox",
            "default": False,
            "info": "语音转换自动预测音高，转换歌声时不要打开这个会严重跑调",
            "label": "自动 f0 预测",
        },
        "second_encoding": {
            "type": "checkbox",
            "default": False,
            "label": "二次编码",
            "info": "浅扩散前会对原始音频进行二次编码，玄学选项，有时候效果好，有时候效果差",
        },
    }

    train_form = {
        "batch_size": {
            "type": "slider",
            "default": 2,
            "label": "训练批次大小",
            "info": "越大越好，越大越占显存，注意不能超过训练集条数",
            "max": 9999,
            "min": 1,
            "step": 1,
        },
    }

    _preprocess_form = {
        "use_diff": {
            "type": "checkbox",
            "default": False,
            "label": "训练浅扩散",
            "info": "勾选后将会生成训练浅扩散需要的文件，会比不选慢",
        }
    }

    model_types = {
        "main": "主模型",
        "diff": "浅扩散模型",
        "cluster": "聚类/检索模型",
    }

    model_chooser_extra_form = {
        "enhance": {
            "type": "checkbox",
            "default": False,
            "label": "NSFHifigan 音频增强",
            "info": "对部分训练集少的模型有一定的音质增强效果，对训练好的模型有反面效果",
        },
        "feature_retrieval": {
            "type": "checkbox",
            "default": False,
            "label": "启用特征提取",
            "info": "是否使用特征检索，如果使用聚类模型将被禁用",
        },
    }

    def model_filter(self, filepath: str):
        if filepath.endswith(".pth"):
            return "main"
        if os.path.basename(filepath) in ["feature_and_index.pkl", "kmeans_10000.pt"]:
            return "cluster"
        if filepath.endswith(".pt"):
            return "diff"

    def unload_model(self):
        if self.svc_model:
            del self.svc_model
        self.svc_model = None
        torch.cuda.empty_cache()
        gc.collect()

    def load_model(self, args):
        print(args)

        main_path = args["main"]
        cluster_path = args["cluster"]
        self.use_cluster = bool(cluster_path)
        if not self.use_cluster:
            cluster_path = ""

        diffusion_model_path = args["diff"]
        self.use_diff = bool(diffusion_model_path)
        if not self.use_diff:
            diffusion_model_path = ""

        device = args["device"]

        if bool(diffusion_model_path):
            diff_config_path = os.path.dirname(diffusion_model_path) + "/config.yaml"
            if not os.path.exists(diff_config_path):
                diff_config_path = (
                    os.path.dirname(diffusion_model_path) + "/diffusion/config.yaml"
                )
        else:
            diff_config_path = None
        self.svc_model = Svc(
            net_g_path=main_path,
            config_path=os.path.dirname(main_path) + "/config.json",
            device=device,
            cluster_model_path=cluster_path,
            nsf_hifigan_enhance=args["enhance"],
            diffusion_model_path=diffusion_model_path,
            diffusion_config_path=diff_config_path,
            shallow_diffusion=self.use_diff,
            only_diffusion=False,
            spk_mix_enable=False,
            feature_retrieval=args["feature_retrieval"],
        )

        with JSONReader(os.path.dirname(main_path) + "/config.json") as config:
            return list(config["spk"].keys())

    def train(self, params):
        print(params)

    def preprocess(self, params, progress=gr.Progress()):
        # aa
        with open("data/model_type", "w") as f:
            f.write("2")
        auto_normalize_dataset("data/44k", False, progress)
        exec(
            f"{executable} -m SoVITS.preprocess_flist_config --source_dir ./data/44k --speech_encoder {params['encoder'].replace('contentvec','vec')}"
        )
        exec(
            f"{executable} -m SoVITS.preprocess_new --f0_predictor {params['f0']} --num_processes 4 {'--use_diff' if params['use_diff'] else ''}"
        )

    def infer(self, params, progress=gr.Progress()):
        wf, sr = torchaudio.load(params["audio"])
        # 重采样到单声道44100hz 保存到 tmp/时间戳_md5前3位.wav
        resampled_filename = f"tmp/{int(time.time())}.wav"
        torchaudio.save(
            uri=resampled_filename,
            src=torchaudio.functional.resample(
                waveform=wf, orig_freq=sr, new_freq=44100
            ),
            sample_rate=sr,
        )

        kwarg = {
            "raw_audio_path": resampled_filename,
            "spk": params["spk"],
            "tran": params["keychange"],
            "slice_db": params["threshold"],
            "cluster_infer_ratio": params["cluster_infer_ratio"]
            if self.use_cluster
            else 0,
            "auto_predict_f0": params["audio_predict_f0"],
            "noice_scale": 0.4,
            "pad_seconds": 0.5,
            "clip_seconds": 0,
            "lg_num": params["linear_gradient"],
            "lgr_num": 0.75,
            "f0_predictor": params["f0"],
            "enhancer_adaptive_key": params["enhancer_adaptive_key"],
            "cr_threshold": params["f0_filter_threshold"],
            "k_step": params["k_step"],
            "use_spk_mix": False,
            "second_encoding": params["second_encoding"],
            "loudness_envelope_adjustment": 1,
        }
        infer_tool.format_wav(params["audio"])
        # self.svc_model.audio16k_resample_transform = torchaudio.transforms.Resample(
        #     self.svc_model.target_sample, 16000
        # ).to(self.svc_model.dev)
        audio = self.svc_model.slice_inference(**kwarg)
        gc.collect()
        torch.cuda.empty_cache()
        sf.write("tmp.wav", audio, 44100)
        # 删掉 filename
        os.remove(resampled_filename)
        print(params)
        return "tmp.wav"

    def __init__(self) -> None:
        self.infer_form = {}
        self.infer_form.update(common_infer_form)
        self.infer_form.update(self._infer_form)

        self.preprocess_form = {}
        self.preprocess_form.update(self._preprocess_form)
        self.preprocess_form.update(common_preprocess_form)

        self.svc_model = None
