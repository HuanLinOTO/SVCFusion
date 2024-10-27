import gc
import json
import os
import pickle
import sys

import yaml
from SoVITS.compress_model import copyStateDict
from SoVITS.models import SynthesizerTrn
from fap.utils.file import make_dirs
from SVCFusion.exec import executable
import time

import torch
import torchaudio
from SoVITS.inference import infer_tool
from SoVITS.inference.infer_tool import Svc
from SVCFusion.config import JSONReader, YAMLReader, applyChanges, system_config
from SVCFusion.const_vars import WORK_DIR_PATH
from SVCFusion.dataset_utils import auto_normalize_dataset
from SVCFusion.exec import exec, start_with_cmd
from SVCFusion.i18n import I
from SVCFusion.model_utils import get_pretrain_models_form_item, load_pretrained
from SVCFusion.ui.FormTypes import FormDictInModelClass
from .common import common_infer_form, common_preprocess_form
from SoVITS import logger, utils
import gradio as gr


import soundfile as sf


def check_files(directory, use_diff=False):
    # 获取目标目录下所有的wav文件
    wav_files = [f for f in os.listdir(directory) if f.endswith(".wav")]

    missing_files = []

    for wav_file in wav_files:
        base_name = os.path.splitext(wav_file)[0]

        spec_file = os.path.join(directory, f"{base_name}.spec.pt")
        f0_file = os.path.join(directory, f"{base_name}.wav.f0.npy")
        soft_file = os.path.join(directory, f"{base_name}.wav.soft.pt")

        # 以上三个变量缺一个把这个变量就扔进 missingfiles
        for file in [spec_file, f0_file, soft_file]:
            if not os.path.exists(file):
                missing_files.append(file)

        if use_diff:
            mel_file = os.path.join(directory, f"{base_name}.mel.npy")
            vol_file = os.path.join(directory, f"{base_name}.vol.npy")

            for file in [mel_file, vol_file]:
                if not os.path.exists(file):
                    missing_files.append(file)

    return len(missing_files) == 0


class SoVITSModel:
    model_name = "So-VITS-SVC"

    def get_config_main(*args):
        with JSONReader("configs/sovits.json") as config:
            if config["train"].get("num_workers", None) is None:
                config["train"]["num_workers"] = 2
            return config

    def get_config_diff(*args):
        with YAMLReader("configs/sovits_diff.yaml") as config:
            return config

    _infer_form: FormDictInModelClass = {
        "cluster_infer_ratio": {
            "type": "slider",
            "max": 1,
            "min": 0,
            "default": 0,
            "step": 0.1,
            "label": I.sovits.infer.cluster_infer_ratio_label,
            "info": I.sovits.infer.cluster_infer_ratio_info,
        },
        "linear_gradient": {
            "type": "slider",
            "info": I.sovits.infer.linear_gradient_info,
            "label": I.sovits.infer.linear_gradient_label,
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
            "label": I.sovits.infer.k_step_label,
            "info": I.sovits.infer.k_step_info,
        },
        "enhancer_adaptive_key": {
            "type": "slider",
            "max": 12,
            "min": -12,
            "step": 1,
            "default": 0,
            "label": I.sovits.infer.enhancer_adaptive_key_label,
            "info": I.sovits.infer.enhancer_adaptive_key_info,
        },
        "f0_filter_threshold": {
            "type": "slider",
            "max": 1,
            "min": 0,
            "default": 0.05,
            "step": 0.01,
            "label": I.sovits.infer.f0_filter_threshold_label,
            "info": I.sovits.infer.f0_filter_threshold_info,
        },
        "audio_predict_f0": {
            "type": "checkbox",
            "default": False,
            "info": I.sovits.infer.audio_predict_f0_info,
            "label": I.sovits.infer.audio_predict_f0_label,
        },
        "second_encoding": {
            "type": "checkbox",
            "default": False,
            "label": I.sovits.infer.second_encoding_label,
            "info": I.sovits.infer.second_encoding_info,
        },
        "clip": {
            "type": "slider",
            "max": 100,
            "min": 0,
            "default": 0,
            "step": 1,
            "label": I.sovits.infer.clip_label,
            "info": I.sovits.infer.clip_info,
        },
    }

    train_form = {}

    _preprocess_form = {
        "use_diff": {
            "type": "checkbox",
            "default": False,
            "label": I.sovits.preprocess.use_diff_label,
            "info": I.sovits.preprocess.use_diff_info,
        },
        "vol_aug": {
            "type": "checkbox",
            "default": False,
            "label": I.sovits.preprocess.vol_aug_label,
            "info": I.sovits.preprocess.vol_aug_info,
        },
        "num_workers": {
            "type": "slider",
            "default": 4,
            "label": I.sovits.preprocess.num_workers_label,
            "info": I.sovits.preprocess.num_workers_info,
            "max": 10,
            "min": 1,
            "step": 1,
        },
        "subprocess_num_workers": {
            "type": "slider",
            "default": 4,
            "label": I.sovits.preprocess.subprocess_num_workers_label,
            "info": I.sovits.preprocess.subprocess_num_workers_info,
            "max": 64,
            "min": 1,
            "step": 1,
        },
    }

    model_types = {
        "main": I.sovits.model_types.main,
        "diff": I.sovits.model_types.diff,
        "cluster": I.sovits.model_types.cluster,
    }

    model_chooser_extra_form = {
        "enhance": {
            "type": "checkbox",
            "default": False,
            "label": I.sovits.model_chooser_extra.enhance_label,
            "info": I.sovits.model_chooser_extra.enhance_info,
        },
        "feature_retrieval": {
            "type": "checkbox",
            "default": False,
            "label": I.sovits.model_chooser_extra.feature_retrieval_label,
            "info": I.sovits.model_chooser_extra.feature_retrieval_info,
        },
        "only_diffusion": {
            "type": "checkbox",
            "default": False,
            "label": I.sovits.model_chooser_extra.only_diffusion_label,
            "info": I.sovits.model_chooser_extra.only_diffusion_info,
        },
    }

    def install_model(self, package, model_name):
        model_dict = package["model_dict"]
        config_dict = package["config_dict"]
        base_path = os.path.join("models", model_name)
        make_dirs(base_path)

        if model_dict.get("main", None):
            torch.save(model_dict["main"], os.path.join(base_path, "model.pth"))
            # 将 config_dict["main"] 保存为 config.json
            with open(os.path.join(base_path, "config.json"), "w") as f:
                json.dump(config_dict["main"], f)

        if model_dict.get("diff", None):
            torch.save(model_dict["diff"], os.path.join(base_path, "diff_model.pt"))
            with open(os.path.join(base_path, "config.yaml"), "w") as f:
                yaml.dump(config_dict["diff"], f)

        if model_dict.get("cluster", None):
            if model_dict["cluster"]["type"] == "index":
                pickle.dump(
                    model_dict["cluster"]["model"],
                    open(os.path.join(base_path, "feature_and_index.pkl"), "wb"),
                )
            else:
                torch.save(
                    model_dict["cluster"]["model"],
                    os.path.join(base_path, "kmeans_10000.pt"),
                )

    def removeOptimizer(self, config: str, input_model: dict, ishalf: bool):
        hps = utils.get_hparams_from_file(config)

        net_g = SynthesizerTrn(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model,
        )

        optim_g = torch.optim.AdamW(
            net_g.parameters(),
            hps.train.learning_rate,
            betas=hps.train.betas,
            eps=hps.train.eps,
        )

        state_dict_g = input_model
        new_dict_g = copyStateDict(state_dict_g)
        keys = []
        for k, v in new_dict_g["model"].items():
            if "enc_q" in k:
                continue  # noqa: E701
            keys.append(k)

        new_dict_g = (
            {k: new_dict_g["model"][k].half() for k in keys}
            if ishalf
            else {k: new_dict_g["model"][k] for k in keys}
        )

        return {
            "model": new_dict_g,
            "iteration": 0,
            "optimizer": optim_g.state_dict(),
            "learning_rate": 0.0001,
        }

    def pack_model(self, model_dict):
        print(model_dict)
        result = {}
        result["model_dict"] = {}
        result["config_dict"] = {}
        if model_dict.get("main", None):
            # return result["main"]
            config_path = os.path.dirname(model_dict["main"]) + "/config.json"
            result["model_dict"]["main"] = self.removeOptimizer(
                config_path, torch.load(model_dict["main"], map_location="cpu"), True
            )
            with JSONReader(config_path) as config:
                result["config_dict"]["main"] = config

        if model_dict.get("diff", None):
            result["model_dict"]["diff"] = torch.load(
                model_dict["diff"], map_location="cpu"
            )
            config_path = os.path.dirname(model_dict["diff"]) + "/config.yaml"
            with YAMLReader(config_path) as config:
                result["config_dict"]["diff"] = config

        if model_dict.get("cluster", None):
            if model_dict["cluster"].endswith(".pkl"):
                result["model_dict"]["cluster"] = {
                    "type": "index",
                    "model": pickle.load(open(model_dict["cluster"], "rb")),
                }
            else:
                result["model_dict"]["cluster"] = {
                    "type": "cluster",
                    "model": torch.load(model_dict["cluster"], map_location="cpu"),
                }
        return result

    def model_filter(self, filepath: str):
        if (
            filepath in ["model_0.pt", "diffusion/model_0.pt"]
            or filepath.startswith("D_")
            or filepath.startswith("G_0")
        ):
            return
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
            only_diffusion=args["only_diffusion"],
            spk_mix_enable=False,
            feature_retrieval=args["feature_retrieval"],
        )

        with JSONReader(os.path.dirname(main_path) + "/config.json") as config:
            return list(config["spk"].keys())

    def train(self, params, progress: gr.Progress):
        # print(params)
        sub_model_name = params["_model_name"]
        if sub_model_name == f"So-VITS-SVC - {I.sovits.model_types.main}":
            if not check_files("data/44k"):
                gr.Info(I.sovits.dataset_not_complete_tip)
                return
            working_config_path = os.path.join(WORK_DIR_PATH, "config.json")
            if params["train.half_type"] == "fp16":
                params["train.fp16_run"] = True

            if system_config.sovits.resolve_port_clash and sys.platform == "win32":
                from get_free_port import get_dynamic_ports

                params["train.port"] = str(get_dynamic_ports()[0])
                logger.info(
                    f"Try to resolve port clasling with port {params['train.port']}"
                )
            config = applyChanges(
                working_config_path,
                params,
            )

            pretrained_model_config, is_load_success = load_pretrained(
                "sovits",
                {
                    "vec": config["model"]["speech_encoder"],
                },
                params["#pretrain"],
            )

            if pretrained_model_config:
                config = applyChanges(
                    working_config_path, pretrained_model_config, no_skip=True
                )
            if not is_load_success:
                gr.Info(I.train.load_pretrained_failed_tip)
                return

            start_with_cmd(
                f"{executable} -m SoVITS.train -c {working_config_path} -m workdir"
            )
        elif sub_model_name == f"So-VITS-SVC - {I.sovits.model_types.diff}":
            if not check_files("data/44k", True):
                gr.Info(I.sovits.dataset_not_complete_tip)
                return
            working_config_path = os.path.join(
                WORK_DIR_PATH, "diffusion", "config.yaml"
            )

            config = applyChanges(
                working_config_path,
                params,
            )

            pretrained_model_config, is_load_success = load_pretrained(
                "sovits_diff",
                {
                    "vec": config["data"]["encoder"],
                },
                params["#pretrain"],
            )

            if pretrained_model_config:
                config = applyChanges(
                    working_config_path, pretrained_model_config, no_skip=True
                )
            if not is_load_success:
                gr.Info(I.train.load_pretrained_failed_tip)
                return

            start_with_cmd(
                f"{executable} -m SoVITS.train_diff -c {working_config_path}"
            )
        elif sub_model_name == f"So-VITS-SVC - {I.sovits.model_types.cluster}":
            if params["cluster_or_index"] == "cluster":
                cmd = f"{executable} -m SoVITS.cluster.train_cluster --dataset data/44k"
                if params["use_gpu"]:
                    cmd += " --gpu"
                start_with_cmd(cmd)
            else:
                working_config_path = os.path.join(WORK_DIR_PATH, "config.json")
                start_with_cmd(
                    f"{executable} -m SoVITS.train_index --root_dir data/44k -c {working_config_path}"
                )

    def preprocess(self, params, progress=gr.Progress()):
        # aa
        with open("data/model_type", "w") as f:
            f.write("2")
        auto_normalize_dataset("data/44k", False, progress)
        exec(
            f"{executable} -m SoVITS.preprocess_flist_config --source_dir ./data/44k --speech_encoder {params['encoder'].replace('contentvec','vec')} {'--vol_aug' if params['vol_aug'] else ''}"
        )
        exec(
            f"{executable} -m SoVITS.preprocess_new --f0_predictor {params['f0']} --num_processes {params['num_workers']} --subprocess_num_workers {params['subprocess_num_workers']} {'--use_diff' if params['use_diff'] else ''}"
        )
        return I.sovits.finished

    def infer(self, params, progress=gr.Progress()):
        wf, sr = torchaudio.load(params["audio"])
        # 重采样到单声道44100hz 保存到 tmp/时间戳_md5前3位.wav
        resampled_filename = f"tmp/{int(time.time())}.wav"
        torchaudio.save(
            uri=resampled_filename,
            src=torchaudio.functional.resample(
                waveform=wf, orig_freq=sr, new_freq=44100
            ),
            sample_rate=44100,
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
            "clip_seconds": params["clip"],
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
        self.svc_model.audio16k_resample_transform = torchaudio.transforms.Resample(
            self.svc_model.target_sample, 16000
        ).to(self.svc_model.dev)
        audio = self.svc_model.slice_inference(**kwarg)
        gc.collect()
        torch.cuda.empty_cache()
        sf.write("tmp/infer_opt/" + params["hash"] + ".wav", audio, 44100)
        # 删掉 filename
        os.remove(resampled_filename)
        print(params)
        return "tmp/infer_opt/" + params["hash"] + ".wav"

    def __init__(self) -> None:
        self.infer_form = {}
        self.infer_form.update(common_infer_form)
        self.infer_form.update(self._infer_form)

        self.preprocess_form = {}
        self.preprocess_form.update(self._preprocess_form)
        self.preprocess_form.update(common_preprocess_form)

        # 给 SoVITS 的 logger 挂上gradio progress
        logger.use_gradio_progress = True

        self.train_form.update(
            {
                "main": {
                    **get_pretrain_models_form_item("sovits"),
                    "train.log_interval": {
                        "type": "slider",
                        "default": lambda: self.get_config_main()["train"][
                            "log_interval"
                        ],
                        "label": I.sovits.train_main.log_interval_label,
                        "info": I.sovits.train_main.log_interval_info,
                        "max": 10000,
                        "min": 1,
                        "step": 1,
                    },
                    "train.eval_interval": {
                        "type": "slider",
                        "default": lambda: self.get_config_main()["train"][
                            "eval_interval"
                        ],
                        "label": I.sovits.train_main.eval_interval_label,
                        "info": I.sovits.train_main.eval_interval_info,
                        "max": 10000,
                        "min": 1,
                        "step": 1,
                    },
                    "train.all_in_mem": {
                        "type": "dropdown_liked_checkbox",
                        "default": lambda: self.get_config_main()["train"][
                            "all_in_mem"
                        ],
                        "label": I.sovits.train_main.all_in_mem_label,
                        "info": I.sovits.train_main.all_in_mem_info,
                    },
                    "train.keep_ckpts": {
                        "type": "slider",
                        "default": lambda: self.get_config_main()["train"][
                            "keep_ckpts"
                        ],
                        "label": I.sovits.train_main.keep_ckpts_label,
                        "info": I.sovits.train_main.keep_ckpts_info,
                        "max": 100,
                        "min": 1,
                        "step": 1,
                    },
                    "train.batch_size": {
                        "type": "slider",
                        "default": lambda: self.get_config_main()["train"][
                            "batch_size"
                        ],
                        "label": I.sovits.train_main.batch_size_label,
                        "info": I.sovits.train_main.batch_size_info,
                        "max": 1000,
                        "min": 1,
                        "step": 1,
                    },
                    "train.learning_rate": {
                        "type": "slider",
                        "default": lambda: self.get_config_main()["train"][
                            "learning_rate"
                        ],
                        "label": I.sovits.train_main.learning_rate_label,
                        "info": I.sovits.train_main.learning_rate_info,
                        "max": 1,
                        "min": 0,
                        "step": 0.00001,
                    },
                    "train.num_workers": {
                        "type": "slider",
                        "default": lambda: self.get_config_main()["train"][
                            "num_workers"
                        ],
                        "label": I.sovits.train_main.num_workers_label,
                        "info": I.sovits.train_main.num_workers_info,
                        "max": 10,
                        "min": 1,
                        "step": 1,
                    },
                    "train.half_type": {
                        "type": "dropdown",
                        "default": lambda: self.get_config_main()["train"]["half_type"],
                        "label": I.sovits.train_main.half_type_label,
                        "info": I.sovits.train_main.half_type_info,
                        "choices": ["fp16", "fp32", "bf16"],
                    },
                },
                "diff": {
                    **get_pretrain_models_form_item("sovits_diff"),
                    "train.batchsize": {
                        "type": "slider",
                        "default": lambda: self.get_config_diff()["train"][
                            "batch_size"
                        ],
                        "label": I.sovits.train_diff.batchsize_label,
                        "info": I.sovits.train_diff.batchsize_info,
                        "max": 9999,
                        "min": 1,
                        "step": 1,
                    },
                    "train.num_workers": {
                        "type": "slider",
                        "default": lambda: self.get_config_diff()["train"][
                            "num_workers"
                        ],
                        "label": I.sovits.train_diff.num_workers_label,
                        "info": I.sovits.train_diff.num_workers_info,
                        "max": 9999,
                        "min": 0,
                        "step": 1,
                    },
                    "train.amp_dtype": {
                        "type": "dropdown",
                        "default": lambda: self.get_config_diff()["train"]["amp_dtype"],
                        "label": I.sovits.train_diff.amp_dtype_label,
                        "info": I.sovits.train_diff.amp_dtype_info,
                        "choices": ["fp16", "bf16", "fp32"],
                    },
                    "train.lr": {
                        "type": "slider",
                        "default": lambda: self.get_config_diff()["train"]["lr"],
                        "step": 0.00001,
                        "min": 0.00001,
                        "max": 0.1,
                        "label": I.sovits.train_diff.lr_label,
                        "info": I.sovits.train_diff.lr_info,
                    },
                    "train.interval_val": {
                        "type": "slider",
                        "default": lambda: self.get_config_diff()["train"][
                            "interval_val"
                        ],
                        "label": I.sovits.train_diff.interval_val_label,
                        "info": I.sovits.train_diff.interval_val_info,
                        "max": 10000,
                        "min": 1,
                        "step": 1,
                    },
                    "train.interval_log": {
                        "type": "slider",
                        "default": lambda: self.get_config_diff()["train"][
                            "interval_log"
                        ],
                        "label": I.sovits.train_diff.interval_log_label,
                        "info": I.sovits.train_diff.interval_log_info,
                        "max": 10000,
                        "min": 1,
                        "step": 1,
                    },
                    "train.interval_force_save": {
                        "type": "slider",
                        "label": I.sovits.train_diff.interval_force_save_label,
                        "info": I.sovits.train_diff.interval_force_save_info,
                        "min": 0,
                        "max": 100000,
                        "default": lambda: self.get_config_diff()["train"][
                            "interval_force_save"
                        ],
                        "step": 1000,
                    },
                    "train.gamma": {
                        "type": "slider",
                        "label": I.sovits.train_diff.gamma_label,
                        "info": I.sovits.train_diff.gamma_info,
                        "min": 0,
                        "max": 1,
                        "default": lambda: self.get_config_diff()["train"]["gamma"],
                        "step": 0.1,
                    },
                    "train.cache_device": {
                        "type": "dropdown",
                        "label": I.sovits.train_diff.cache_device_label,
                        "info": I.sovits.train_diff.cache_device_info,
                        "choices": ["cuda", "cpu"],
                        "default": lambda: self.get_config_diff()["train"][
                            "cache_device"
                        ],
                    },
                    "train.cache_all_data": {
                        "type": "dropdown_liked_checkbox",
                        "label": I.sovits.train_diff.cache_all_data_label,
                        "info": I.sovits.train_diff.cache_all_data_info,
                        "default": lambda: self.get_config_diff()["train"][
                            "cache_all_data"
                        ],
                    },
                    "train.epochs": {
                        "type": "slider",
                        "label": I.sovits.train_diff.epochs_label,
                        "info": I.sovits.train_diff.epochs_info,
                        "min": 50000,
                        "max": 1000000,
                        "default": lambda: self.get_config_diff()["train"]["epochs"],
                        "step": 1,
                    },
                    "use_pretrain": {
                        "type": "dropdown_liked_checkbox",
                        "label": I.sovits.train_diff.use_pretrain_label,
                        "info": I.sovits.train_diff.use_pretrain_info,
                        "default": True,
                    },
                },
                "cluster": {
                    "cluster_or_index": {
                        "type": "dropdown",
                        "label": I.sovits.train_cluster.cluster_or_index_label,
                        "info": I.sovits.train_cluster.cluster_or_index_info,
                        "choices": ["cluster", "index"],
                        "default": "cluster",
                    },
                    "use_gpu": {
                        "type": "dropdown_liked_checkbox",
                        "label": I.sovits.train_cluster.use_gpu_label,
                        "info": I.sovits.train_cluster.use_gpu_info,
                        "default": True,
                    },
                },
            },
        )

        self.svc_model = None
