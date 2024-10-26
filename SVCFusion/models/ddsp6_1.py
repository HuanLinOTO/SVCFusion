import gc
import hashlib
import os
from shutil import rmtree
from SVCFusion.exec import executable

import librosa
import numpy as np
import torch

import soundfile as sf
import yaml

from SVCFusion.config import YAMLReader, applyChanges, system_config
from SVCFusion.dataset_utils import DrawArgs, auto_normalize_dataset
from SVCFusion.i18n import I
from SVCFusion.model_utils import get_pretrain_models_form_item, load_pretrained
from .common import (
    common_infer_form,
    ddsp_based_infer_form,
    common_preprocess_form,
    ddsp_based_preprocess_form,
)
from ddspsvc_6_1.reflow.vocoder import load_model_vocoder
from ddspsvc_6_1.ddsp.vocoder import F0_Extractor, Volume_Extractor, Units_Encoder
from ddspsvc_6_1.ddsp.core import upsample
from ddspsvc_6_1.main_reflow import cross_fade, split
import gradio as gr
from ddspsvc_6_1.draw import main as draw_main

from SVCFusion.exec import exec, start_with_cmd


class DDSP_6_1Model:
    def get_config(*args):
        with YAMLReader("configs/ddsp6.1.yaml") as config:
            return config

    model_name = "DDSP-SVC 6.1"

    infer_form = {}

    train_form = {}

    preprocess_form = {}

    model_types = {
        "cascade": I.ddsp6.model_types.cascade,
    }

    def model_filter(self, filepath: str):
        if filepath.endswith(".pt") and filepath != "model_0.pt":
            return "cascade"

    def pack_model(self, model_dict):
        result = {}
        result["model_dict"] = {}
        result["config_dict"] = {}
        if model_dict.get("cascade", None):
            result["model_dict"]["cascade"] = torch.load(
                model_dict["cascade"], map_location="cpu"
            )
            config_path = os.path.dirname(model_dict["cascade"]) + "/config.yaml"

            with YAMLReader(config_path) as config:
                result["config_dict"]["cascade"] = config
        return result

    def install_model(self, package, model_name):
        model_dict = package["model_dict"]
        config_dict = package["config_dict"]
        base_path = os.path.join("models", model_name)
        os.makedirs(base_path, exist_ok=True)
        if "cascade" in model_dict:
            torch.save(model_dict["cascade"], os.path.join(base_path, "model.pt"))
            with open(os.path.join(base_path, "config.yaml"), "w") as f:
                yaml.dump(config_dict["cascade"], f, default_flow_style=False)

    def unload_model(self):
        # 回收资源
        if self.model is not None:
            del self.model
            self.model = None
        if self.vocoder is not None:
            del self.vocoder
            self.vocoder = None
        if self.units_encoder is not None:
            del self.units_encoder
            self.units_encoder = None
        if self.args is not None:
            del self.args
            self.args = None
        torch.cuda.empty_cache()
        gc.collect()

        self.model = None
        self.vocoder = None
        self.args = None
        self.units_encoder = None
        self.model_device = None

    def load_model(self, model_path_dict) -> None:
        device = model_path_dict["device"]
        path = model_path_dict["cascade"]

        self.unload_model()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # load diffusion model
        self.model, self.vocoder, self.args = load_model_vocoder(path, device=device)
        self.model_device = device

        # load units encoder
        if self.args.data.encoder == "cnhubertsoftfish":
            cnhubertsoft_gate = self.args.data.cnhubertsoft_gate
        else:
            cnhubertsoft_gate = 10
        self.units_encoder = Units_Encoder(
            self.args.data.encoder,
            self.args.data.encoder_ckpt,
            self.args.data.encoder_sample_rate,
            self.args.data.encoder_hop_size,
            cnhubertsoft_gate=cnhubertsoft_gate,
            device=device,
        )

        config_path = os.path.join(os.path.dirname(path), "config.yaml")
        with YAMLReader(config_path) as config:
            self.spks = config.get("spks", [I.default_spk_name])
        return self.spks

    def train(self, params, progress: gr.Progress):
        # print(params)
        working_config_path = "configs/ddsp6.1.yaml"
        del params["_model_name"]

        config = applyChanges(working_config_path, params, no_skip=True)

        if params["use_pretrain"]:
            pretrained_model_config, is_load_success = load_pretrained(
                "ddsp6.1",
                {
                    "vec": config["data"]["encoder"],
                },
                path=params["#pretrain"],
            )
            if pretrained_model_config:
                config = applyChanges(
                    working_config_path, pretrained_model_config, no_skip=True
                )
            if not is_load_success:
                gr.Info(I.train.load_pretrained_failed_tip)
                return

        start_with_cmd(
            f"{executable} -m ddspsvc_6_1.train_reflow -c configs/ddsp6.1.yaml"
        )

    def preprocess(self, params, progress: gr.Progress):
        # 给 data/model_type 文件写入 3
        with open("data/model_type", "w") as f:
            f.write("3")

        # 将 dataset_raw 下面的 文件夹 变成一个数组
        spks = []
        for f in os.listdir("dataset_raw"):
            if f.startswith("."):
                print(f, "has been skiped")
                continue
            if os.path.isdir(os.path.join("dataset_raw", f)):
                spks.append(f)
            # 扫描角色目录，如果发现 .WAV 文件 改成 .wav
            for root, dirs, files in os.walk(f"dataset_raw/{f}"):
                for file in files:
                    if file.endswith(".WAV"):
                        print(f"Renamed {file} to {file.replace('.WAV', '.wav')}")
                        os.rename(
                            os.path.join(root, file),
                            os.path.join(root, file.replace(".WAV", ".wav")),
                        )

        auto_normalize_dataset("data/train/audio", True, progress)

        for i in progress.tqdm(range(1), desc=I.preprocess_draw_desc):
            rmtree("data/val")
            draw_main(DrawArgs())

        with YAMLReader("configs/ddsp6.1.yaml") as config:
            config["data"]["f0_extractor"] = params["f0"]
            config["data"]["encoder"] = params["encoder"]
            config["model"]["n_spk"] = len(spks)
            config["spks"] = spks

        with open("configs/ddsp6.1.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        for i in progress.tqdm(range(1), desc=I.preprocess_desc):
            assert (
                exec(
                    f"{executable} -m ddspsvc_6_1.preprocess -c configs/ddsp6.1.yaml -d {params['device']}"
                )
                == 0
            ), I.preprocess_failed_tip
        return gr.update(value=I.preprocess_finished)

    def infer(
        self,
        params,
        progress: gr.Progress = None,
    ):
        sample_rate = 44100
        num_formant_shift_key = params["num_formant_shift_key"]
        f0_extractor = params["f0"]
        input_file = params["audio"]
        keychange = params["keychange"]
        method = params["method"]
        threhold = params["threshold"]
        infer_step = params["infer_step"]
        t_start = params["t_start"]

        spk = self.spks.index(params["spk"]) + 1

        audio, sample_rate = librosa.load(input_file, sr=sample_rate)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio)

        hop_size = (
            self.args.data.block_size * sample_rate / self.args.data.sampling_rate
        )

        f0 = None

        # get MD5 hash from wav file
        md5_hash = ""
        with open(input_file, "rb") as f:
            data = f.read()
            md5_hash = hashlib.md5(data).hexdigest()
            print("MD5: " + md5_hash)

        cache_dir_path = os.path.join("tmp", "f0_cache")
        cache_file_path = os.path.join(
            cache_dir_path,
            f"{f0_extractor}_{hop_size}_{self.args.data.f0_min}_{self.args.data.f0_max}_{md5_hash}.npy",
        )

        is_cache_available = os.path.exists(cache_file_path)
        if is_cache_available:
            # f0 cache load
            print("Loading pitch curves for input audio from cache directory...")
            f0 = np.load(cache_file_path, allow_pickle=False)
        if type(f0) == type(None):
            # extract f0
            print("Pitch extractor type: " + f0_extractor)

            pitch_extractor = F0_Extractor(
                f0_extractor,
                sample_rate,
                hop_size,
                float(self.args.data.f0_min),
                float(self.args.data.f0_max),
            )
            print("Extracting the pitch curve of the input audio...")
            f0 = pitch_extractor.extract(
                audio, uv_interp=True, device=self.model_device
            )
            if not not md5_hash:
                # f0 cache save
                os.makedirs(cache_dir_path, exist_ok=True)
                np.save(cache_file_path, f0, allow_pickle=False)
        f0 = (
            torch.from_numpy(f0)
            .float()
            .to(self.model_device)
            .unsqueeze(-1)
            .unsqueeze(0)
        )

        # key change
        f0 = f0 * 2 ** (float(keychange) / 12)

        # formant change
        formant_shift_key = (
            torch.from_numpy(np.array([[float(num_formant_shift_key)]]))
            .float()
            .to(self.model_device)
        )

        # extract volume
        print("Extracting the volume envelope of the input audio...")
        volume_extractor = Volume_Extractor(hop_size)
        volume = volume_extractor.extract(audio)
        mask = (volume > 10 ** (float(threhold) / 20)).astype("float")
        mask = np.pad(mask, (4, 4), constant_values=(mask[0], mask[-1]))
        mask = np.array([np.max(mask[n : n + 9]) for n in range(len(mask) - 8)])
        mask = (
            torch.from_numpy(mask)
            .float()
            .to(self.model_device)
            .unsqueeze(-1)
            .unsqueeze(0)
        )
        mask = upsample(mask, self.args.data.block_size).squeeze(-1)

        volume = (
            torch.from_numpy(volume)
            .float()
            .to(self.model_device)
            .unsqueeze(-1)
            .unsqueeze(0)
        )

        # load units encoder
        # if self.args.data.encoder == "cnhubertsoftfish":
        #     cnhubertsoft_gate = self.args.data.cnhubertsoft_gate
        # else:
        #     cnhubertsoft_gate = 10

        # speaker id or mix-speaker dictionary
        spk_id = torch.LongTensor(np.array([[int(spk)]])).to(self.model_device)
        print("Speaker ID: " + str(int(spk_id)))

        # sampling method
        if method == "auto":
            method = self.args.infer.method
        else:
            method = method

        # infer step
        if infer_step == "auto":
            infer_step = self.args.infer.infer_step
        else:
            infer_step = int(infer_step)

        # t_start
        if t_start == "auto":
            if self.args.model.t_start is not None:
                t_start = float(self.args.model.t_start)
            else:
                t_start = 0.0
        else:
            t_start = float(t_start)
            if (
                self.args.model.t_start is not None
                and t_start < self.args.model.t_start
            ):
                t_start = self.args.model.t_start

        if infer_step > 0:
            print("Sampling method: " + method)
            print("infer step: " + str(infer_step))
            print("t_start: " + str(t_start))
        elif infer_step < 0:
            print("infer step cannot be negative!")
            exit(0)

        # forward and save the output
        result = np.zeros(0)
        current_length = 0
        print("Start cutting")
        is_small_audio = audio.size > 10 * sample_rate
        segments = (
            split(audio, sample_rate, hop_size) if is_small_audio else [(0, audio)]
        )
        print("Cut the input audio into " + str(len(segments)) + " slices")
        with torch.no_grad():
            pgs = (
                progress.tqdm(segments, desc=I.ddsp6.infer_tip)
                if type(progress) is not type(None)
                else segments
            )
            for segment in pgs:
                start_frame = segment[0]
                seg_input = (
                    torch.from_numpy(segment[1])
                    .float()
                    .unsqueeze(0)
                    .to(self.model_device)
                )
                seg_units = self.units_encoder.encode(seg_input, sample_rate, hop_size)
                seg_f0 = f0[:, start_frame : start_frame + seg_units.size(1), :]
                seg_volume = volume[:, start_frame : start_frame + seg_units.size(1), :]

                seg_output = self.model(
                    seg_units,
                    seg_f0,
                    seg_volume,
                    spk_id=spk_id,
                    spk_mix_dict=None,
                    aug_shift=formant_shift_key,
                    vocoder=self.vocoder,
                    infer=True,
                    return_wav=True,
                    infer_step=infer_step,
                    method=method,
                    t_start=t_start,
                )
                seg_output *= mask[
                    :,
                    start_frame * self.args.data.block_size : (
                        start_frame + seg_units.size(1)
                    )
                    * self.args.data.block_size,
                ]
                seg_output = seg_output.squeeze().cpu().numpy()

                silent_length = (
                    round(start_frame * self.args.data.block_size) - current_length
                )
                if silent_length >= 0:
                    result = np.append(result, np.zeros(silent_length))
                    result = np.append(result, seg_output)
                else:
                    result = cross_fade(
                        result, seg_output, current_length + silent_length
                    )
                current_length = current_length + silent_length + len(seg_output)
            gc.collect()
            torch.cuda.empty_cache()
            sf.write("tmp/infer_opt/" + params["hash"] + ".wav", result, sample_rate)
            return "tmp/infer_opt/" + params["hash"] + ".wav"

    def __init__(self) -> None:
        self.infer_form.update(common_infer_form)
        self.infer_form.update(ddsp_based_infer_form)

        self.train_form.update(
            {
                "cascade": {
                    **get_pretrain_models_form_item("ddsp6.1"),
                    "device": {
                        "type": "device_chooser",
                        "individual": True,
                    },
                    "train.batch_size": {
                        "type": "slider",
                        "default": lambda: self.get_config()["train"]["batch_size"],
                        "label": I.ddsp6.train.batch_size_label,
                        "info": I.ddsp6.train.batch_size_info,
                        "max": 9999,
                        "min": 1,
                        "step": 1,
                    },
                    "train.num_workers": {
                        "type": "slider",
                        "default": lambda: self.get_config()["train"]["num_workers"],
                        "label": I.ddsp6.train.num_workers_label,
                        "info": I.ddsp6.train.num_workers_info,
                        "max": 9999,
                        "min": 0,
                        "step": 1,
                    },
                    "train.amp_dtype": {
                        "type": "dropdown",
                        "default": lambda: self.get_config()["train"]["amp_dtype"],
                        "label": I.ddsp6.train.amp_dtype_label,
                        "info": I.ddsp6.train.amp_dtype_info,
                        "choices": ["fp16", "bf16", "fp32"],
                    },
                    "train.lr": {
                        "type": "slider",
                        "default": lambda: self.get_config()["train"]["lr"],
                        "step": 0.00001,
                        "min": 0.00001,
                        "max": 0.1,
                        "label": I.ddsp6.train.lr_label,
                        "info": I.ddsp6.train.lr_info,
                    },
                    "train.interval_val": {
                        "type": "slider",
                        "default": lambda: self.get_config()["train"]["interval_val"],
                        "label": I.ddsp6.train.interval_val_label,
                        "info": I.ddsp6.train.interval_val_info,
                        "max": 10000,
                        "min": 1,
                        "step": 1,
                    },
                    "train.interval_log": {
                        "type": "slider",
                        "default": lambda: self.get_config()["train"]["interval_log"],
                        "label": I.ddsp6.train.interval_log_label,
                        "info": I.ddsp6.train.interval_log_info,
                        "max": 10000,
                        "min": 1,
                        "step": 1,
                    },
                    "train.interval_force_save": {
                        "type": "slider",
                        "label": I.ddsp6.train.interval_force_save_label,
                        "info": I.ddsp6.train.interval_force_save_info,
                        "min": 0,
                        "max": 100000,
                        "default": lambda: self.get_config()["train"][
                            "interval_force_save"
                        ],
                        "step": 1000,
                    },
                    "train.gamma": {
                        "type": "slider",
                        "label": I.ddsp6.train.gamma_label,
                        "info": I.ddsp6.train.gamma_info,
                        "min": 0,
                        "max": 1,
                        "default": lambda: self.get_config()["train"]["gamma"],
                        "step": 0.1,
                    },
                    "train.cache_device": {
                        "type": "dropdown",
                        "label": I.ddsp6.train.cache_device_label,
                        "info": I.ddsp6.train.cache_device_info,
                        "choices": ["cuda", "cpu"],
                        "default": lambda: self.get_config()["train"]["cache_device"],
                    },
                    "train.cache_all_data": {
                        "type": "dropdown_liked_checkbox",
                        "label": I.ddsp6.train.cache_all_data_label,
                        "info": I.ddsp6.train.cache_all_data_info,
                        "default": lambda: self.get_config()["train"]["cache_all_data"],
                    },
                    "train.epochs": {
                        "type": "slider",
                        "label": I.ddsp6.train.epochs_label,
                        "info": I.ddsp6.train.epochs_info,
                        "min": 50000,
                        "max": 1000000,
                        "default": lambda: self.get_config()["train"]["epochs"],
                        "step": 1,
                    },
                    "use_pretrain": {
                        "type": "dropdown_liked_checkbox",
                        "label": I.ddsp6.train.use_pretrain_label,
                        "info": I.ddsp6.train.use_pretrain_info,
                        "default": True,
                    },
                }
            }
        )

        self.preprocess_form.update(common_preprocess_form)
        self.preprocess_form.update(ddsp_based_preprocess_form)

        self.model = None
        self.vocoder = None
        self.args = None
        self.units_encoder = None
        self.model_device = None
