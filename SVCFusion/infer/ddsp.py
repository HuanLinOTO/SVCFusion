import gc
import os
import torch
import numpy as np
import gradio as gr

from ddspsvc.reflow.vocoder import load_model_vocoder
from ddspsvc.ddsp.vocoder import F0_Extractor, Volume_Extractor, Units_Encoder
from ddspsvc.ddsp.core import upsample
from ddspsvc.main_diff import cross_fade, split


class DDSPInfer:
    def __init__(self) -> None:
        self.model = None
        self.vocoder = None
        self.args = None
        self.units_encoder = None
        self.model_device = None

    def load_model(self, device, path):
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

    def infer_core(
        self,
        audio,
        sample_rate,
        cache_md5,
        keychange=0,
        method="rk4",
        infer_step=20,
        t_start=0.7,
        f0_extractor="fcpe",
        threhold=-60,
        num_formant_shift_key=0,
        spk=0,
        progress: gr.Progress = None,
    ):
        hop_size = (
            self.args.data.block_size * sample_rate / self.args.data.sampling_rate
        )

        f0 = None

        if not not cache_md5:
            cache_dir_path = os.path.join(os.path.dirname("tmp"), "f0_cache")
            cache_file_path = os.path.join(
                cache_dir_path,
                f"{f0_extractor}_{hop_size}_{self.args.data.f0_min}_{self.args.data.f0_max}_{cache_md5}.npy",
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
            if not not cache_md5:
                # f0 cache save
                os.makedirs(cache_dir_path, exist_ok=True)
                np.save(cache_file_path, f0, allow_pickle=False)
        f0_original = f0
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
        if self.args.data.encoder == "cnhubertsoftfish":
            cnhubertsoft_gate = self.args.data.cnhubertsoft_gate
        else:
            cnhubertsoft_gate = 10

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
                progress.tqdm(segments, desc="推理 DDSP 模型")
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
            # sf.write(output_path, result, args.data.sampling_rate)
            gc.collect()
            torch.cuda.empty_cache()
            return result, self.args.data.sampling_rate, f0_original
