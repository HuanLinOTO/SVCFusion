import gc
import os
import gradio as gr
import numpy as np
import torch
from tqdm import tqdm

from ReFlowVaeSVC.main import cross_fade, upsample, split
from ReFlowVaeSVC.reflow.vocoder import load_model_vocoder
from ReFlowVaeSVC.reflow.extractors import F0_Extractor, Volume_Extractor, Units_Encoder


class ReflowVAESVCInfer:
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

        self.model, self.vocoder, self.args = load_model_vocoder(path, device=device)

        self.model_device = device

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
        source_spk_id=None,
        progress: gr.Progress = None,
    ):
        hop_size = (
            self.args.data.block_size * sample_rate / self.args.data.sampling_rate
        )

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
        else:
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

            # f0 cache save
            os.makedirs(cache_dir_path, exist_ok=True)
            np.save(cache_file_path, f0, allow_pickle=False)

        # key change
        input_f0 = (
            torch.from_numpy(f0)
            .float()
            .to(self.model_device)
            .unsqueeze(-1)
            .unsqueeze(0)
        )
        output_f0 = input_f0 * 2 ** (float(keychange) / 12)

        # formant change
        formant_shift_key = (
            torch.from_numpy(np.array([[float(num_formant_shift_key)]]))
            .float()
            .to(self.model_device)
        )

        # source speaker id
        if source_spk_id == None:
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
                    device=self.model_device,
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

        else:
            source_spk_id = torch.LongTensor(np.array([[int(source_spk_id)]])).to(
                self.model_device
            )
            print("Using VAE mode...")
            print("Source Speaker ID: " + str(int(source_spk_id)))

        # targer speaker id or mix-speaker dictionary
        spk_mix_dict = None
        target_spk_id = torch.LongTensor(np.array([[int(spk)]])).to(self.model_device)

        print("Target Speaker ID: " + str(int(spk)))

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

        if infer_step < 0:
            print("infer step cannot be negative!")
            exit(0)

        # forward and save the output
        result = np.zeros(0)
        current_length = 0
        segments = split(audio, sample_rate, hop_size)
        print("Cut the input audio into " + str(len(segments)) + " slices")
        with torch.no_grad():
            for segment in tqdm(segments):
                start_frame = segment[0]
                seg_input = (
                    torch.from_numpy(segment[1])
                    .float()
                    .unsqueeze(0)
                    .to(self.model_device)
                )
                if source_spk_id == None:
                    seg_units = self.units_encoder.encode(
                        seg_input, sample_rate, hop_size
                    )
                    seg_f0 = output_f0[
                        :, start_frame : start_frame + seg_units.size(1), :
                    ]
                    seg_volume = volume[
                        :, start_frame : start_frame + seg_units.size(1), :
                    ]

                    seg_output = self.model(
                        seg_units,
                        seg_f0,
                        seg_volume,
                        spk_id=target_spk_id,
                        spk_mix_dict=spk_mix_dict,
                        aug_shift=formant_shift_key,
                        vocoder=self.vocoder,
                        infer=True,
                        return_wav=True,
                        infer_step=infer_step,
                        method=method,
                    )
                    seg_output *= mask[
                        :,
                        start_frame * self.args.data.block_size : (
                            start_frame + seg_units.size(1)
                        )
                        * self.args.data.block_size,
                    ]
                else:
                    seg_input_mel = self.vocoder.extract(seg_input, sample_rate)
                    seg_input_mel = torch.cat(
                        (seg_input_mel, seg_input_mel[:, -1:, :]), 1
                    )
                    seg_input_f0 = input_f0[
                        :, start_frame : start_frame + seg_input_mel.size(1), :
                    ]
                    seg_output_f0 = output_f0[
                        :, start_frame : start_frame + seg_input_mel.size(1), :
                    ]

                    seg_output_mel = self.model.vae_infer(
                        seg_input_mel,
                        seg_input_f0,
                        source_spk_id,
                        seg_output_f0,
                        target_spk_id,
                        spk_mix_dict,
                        formant_shift_key,
                        infer_step,
                        method,
                    )
                    seg_output = self.vocoder.infer(seg_output_mel, seg_output_f0)

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
        return result, sample_rate, input_f0
