import FreeSimpleGUI as sg
import sounddevice as sd
import torch
import librosa
import pickle
import numpy as np
from torch.nn import functional as F
from torchaudio.transforms import Resample
from .ddsp.vocoder import F0_Extractor, Volume_Extractor, Units_Encoder
from .ddsp.core import upsample
import time
from .gui_diff_locale import I18nAuto
from .reflow.vocoder import load_model_vocoder

flag_vc = False


def phase_vocoder(a, b, fade_out, fade_in):
    window = torch.sqrt(fade_out * fade_in)
    fa = torch.fft.rfft(a * window)
    fb = torch.fft.rfft(b * window)
    absab = torch.abs(fa) + torch.abs(fb)
    n = a.shape[0]
    if n % 2 == 0:
        absab[1:-1] *= 2
    else:
        absab[1:] *= 2
    phia = torch.angle(fa)
    phib = torch.angle(fb)
    deltaphase = phib - phia
    deltaphase = deltaphase - 2 * np.pi * torch.floor(deltaphase / 2 / np.pi + 0.5)
    w = 2 * np.pi * torch.arange(n // 2 + 1).to(a) + deltaphase
    t = torch.arange(n).unsqueeze(-1).to(a) / n
    result = (
        a * (fade_out**2)
        + b * (fade_in**2)
        + torch.sum(absab * torch.cos(w * t + phia), -1) * window / n
    )
    return result


class SvcDDSP:
    def __init__(self) -> None:
        self.reflow_model = None
        self.args = None
        self.units_encoder = None
        self.encoder_type = None
        self.encoder_ckpt = None
        self.formant_shift_key = None

    def update_model(self, reflow_model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # load reflow model
        self.reflow_model, self.vocoder, self.args = load_model_vocoder(
            reflow_model_path, device=self.device
        )

        # load units encoder
        if (
            self.units_encoder is None
            or self.args.data.encoder != self.encoder_type
            or self.args.data.encoder_ckpt != self.encoder_ckpt
        ):
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
                device=self.device,
            )
            self.encoder_type = self.args.data.encoder
            self.encoder_ckpt = self.args.data.encoder_ckpt

    def infer(
        self,
        audio,
        sample_rate,
        spk_id=1,
        threhold=-45,
        pitch_adjust=0,
        formant_shift_key=0,
        use_spk_mix=False,
        spk_mix_dict=None,
        enhancer_adaptive_key="auto",
        pitch_extractor_type="crepe",
        f0_min=50,
        f0_max=1100,
        safe_prefix_pad_length=0,
        sampling_method="euler",
        infer_step=None,
        t_start=0.1,
        audio_alignment=False,
    ):
        print("Infering...")
        # load input
        # audio, sample_rate = librosa.load(input_wav, sr=None, mono=True)
        hop_size = (
            self.args.data.block_size * sample_rate / self.args.data.sampling_rate
        )
        if audio_alignment:
            audio_length = len(audio)
        # safe front silence
        if safe_prefix_pad_length > 0.03:
            silence_front = safe_prefix_pad_length - 0.03
        else:
            silence_front = 0
        audio_t = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)

        # extract f0
        pitch_extractor = F0_Extractor(
            pitch_extractor_type, sample_rate, hop_size, float(f0_min), float(f0_max)
        )
        f0 = pitch_extractor.extract(
            audio, uv_interp=True, device=self.device, silence_front=silence_front
        )
        f0 = torch.from_numpy(f0).float().to(self.device).unsqueeze(-1).unsqueeze(0)
        f0 = f0 * 2 ** (float(pitch_adjust) / 12)  # 变调

        # formant change
        formant_shift_key = (
            torch.from_numpy(np.array([[float(formant_shift_key)]]))
            .float()
            .to(self.device)
        )

        # extract volume
        volume_extractor = Volume_Extractor(hop_size)
        volume = volume_extractor.extract(audio)
        mask = (volume > 10 ** (float(threhold) / 20)).astype("float")
        mask = np.pad(mask, (4, 4), constant_values=(mask[0], mask[-1]))
        mask = np.array([np.max(mask[n : n + 9]) for n in range(len(mask) - 8)])
        mask = torch.from_numpy(mask).float().to(self.device).unsqueeze(-1).unsqueeze(0)
        mask = upsample(mask, self.args.data.block_size).squeeze(-1)
        volume = (
            torch.from_numpy(volume).float().to(self.device).unsqueeze(-1).unsqueeze(0)
        )

        # extract units
        units = self.units_encoder.encode(audio_t, sample_rate, hop_size)

        # spk_id or spk_mix_dict
        spk_id = torch.LongTensor(np.array([[spk_id]])).to(self.device)
        dictionary = None
        if use_spk_mix:
            dictionary = spk_mix_dict

        # forward and return the output
        with torch.no_grad():
            output = self.reflow_model(
                units,
                f0,
                volume,
                spk_id=spk_id,
                spk_mix_dict=spk_mix_dict,
                aug_shift=formant_shift_key,
                vocoder=self.vocoder,
                infer=True,
                return_wav=True,
                infer_step=infer_step,
                method=sampling_method,
                t_start=t_start,
            )
            output *= mask
            output = output.squeeze()
            if audio_alignment:
                output[:audio_length]
            return output, self.args.data.sampling_rate


class Config:
    def __init__(self) -> None:
        self.samplerate = 44100  # Hz
        self.block_time = 0.5  # s
        self.f_pitch_change: float = 0.0  # float(request_form.get("fPitchChange", 0))
        self.formant_shift_key = 0
        self.spk_id = 1  # 默认说话人。
        self.spk_mix_dict = (
            None  # {1:0.5, 2:0.5} 表示1号说话人和2号说话人的音色按照0.5:0.5的比例混合
        )
        self.use_phase_vocoder = False
        self.checkpoint_path = ""
        self.threhold = -45
        self.crossfade_time = 0.04
        self.extra_time = 2
        self.select_pitch_extractor = "harvest"  # F0预测器["parselmouth", "dio", "harvest", "crepe", "rmvpe", "fcpe"]
        self.use_spk_mix = False
        self.sounddevices = ["", ""]
        self.reflow_model = ""
        self.infer_step = 50
        self.sampling_method = "euler"

    def save(self, path):
        with open(path + "\\config.pkl", "wb") as f:
            pickle.dump(vars(self), f)

    def load(self, path) -> bool:
        try:
            with open(path + "\\config.pkl", "rb") as f:
                self.update(pickle.load(f))
            return True
        except:
            print("config.pkl does not exist")
            return False

    def update(self, data_dict):
        for key, value in data_dict.items():
            setattr(self, key, value)


class GUI:
    def __init__(self) -> None:
        self.config = Config()
        self.block_frame = 0
        self.crossfade_frame = 0
        self.sola_search_frame = 0
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.svc_model: SvcDDSP = SvcDDSP()
        self.fade_in_window: torch.Tensor = None
        self.fade_out_window: torch.Tensor = None
        self.input_wav: np.ndarray = None
        self.sola_buffer: torch.Tensor = None
        self.f0_mode_list = [
            "parselmouth",
            "dio",
            "harvest",
            "crepe",
            "rmvpe",
            "fcpe",
        ]  # F0预测器
        self.sampling_method_list = ["euler", "rk4"]  # 采样方法
        self.f_safe_prefix_pad_length: float = 0.0
        self.resample_kernel = {}
        self.stream = None
        self.input_devices = None
        self.output_devices = None
        self.input_devices_indices = None
        self.output_devices_indices = None
        self.update_devices()
        self.default_input_device = self.input_devices[
            self.input_devices_indices.index(sd.default.device[0])
        ]
        self.default_output_device = self.output_devices[
            self.output_devices_indices.index(sd.default.device[1])
        ]
        self.launcher()  # start

    def launcher(self):
        """窗口加载"""
        sg.theme("DarkBlue12")  # 设置主题
        # 界面布局
        layout = [
            [
                sg.Frame(
                    layout=[
                        [
                            sg.Text(i18n("输入设备")),
                            sg.Combo(
                                self.input_devices,
                                key="sg_input_device",
                                default_value=self.default_input_device,
                                enable_events=True,
                            ),
                        ],
                        [
                            sg.Text(i18n("输出设备")),
                            sg.Combo(
                                self.output_devices,
                                key="sg_output_device",
                                default_value=self.default_output_device,
                                enable_events=True,
                            ),
                        ],
                    ],
                    title=i18n("音频设备"),
                )
            ],
            [
                sg.Frame(
                    layout=[
                        [
                            sg.Text(
                                i18n("模型文件：.pt格式(自动识别同目录下config.yaml)")
                            )
                        ],
                        [
                            sg.Input(
                                key="reflow_model",
                                default_text="exp\\diffusion-new-demo\\model_200000.pt",
                            ),
                            sg.FileBrowse(i18n("选择模型文件"), key="choose_model"),
                        ],
                        [
                            sg.Text(i18n("采样步数")),
                            sg.Input(key="infer_step", default_text="50", size=18),
                        ],
                        [
                            sg.Text(i18n("时间起点")),
                            sg.Input(key="t_start", default_text="0.5", size=18),
                        ],
                        [
                            sg.Text(i18n("采样算法")),
                            sg.Combo(
                                values=self.sampling_method_list,
                                key="sampling_method",
                                default_value=self.sampling_method_list[0],
                                enable_events=True,
                            ),
                        ],
                    ],
                    title=i18n("Reflow设置"),
                ),
            ],
            [
                sg.Frame(
                    layout=[
                        [
                            sg.Text(i18n("选择配置文件所在目录")),
                            sg.Input(key="config_file_dir", default_text="exp"),
                            sg.FolderBrowse(i18n("打开文件夹"), key="choose_config"),
                        ],
                        [
                            sg.Button(i18n("读取配置文件"), key="load_config"),
                            sg.Button(i18n("保存配置文件"), key="save_config"),
                        ],
                    ],
                    title=i18n("快速配置文件"),
                )
            ],
            [
                sg.Frame(
                    layout=[
                        [
                            sg.Text(i18n("说话人id")),
                            sg.Input(key="spk_id", default_text="1", size=8),
                        ],
                        [
                            sg.Text(i18n("响应阈值")),
                            sg.Slider(
                                range=(-60, 0),
                                orientation="h",
                                key="threhold",
                                resolution=1,
                                default_value=-45,
                                enable_events=True,
                            ),
                        ],
                        [
                            sg.Text(i18n("变调")),
                            sg.Slider(
                                range=(-24, 24),
                                orientation="h",
                                key="pitch",
                                resolution=1,
                                default_value=0,
                                enable_events=True,
                            ),
                        ],
                        [
                            sg.Text(i18n("共振峰偏移")),
                            sg.Slider(
                                range=(-2, 2),
                                orientation="h",
                                key="formant_shift_key",
                                resolution=0.05,
                                default_value=0,
                                enable_events=True,
                            ),
                        ],
                        [
                            sg.Text(i18n("采样率")),
                            sg.Input(key="samplerate", default_text="44100", size=8),
                        ],
                        [
                            sg.Checkbox(
                                text=i18n("启用捏音色功能"),
                                default=False,
                                key="spk_mix",
                                enable_events=True,
                            ),
                            sg.Button(i18n("设置混合音色"), key="set_spk_mix"),
                        ],
                    ],
                    title=i18n("普通设置"),
                ),
                sg.Frame(
                    layout=[
                        [
                            sg.Text(i18n("音频切分大小")),
                            sg.Slider(
                                range=(0.05, 3.0),
                                orientation="h",
                                key="block",
                                resolution=0.01,
                                default_value=0.5,
                                enable_events=True,
                            ),
                        ],
                        [
                            sg.Text(i18n("交叉淡化时长")),
                            sg.Slider(
                                range=(0.01, 0.15),
                                orientation="h",
                                key="crossfade",
                                resolution=0.01,
                                default_value=0.04,
                                enable_events=True,
                            ),
                        ],
                        [
                            sg.Text(i18n("额外推理时长")),
                            sg.Slider(
                                range=(0.05, 5),
                                orientation="h",
                                key="extra",
                                resolution=0.01,
                                default_value=2.0,
                                enable_events=True,
                            ),
                        ],
                        [
                            sg.Text(i18n("f0预测模式")),
                            sg.Combo(
                                values=self.f0_mode_list,
                                key="f0_mode",
                                default_value=self.f0_mode_list[-1],
                                enable_events=True,
                            ),
                        ],
                        [
                            sg.Checkbox(
                                text=i18n("启用相位声码器"),
                                default=False,
                                key="use_phase_vocoder",
                                enable_events=True,
                            )
                        ],
                    ],
                    title=i18n("性能设置"),
                ),
            ],
            [
                sg.Button(i18n("开始音频转换"), key="start_vc"),
                sg.Button(i18n("停止音频转换"), key="stop_vc"),
                sg.Text(i18n("推理所用时间(ms):")),
                sg.Text("0", key="infer_time"),
            ],
        ]

        # 创造窗口
        self.window = sg.Window("DDSP - GUI", layout, finalize=True)
        self.window["spk_id"].bind("<Return>", "")
        self.window["samplerate"].bind("<Return>", "")
        self.window["infer_step"].bind("<Return>", "")
        self.window["t_start"].bind("<Return>", "")
        self.event_handler()

    def event_handler(self):
        """事件处理"""
        global flag_vc
        while True:  # 事件处理循环
            event, values = self.window.read()
            print("event: " + event)
            if event == sg.WINDOW_CLOSED:  # 如果用户关闭窗口
                flag_vc = False
                exit()
            elif event == "start_vc" and not flag_vc:
                # set values 和界面布局layout顺序一一对应
                self.set_values(values)
                print("block_time:" + str(self.config.block_time))
                print("crossfade_time:" + str(self.config.crossfade_time))
                print("extra_time:" + str(self.config.extra_time))
                print("samplerate:" + str(self.config.samplerate))
                print("prefix_pad_length:" + str(self.f_safe_prefix_pad_length))
                print("mix_mode:" + str(self.config.spk_mix_dict))
                print("using_cuda:" + str(torch.cuda.is_available()))
                self.start_vc()
            elif event == "sampling_method":
                self.config.sampling_method = values["sampling_method"]
            elif event == "spk_id":
                self.config.spk_id = int(values["spk_id"])
            elif event == "threhold":
                self.config.threhold = values["threhold"]
            elif event == "pitch":
                self.config.f_pitch_change = values["pitch"]
            elif event == "spk_mix":
                self.config.use_spk_mix = values["spk_mix"]
            elif event == "set_spk_mix":
                spk_mix = sg.popup_get_text(
                    message="示例：1:0.3,2:0.5,3:0.2", title="设置混合音色，支持多人"
                )
                if spk_mix != None:
                    self.config.spk_mix_dict = eval(
                        "{" + spk_mix.replace("，", ",").replace("：", ":") + "}"
                    )
            elif event == "f0_mode":
                self.config.select_pitch_extractor = values["f0_mode"]
            elif event == "use_phase_vocoder":
                self.config.use_phase_vocoder = values["use_phase_vocoder"]
            elif event == "load_config" and not flag_vc:
                if self.config.load(values["config_file_dir"]):
                    self.update_values()
            elif event == "save_config" and not flag_vc:
                self.set_values(values)
                self.config.save(values["config_file_dir"])
            elif event != "start_vc" and flag_vc:
                self.stop_stream()

    def set_values(self, values):
        self.set_devices(values["sg_input_device"], values["sg_output_device"])
        self.config.sounddevices = [
            values["sg_input_device"],
            values["sg_output_device"],
        ]
        self.config.spk_id = int(values["spk_id"])
        self.config.threhold = values["threhold"]
        self.config.f_pitch_change = values["pitch"]
        self.config.formant_shift_key = values["formant_shift_key"]
        self.config.samplerate = int(values["samplerate"])
        self.config.block_time = float(values["block"])
        self.config.crossfade_time = float(values["crossfade"])
        self.config.extra_time = float(values["extra"])
        self.config.select_pitch_extractor = values["f0_mode"]
        self.config.use_phase_vocoder = values["use_phase_vocoder"]
        self.config.use_spk_mix = values["spk_mix"]
        self.config.sampling_method = values["sampling_method"]
        self.config.reflow_model = values["reflow_model"]
        self.config.t_start = values["t_start"]
        self.config.infer_step = int(values["infer_step"])
        self.block_frame = int(self.config.block_time * self.config.samplerate)
        self.crossfade_frame = int(self.config.crossfade_time * self.config.samplerate)
        self.sola_search_frame = int(0.01 * self.config.samplerate)
        self.last_delay_frame = int(0.02 * self.config.samplerate)
        self.extra_frame = int(self.config.extra_time * self.config.samplerate)
        self.input_frame = max(
            self.block_frame
            + self.crossfade_frame
            + self.sola_search_frame
            + 2 * self.last_delay_frame,
            self.block_frame + self.extra_frame,
        )
        self.f_safe_prefix_pad_length = (
            self.config.extra_time - self.config.crossfade_time - 0.01 - 0.02
        )

    def update_values(self):
        self.window["sg_input_device"].update(self.config.sounddevices[0])
        self.window["sg_output_device"].update(self.config.sounddevices[1])
        self.window["spk_id"].update(self.config.spk_id)
        self.window["threhold"].update(self.config.threhold)
        self.window["pitch"].update(self.config.f_pitch_change)
        self.window["formant_shift_key"].update(self.config.formant_shift_key)
        self.window["samplerate"].update(self.config.samplerate)
        self.window["spk_mix"].update(self.config.use_spk_mix)
        self.window["block"].update(self.config.block_time)
        self.window["crossfade"].update(self.config.crossfade_time)
        self.window["extra"].update(self.config.extra_time)
        self.window["f0_mode"].update(self.config.select_pitch_extractor)
        self.window["sampling_method"].update(self.config.sampling_method)
        self.window["reflow_model"].update(self.config.reflow_model)
        self.window["t_start"].update(self.config.t_start)
        self.window["infer_step"].update(self.config.infer_step)

    def start_vc(self):
        """开始音频转换"""
        torch.cuda.empty_cache()
        self.input_wav = np.zeros(self.input_frame, dtype="float32")
        self.sola_buffer = torch.zeros(self.crossfade_frame, device=self.device)
        self.fade_in_window = (
            torch.sin(
                np.pi
                * torch.arange(0, 1, 1 / self.crossfade_frame, device=self.device)
                / 2
            )
            ** 2
        )
        self.fade_out_window = 1 - self.fade_in_window
        self.svc_model.update_model(self.config.reflow_model)
        self.start_stream()

    def start_stream(self):
        global flag_vc
        if not flag_vc:
            flag_vc = True
            self.stream = sd.Stream(
                channels=2,
                callback=self.audio_callback,
                blocksize=self.block_frame,
                samplerate=self.config.samplerate,
                dtype="float32",
            )
            self.stream.start()

    def stop_stream(self):
        global flag_vc
        if flag_vc:
            flag_vc = False
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
                self.stream = None

    def audio_callback(
        self, indata: np.ndarray, outdata: np.ndarray, frames, times, status
    ):
        """
        音频处理
        """
        start_time = time.perf_counter()
        print("\nStarting callback")
        self.input_wav[:] = np.roll(self.input_wav, -self.block_frame)
        self.input_wav[-self.block_frame :] = librosa.to_mono(indata.T)

        # infer
        _audio, _model_sr = self.svc_model.infer(
            self.input_wav,
            self.config.samplerate,
            spk_id=self.config.spk_id,
            threhold=self.config.threhold,
            pitch_adjust=self.config.f_pitch_change,
            formant_shift_key=self.config.formant_shift_key,
            use_spk_mix=self.config.use_spk_mix,
            spk_mix_dict=self.config.spk_mix_dict,
            pitch_extractor_type=self.config.select_pitch_extractor,
            safe_prefix_pad_length=self.f_safe_prefix_pad_length,
            sampling_method=self.config.sampling_method,
            infer_step=self.config.infer_step,
        )

        # debug sola
        """
        _audio, _model_sr = self.input_wav, self.config.samplerate
        rs = int(np.random.uniform(-200,200))
        print('debug_random_shift: ' + str(rs))
        _audio = np.roll(_audio, rs)
        _audio = torch.from_numpy(_audio).to(self.device)
        """

        if _model_sr != self.config.samplerate:
            key_str = str(_model_sr) + "_" + str(self.config.samplerate)
            if key_str not in self.resample_kernel:
                self.resample_kernel[key_str] = Resample(
                    _model_sr, self.config.samplerate, lowpass_filter_width=128
                ).to(self.device)
            _audio = self.resample_kernel[key_str](_audio)
        temp_wav = _audio[
            -self.block_frame
            - self.crossfade_frame
            - self.sola_search_frame
            - self.last_delay_frame : -self.last_delay_frame
        ]

        # sola shift
        conv_input = temp_wav[
            None, None, : self.crossfade_frame + self.sola_search_frame
        ]
        cor_nom = F.conv1d(conv_input, self.sola_buffer[None, None, :])
        cor_den = torch.sqrt(
            F.conv1d(
                conv_input**2,
                torch.ones(1, 1, self.crossfade_frame, device=self.device),
            )
            + 1e-8
        )
        sola_shift = torch.argmax(cor_nom[0, 0] / cor_den[0, 0])
        temp_wav = temp_wav[
            sola_shift : sola_shift + self.block_frame + self.crossfade_frame
        ]
        print("sola_shift: " + str(int(sola_shift)))

        # phase vocoder
        if self.config.use_phase_vocoder:
            temp_wav[: self.crossfade_frame] = phase_vocoder(
                self.sola_buffer,
                temp_wav[: self.crossfade_frame],
                self.fade_out_window,
                self.fade_in_window,
            )
        else:
            temp_wav[: self.crossfade_frame] *= self.fade_in_window
            temp_wav[: self.crossfade_frame] += self.sola_buffer * self.fade_out_window

        self.sola_buffer = temp_wav[-self.crossfade_frame :]

        outdata[:] = temp_wav[: -self.crossfade_frame, None].repeat(1, 2).cpu().numpy()
        end_time = time.perf_counter()
        print("infer_time: " + str(end_time - start_time))
        if flag_vc:
            self.window["infer_time"].update(int((end_time - start_time) * 1000))

    def update_devices(self):
        """获取设备列表"""
        sd._terminate()
        sd._initialize()
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
        for hostapi in hostapis:
            for device_idx in hostapi["devices"]:
                devices[device_idx]["hostapi_name"] = hostapi["name"]
        self.input_devices = [
            f"{d['name']} ({d['hostapi_name']})"
            for d in devices
            if d["max_input_channels"] > 0
        ]
        self.output_devices = [
            f"{d['name']} ({d['hostapi_name']})"
            for d in devices
            if d["max_output_channels"] > 0
        ]
        self.input_devices_indices = [
            d["index"] for d in devices if d["max_input_channels"] > 0
        ]
        self.output_devices_indices = [
            d["index"] for d in devices if d["max_output_channels"] > 0
        ]

    def set_devices(self, input_device, output_device):
        """设置输出设备"""
        sd.default.device[0] = self.input_devices_indices[
            self.input_devices.index(input_device)
        ]
        sd.default.device[1] = self.output_devices_indices[
            self.output_devices.index(output_device)
        ]
        print("input device:" + str(sd.default.device[0]) + ":" + str(input_device))
        print("output device:" + str(sd.default.device[1]) + ":" + str(output_device))


if __name__ == "__main__":
    i18n = I18nAuto()
    gui = GUI()
