import hashlib
import os
import time
import traceback
import gradio as gr
import torchaudio

from Music_Source_Separation_Training import inference as msst_inference

import logging
import ffmpeg
import torch
from vr import AudioPre, AudioPreDeEcho


logger = logging.getLogger(__name__)

os.environ["PATH"] += os.pathsep + os.getcwd()

device = "cuda"
is_half = True


def preprocess_path(path):
    return path.strip(" ").strip('"').strip("\n").strip('"').strip(" ")


def uvr(
    model_name, inp_root, save_root_vocal, paths, save_root_ins, agg, format0, output
):
    try:
        inp_root = preprocess_path(inp_root)
        save_root_vocal = preprocess_path(save_root_vocal)
        save_root_ins = preprocess_path(save_root_ins)
        is_hp3 = "HP3" in model_name

        func = AudioPre if "DeEcho" not in model_name else AudioPreDeEcho
        pre_fun = func(
            agg=int(agg),
            model_path=os.path.join("other_weights", model_name + ".pth"),
            device=device,
            is_half=is_half,
        )
        for path in paths:
            inp_path = os.path.join(inp_root, path)
            print(inp_path)
            if os.path.isfile(inp_path) is False:
                print(f"File {inp_path} not found")
                continue
            # need_reformat = 1
            done = 0
            # try:
            #     info = ffmpeg.probe(inp_path, cmd="ffprobe")
            #     if (
            #         info["streams"][0]["channels"] == 2
            #         and info["streams"][0]["sample_rate"] == "44100"
            #     ):
            #         need_reformat = 0
            #         pre_fun._path_audio_(
            #             inp_path,
            #             save_root_ins,
            #             save_root_vocal,
            #             format0,
            #             is_hp3,
            #             output,
            #         )
            #         done = 1
            # except Exception:
            #     need_reformat = 1
            #     traceback.print_exc()
            # if need_reformat == 1:
            #     tmp_path = "%s/%s.reformatted.wav" % (
            #         os.path.join("tmp"),
            #         os.path.basename(inp_path),
            #     )
            #     os.system(
            #         'ffmpeg -i "%s" -vn -acodec pcm_s16le -ac 2 -ar 44100 "%s" -y'
            #         % (inp_path, tmp_path)
            #     )
            #     inp_path = tmp_path
            try:
                if done == 0:
                    pre_fun._path_audio_(
                        inp_path,
                        save_root_ins,
                        save_root_vocal,
                        format0,
                        is_hp3,
                        output,
                    )
            except Exception:
                traceback.print_exc()
    except Exception:
        print(traceback.format_exc())
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


bsroformer_model = None
bsroformer_device = None


class MSSTArgs:
    input_folder = None
    store_dir = None
    model_type = None

    def __init__(self, input_folder, store_dir, model_type) -> None:
        self.input_folder = input_folder
        self.store_dir = store_dir
        self.model_type = model_type


def run_bsroformer(inp_path, inp_hash, progress=gr.Progress()):
    global bsroformer_model, bsroformer_config
    vocal_path = f"./tmp/bsroformer_opt/{inp_hash}_Vocals.wav"
    inst_path = f"./tmp/bsroformer_opt/{inp_hash}_Instrument.wav"
    if os.path.exists(vocal_path) and os.path.exists(inst_path):
        return vocal_path, inst_path

    args = MSSTArgs(
        input_folder=inp_path,
        store_dir=f"./tmp/bsroformer_opt/{inp_hash}",
        model_type="bs_roformer",
    )
    if not bsroformer_model:
        bsroformer_model, bsroformer_config = msst_inference.get_model_from_config(
            "bs_roformer",
            "Music_Source_Separation_Training/configs/model_bs_roformer_ep_368_sdr_12.9628.yaml",
        )
        model_path = "other_weights/model_bs_roformer_ep_368_sdr_12.9628.ckpt"
        print("Start from checkpoint: {}".format(model_path))
        state_dict = torch.load(model_path)
        if args.model_type == "htdemucs":
            # Fix for htdemucs pround etrained models
            if "state" in state_dict:
                state_dict = state_dict["state"]
        bsroformer_model.load_state_dict(state_dict)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        bsroformer_model = bsroformer_model.to(device)
    else:
        device = "cpu"
        print("CUDA is not avilable. Run inference on CPU. It will be very slow...")
        bsroformer_model = bsroformer_model.to(device)
    msst_inference.run_folder(
        bsroformer_model,
        args,
        bsroformer_config,
        device,
        verbose=False,
        progress=progress,
    )
    # subprocess.run(
    #     [
    #         '.conda\\python',
    #         'inference.py',
    #         '--model_type',
    #         'bs_roformer',
    #         '--config_path',
    #         'configs/model_bs_roformer_ep_368_sdr_12.9628.yaml',
    #         '--start_check_point',
    #         '../other_weights/model_bs_roformer_ep_368_sdr_12.9628.ckpt',
    #         '--input_folder',
    #         inp_path,
    #         '--store_dir',
    #         f'../tmp/bsroformer_opt/{inp_hash}',
    #     ],
    #     cwd='./Music-Source-Separation-Training',
    # )
    return vocal_path, inst_path


def getVocalAndInstrument(inp_path, progress=gr.Progress()):
    # inp_path = 'C:\\Users\\Administrator\\AppData\\Local\\Temp\\gradio\\8318a6ff3ae407ac036a5d3c40b910e341ee768b\\AI Kikyou  名もない花.mp3'
    if not inp_path:
        raise gr.Error("Please upload an audio file")
    inp_hash = hashlib.md5(inp_path.encode()).hexdigest()

    # uvr('karaoke_remove_inst', '', 'tmp/uvr5_opt', [inp_path], 'tmp/uvr5_opt', 10, 'wav', f'vocal_{inp_hash}.wav')
    vocal_path, inst_path = run_bsroformer(inp_path, inp_hash, progress=progress)
    deecho_path = f"tmp/uvr5_opt/deecho_{inp_hash}.wav"
    if os.path.exists(deecho_path):
        return deecho_path, inst_path

    wf, sr = torchaudio.load(vocal_path)
    wf = torchaudio.functional.resample(wf, sr, 44100)
    # 写入 tmp/时间戳.wav
    path = f"tmp/{time.time()}.wav"
    torchaudio.save(path, wf, 44100)

    for i in progress.tqdm([1], desc="去混响"):
        uvr(
            "UVR-DeEcho-DeReverb",
            "",
            "tmp/uvr5_opt",
            [path],
            "tmp/uvr5_opt",
            10,
            "wav",
            f"deecho_{inp_hash}.wav",
        )
    return deecho_path, inst_path
