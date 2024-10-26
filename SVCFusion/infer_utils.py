import hashlib
import gradio as gr
import librosa
from SVCFusion.file import getResultFileName, make_dirs

import soundfile as sf

from SVCFusion.models.ddsp import DDSPModel
from SVCFusion.models.reflow import ReflowVAESVCModel

loaded_model: DDSPModel | ReflowVAESVCModel = None


def load_model(device, path, model_type_index):
    global loaded_model

    if model_type_index == 0:
        loaded_model = DDSPModel()
        loaded_model.load_model(device, path)
    elif model_type_index == 1:
        loaded_model = ReflowVAESVCModel()
        loaded_model.load_model(device, path)


def check_model():
    global loaded_model
    return loaded_model is not None and loaded_model.model is not None


def infer(
    audio_path,
    keychange=0,
    method="rk4",
    f0_extractor="fcpe",
    output_path="tmp.wav",
    threhold=-60,
    num_formant_shift_key=0,
    spk=0,
    progress=gr.Progress(),
):
    # load input
    audio, sample_rate = librosa.load(audio_path, sr=None)
    if len(audio.shape) > 1:
        audio = librosa.to_mono(audio)

    # get MD5 hash from wav file
    md5_hash = ""
    with open(audio_path, "rb") as f:
        data = f.read()
        md5_hash = hashlib.md5(data).hexdigest()
        print("MD5: " + md5_hash)

    result, sr, f0 = loaded_model.infer_core(
        audio=audio,
        sample_rate=sample_rate,
        cache_md5=md5_hash,
        keychange=keychange,
        method=method,
        f0_extractor=f0_extractor,
        threhold=threhold,
        num_formant_shift_key=num_formant_shift_key,
        spk=spk,
        progress=progress,
    )
    sf.write(output_path, result, sr)
    return output_path, f0


def batch(
    audio_paths,
    keychange,
    method,
    f0_extractor,
    threhold=-60,
    num_formant_shift_key=0,
    spk=0,
):
    global model, vocoder, args, units_encoder

    make_dirs("results")
    results = []
    for audio_path in audio_paths:
        filename, is_exists = getResultFileName(audio_path)
        if not is_exists:
            infer(
                audio_path=audio_path,
                keychange=keychange,
                method=method,
                f0_extractor=f0_extractor,
                output_path=f"{filename}.wav",
                threhold=threhold,
                num_formant_shift_key=num_formant_shift_key,
                spk=0,
            )
        results.append(f"{filename}.wav")
    return results
