import asyncio
import io
import os
import sys

import numpy as np
import soundfile
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from SoVITS.inference import infer_tool, slicer

from . import logger

app = FastAPI()

model_name = "logs/44k-1/G_14500.pth"  # 模型地址
config_name = "logs/44k-1/config-an.json"  # config地址
svc_model = infer_tool.Svc(model_name, config_name)


class RequestBody(BaseModel):
    text: str


@app.get("/wav/{id}")
async def get_wav(id):
    if not os.path.exists(f"tmp/tts{id}.wav"):
        return {"status": "error", "message": "wav not found"}
    return FileResponse(f"tmp/tts{id}.wav", media_type="audio/wav", filename="tts.wav")


@app.post("/infer/{spk}")
async def tts(spk, req: RequestBody):
    tran = int(0)  # 音调
    wav_format = "wav"  # 范围文件格式
    text = req.text
    unique_value = hash((text, "Auto", "+0%", "+0%", "Male"))
    audio_path = f"tmp/tts{unique_value}.wav"

    if not os.path.exists("tmp"):
        os.mkdir("tmp")
    if os.path.exists(audio_path):
        return {"status": "success", "src": "/wav/" + str(unique_value)}

    tts_process = await asyncio.create_subprocess_exec(
        sys.executable, "edgetts/tts.py", text, "Auto", "+0%", "+0%", "Male", audio_path
    )
    await tts_process.wait()
    # audio_path = f"tmp/tts-4798745368944844903.wav"

    infer_tool.format_wav(audio_path)
    chunks = slicer.cut(audio_path, db_thresh=-40)
    audio_data, audio_sr = slicer.chunks2audio(audio_path, chunks)

    audio = []
    for slice_tag, data in audio_data:
        logger.info(f"#=====segment start, {round(len(data) / audio_sr, 3)}s======")

        length = int(np.ceil(len(data) / audio_sr * svc_model.target_sample))
        if slice_tag:
            print("jump empty segment")
            _audio = np.zeros(length)
        else:
            # padd
            pad_len = int(audio_sr * 0.5)
            data = np.concatenate([np.zeros([pad_len]), data, np.zeros([pad_len])])
            raw_path = io.BytesIO()
            soundfile.write(raw_path, data, audio_sr, format="wav")
            raw_path.seek(0)
            out_audio, out_sr, fr = svc_model.infer(spk, tran, raw_path)
            svc_model.clear_empty()
            _audio = out_audio.cpu().numpy()
            pad_len = int(svc_model.target_sample * 0.5)
            _audio = _audio[pad_len:-pad_len]

        audio.extend(list(infer_tool.pad_array(_audio, length)))
    out_wav_path = audio_path
    soundfile.write(out_wav_path, audio, svc_model.target_sample, format=wav_format)

    return {"status": "success", "src": "/wav/" + str(unique_value)}
