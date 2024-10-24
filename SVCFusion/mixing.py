import os
import numpy as np
import pyroomacoustics as pra
import soundfile as sf
import torch
import torchaudio


def mixAudio(vocal, inst, room_length=7.0, room_width=5.0, room_height=4.0, vocal_db=0):
    if room_width < 1.0 or room_length < 1.0:
        raise ValueError("房间需要大于 1x1")
    corner = np.array(
        [[0, 0], [room_length, 0], [room_length, room_width], [0, room_width]]
    ).T
    room = pra.Room.from_corners(corner)

    corner = np.array(
        [[0, 0], [room_length, 0], [room_length, room_width], [0, room_width]]
    ).T  # 房间的长宽为room_length米，room_width米
    room = pra.Room.from_corners(corner)
    room.extrude(room_height)  # 高为4米的房间

    audio, sr = sf.read(vocal)

    # 如果 audio 是多声道的，将其转换为单声道
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    corners = np.array(
        [[0, 0], [room_length, 0], [room_length, room_width], [0, room_width]]
    ).T
    # corners = np.array([[0, 0], [3.15, 0], [3.15, 3.15], [0, 3.15]]).T
    room1 = pra.Room.from_corners(
        corners,
        fs=sr,
        max_order=3,
        materials=pra.Material(0.2, 0.15),  # 0.2，0.15 / 0.45,0.33
        ray_tracing=True,
        air_absorption=True,
    )
    room1.add_source([1, 1], signal=audio)  # 人的位置

    R = pra.circular_2D_array(center=[2.0, 2.0], M=1, phi0=0, radius=0.3)
    room1.add_microphone_array(pra.MicrophoneArray(R, room1.fs))

    room1.image_source_model()

    room1.plot_rir()
    room1.simulate()

    filename = os.path.basename(vocal)
    filename = f"results/mixed_{filename}.wav"

    sf.write(filename, room1.mic_array.signals.T, samplerate=sr)

    # 混合 filename 和 inst
    mixed_audio, sr = torchaudio.load(filename)
    inst_audio, sr = torchaudio.load(inst)

    mixed_audio = torchaudio.functional.gain(mixed_audio, vocal_db)

    max_len = min(mixed_audio.shape[1], inst_audio.shape[1])
    mixed_audio = mixed_audio[:, :max_len]
    inst_audio = inst_audio[:, :max_len]
    mixed_audio = mixed_audio + inst_audio
    mixed_audio = mixed_audio / torch.max(mixed_audio)

    # 写入新的文件
    torchaudio.save(filename, mixed_audio, sr)
    return filename
