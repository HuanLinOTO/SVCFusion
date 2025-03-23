import os
import shutil

from tqdm import tqdm
from SVCFusion.models.ddsp6_1 import DDSP_6_1Model

dirs = [
    # r"F:\data\aidol\raw\wavs",
    # "F:/data/karasu/raw/wavs",
    # "F:/data/lianhua/wavs",
    "F:/data/weishu/wavs",
]

output_dir = [
    # r"F:\data\liliko_washed\aidol\wavs",
    # "F:/data/liliko_washed/karasu/wavs",
    # "F:/data/liliko_washed/lianhua/wavs",
    "F:/data/liliko_washed/weishu/wavs",
]

model = DDSP_6_1Model()

model.load_model(
    {
        "cascade": "./models/白菜liliko/model_400000.pt",
        "device": "cuda:0",
    }
)

for i, o in zip(dirs, output_dir):
    for d in tqdm(os.listdir(i)):
        os.makedirs(o, exist_ok=True)
        d = os.path.join(i, d)
        print(d)
        if not d.endswith(".wav"):
            continue
        if os.path.exists(os.path.join(o, os.path.basename(d))):
            continue
        audio = model.infer(
            {
                "audio": d,
                "audio_batch": None,
                "use_batch": True,
                "use_vocal_separation": False,
                "use_de_reverb": False,
                "use_harmonic_remove": False,
                "f0": "fcpe",
                "keychange": 0,
                "threshold": -60,
                "method": "euler",
                "infer_step": 50,
                "t_start": 0.7,
                "num_formant_shift_key": 0,
                "spk": "liliko",
                "_model_name": "DDSP-SVC 6.1",
                "hash": "it the hash",
            }
        )
        shutil.copy(audio, os.path.join(o, os.path.basename(d)))
