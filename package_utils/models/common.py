common_infer_form = {
    "audio": {
        "type": "audio",
        "label": "音频文件",
        "info": "需要进行推理的音频文件",
    },
    "f0": {
        "type": "dropdown",
        "info": "用于音高提取/预测的模型",
        "choices": [
            "parselmouth",
            "dio",
            "harvest",
            "crepe",
            "rmvpe",
            "fcpe",
        ],
        "default": "fcpe",
        "label": "f0 提取器",
    },
    "keychange": {
        "type": "slider",
        "max": 20,
        "min": -20,
        "default": 0,
        "step": 1,
        "info": "参考：男转女 12，女转男 -12，音色不像可以调节这个",
        "label": "变调",
    },
    "threshold": {
        "type": "slider",
        "max": 0,
        "min": -100,
        "default": -60,
        "step": 1,
        "label": "切片阈值",
        "info": "人声切片的阈值，如果有底噪可以调为 -40 或更高",
    },
}

diff_based_infer_form = {
    "method": {
        "type": "dropdown",
        "info": "用于 reflow 的采样器",
        "choices": ["euler", "rk4"],
        "default": "euler",
        "label": "采样器",
    },
    "infer_step": {
        "type": "slider",
        "max": 100,
        "min": 1,
        "default": 20,
        "step": 1,
        "label": "推理步数",
        "info": "推理步长，默认就行",
    },
    "t_start": {
        "type": "slider",
        "max": 1,
        "min": 0.1,
        "default": 0.7,
        "step": 0.1,
        "label": "T Start",
        "info": "不知道",
    },
}

common_preprocess_form = {
    "encoder": {
        "type": "dropdown",
        "info": "用于对声音进行编码的模型",
        "choices": [
            # "hubertsoft",
            # "hubertbase",
            # "hubertbase768",
            # "contentvec",
            # "contentvec768",
            "contentvec768l12",
            # "cnhubertsoftfish",
        ],
        "default": "contentvec768l12",
        "label": "声音编码器",
    },
    "f0": {
        "type": "dropdown",
        "info": "用于音高提取/预测的模型",
        "choices": [
            "parselmouth",
            "dio",
            "harvest",
            "crepe",
            "rmvpe",
            "fcpe",
        ],
        "default": "fcpe",
        "label": "f0 提取器",
    },
    "device": {
        "type": "device_chooser",
    },
    # "force_cut": {
    #     "type": "checkbox",
    #     "info": "是否使用强制切片 (看不懂默认就对了)",
    #     "default": True,
    #     "label": "强制切片",
    # },
}

diff_based_preprocess_form = {
    "method": {
        "type": "dropdown",
        "info": "用于 reflow 的采样器",
        "choices": ["euler", "rk4"],
        "default": "euler",
        "label": "f0 提取器",
    }
}
